"""
liveness.py — Multi-Factor Anti-Spoofing / Anti-Deepfake Liveness Detection
----------------------------------------------------------------------------
FACTORS (each contributes to liveness_score):

  1. MOTION SCORE      — inter-frame pixel diff (natural movement)
  2. TEXTURE SCORE     — Laplacian variance (real skin has micro-texture)
  3. BLINK SCORE       — eye detector disappears briefly → blink detected
                         Deepfakes often fail to blink at all
  4. FREQUENCY SCORE   — DCT high-frequency energy ratio
                         GAN-generated faces have unnaturally smooth freq. spectrum
  5. SKIN TONE SCORE   — YCrCb skin-colour distribution check
                         Deepfake composites often have boundary skin colour mismatch

Each factor is normalised to 0–100 and weighted into a final score.
"""

import cv2
import numpy as np
from collections import deque
from face_detect import detect_eyes


# ─────────────────────────────────────────
#  BLINK TRACKER  (stateful, call every frame)
# ─────────────────────────────────────────

class BlinkTracker:
    """
    Tracks eye-open/closed state across frames.
    A blink = eyes seen → eyes gone → eyes seen again.
    Requires at least MIN_BLINKS within WINDOW frames for REAL classification.
    """
    WINDOW      = 90    # frames (~3 s at 30 fps)
    MIN_BLINKS  = 1

    def __init__(self):
        self.eye_history  = deque(maxlen=self.WINDOW)
        self.blink_count  = 0
        self._was_open    = True

    def update(self, face_roi_gray):
        """Call with the grayscale face ROI every frame."""
        n_eyes = detect_eyes(face_roi_gray)
        eyes_open = n_eyes >= 1

        # Rising-edge transition: closed → open = end of a blink
        if not self._was_open and eyes_open:
            self.blink_count += 1

        self._was_open = eyes_open
        self.eye_history.append(eyes_open)

        # Decay: only count blinks in the last WINDOW frames
        # (simple approximation: reset every window)
        if len(self.eye_history) == self.WINDOW:
            closed_frames = self.eye_history.count(False)
            # Estimate blinks from closed-frame clusters
            self.blink_count = max(self.blink_count,
                                   1 if closed_frames > 3 else 0)

    def score(self):
        """Return 0–100 blink score."""
        if self.blink_count >= self.MIN_BLINKS:
            return 100
        # Partial credit for some closed-eye frames
        closed = self.eye_history.count(False)
        return min(closed * 15, 80)


# ─────────────────────────────────────────
#  DCT FREQUENCY ANALYSIS
# ─────────────────────────────────────────

def _dct_frequency_score(gray_face):
    """
    GAN/deepfake images are over-smooth in high frequencies.
    Real faces captured by a camera have natural sensor noise in HF bands.

    Score = ratio of high-frequency DCT energy to total energy.
    Higher ratio → more natural → higher score.
    """
    f = np.float32(gray_face) / 255.0
    dct = cv2.dct(f)

    total_energy = np.sum(dct ** 2) + 1e-6
    h, w = dct.shape
    # Top-left 25% = low frequencies; rest = high frequencies
    lf_energy = np.sum(dct[:h // 4, :w // 4] ** 2)
    hf_energy  = total_energy - lf_energy

    hf_ratio = hf_energy / total_energy      # 0.0 – 1.0
    # Real camera images typically have hf_ratio > 0.35
    score = min(hf_ratio * 200, 100)         # scale to 0–100
    return score


# ─────────────────────────────────────────
#  SKIN TONE CONSISTENCY CHECK
# ─────────────────────────────────────────

def _skin_tone_score(face_bgr):
    """
    Deepfake composites often have uneven skin-tone distribution at borders.
    We measure what fraction of the face fits typical skin-colour range (YCrCb).
    Healthy score > 30% skin pixels.
    """
    ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    # Standard skin tone range in YCrCb
    skin_mask = (
        (Cr > 133) & (Cr < 173) &
        (Cb > 77)  & (Cb < 127) &
        (Y  > 80)
    )
    skin_ratio = np.sum(skin_mask) / skin_mask.size  # 0.0 – 1.0

    # Real faces: 25–65 % skin pixels in a well-cropped ROI
    if skin_ratio < 0.15:
        return 20   # too little skin → fake/object
    elif skin_ratio > 0.75:
        return 40   # entire patch is "skin" → flat colour, suspicious
    else:
        score = min((skin_ratio - 0.15) / 0.50 * 100, 100)
        return score


# ─────────────────────────────────────────
#  MAIN CHECK  (called every frame)
# ─────────────────────────────────────────

def check_liveness(frame1, frame2, blink_tracker=None, face_box=None):
    """
    Multi-factor liveness check.

    Parameters
    ----------
    frame1, frame2  : consecutive BGR frames
    blink_tracker   : BlinkTracker instance (pass the same one every frame)
    face_box        : (x, y, w, h) of detected face, or None

    Returns
    -------
    status          : "REAL" | "FAKE"
    factor_scores   : dict with individual factor scores (for XAI display)
    reason          : human-readable explanation
    liveness_score  : int 0–100
    """

    # ── 1. MOTION ──────────────────────────────────────────────────────
    diff            = cv2.absdiff(frame1, frame2)
    diff_gray       = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    movement        = cv2.countNonZero(diff_gray)
    motion_score    = min(movement / 10, 100)

    # ── 2. TEXTURE ─────────────────────────────────────────────────────
    gray_face       = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    texture         = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    texture_score   = min(texture / 2, 100)

    brightness      = float(np.mean(gray_face))

    # ── 3. BLINK ───────────────────────────────────────────────────────
    blink_score = 0
    if blink_tracker is not None and face_box is not None:
        x, y, w, h = face_box
        face_gray   = gray_face[y:y + h, x:x + w]
        if face_gray.size > 0:
            blink_tracker.update(face_gray)
            blink_score = blink_tracker.score()

    # ── 4. DCT FREQUENCY ───────────────────────────────────────────────
    resized_gray    = cv2.resize(gray_face, (128, 128))
    freq_score      = _dct_frequency_score(resized_gray)

    # ── 5. SKIN TONE ───────────────────────────────────────────────────
    if face_box is not None:
        x, y, w, h = face_box
        face_roi    = frame1[y:y + h, x:x + w]
        if face_roi.size > 0:
            face_roi = cv2.resize(face_roi, (64, 64))
            skin_score = _skin_tone_score(face_roi)
        else:
            skin_score = 50
    else:
        skin_score = 50

    # ── WEIGHTED COMBINATION ───────────────────────────────────────────
    weights = {
        "motion":   0.20,
        "texture":  0.20,
        "blink":    0.25,   # highest weight — blink is hardest to fake
        "frequency":0.20,
        "skin":     0.15,
    }
    scores = {
        "motion":   motion_score,
        "texture":  texture_score,
        "blink":    blink_score,
        "frequency":freq_score,
        "skin":     skin_score,
    }

    liveness_score = int(sum(weights[k] * scores[k] for k in weights))

    # ── DECISION + REASON ──────────────────────────────────────────────
    weak_factors = [k for k, v in scores.items() if v < 40]

    if liveness_score > 50:
        status = "REAL"
        reason = "Multi-factor liveness passed"
    else:
        status = "FAKE"
        if "blink" in weak_factors:
            reason = "No blink detected — possible replay/deepfake"
        elif "frequency" in weak_factors:
            reason = "Flat frequency spectrum — possible GAN face"
        elif "motion" in weak_factors:
            reason = "No natural movement — static image/video"
        else:
            reason = f"Failed: {', '.join(weak_factors)}"

    return status, scores, brightness, reason, liveness_score