"""
app.py — Face Authentication with Anti-Deepfake Liveness (Desktop/OpenCV)
--------------------------------------------------------------------------
NEW IN THIS VERSION:
  • LBP histogram face matching (lighting-robust)
  • 5-factor liveness: motion, texture, blink, DCT frequency, skin tone
  • Challenge-Response gate: user must blink before login is accepted
  • Per-user BlinkTracker instances
  • Audit log (logs/auth_log.txt) — timestamps every auth event
  • Cleaner, colour-coded XAI dashboard on screen

CONTROLS:
  1–4  : switch active user slot
  R    : register current face in active slot
  D    : begin delete (confirm with Y / cancel with N)
  ESC  : quit
"""

import cv2
import os
import datetime
from collections import defaultdict

from face_detect import detect_face, get_face_roi
from face_auth   import encode_face, compare_faces
from liveness    import check_liveness, BlinkTracker


# ─────────────────────────────────────────
#  SETUP
# ─────────────────────────────────────────

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "auth_log.txt")


def log_event(event: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {event}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ─────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ret, frame1 = cap.read()
ret, frame2 = cap.read()

# Stores LBP descriptors (not raw pixels)
registered_descriptors: dict[str, any] = {}

# One BlinkTracker per user slot
blink_trackers: dict[str, BlinkTracker] = defaultdict(BlinkTracker)

# Challenge-response: user must blink at least once before login
challenge_passed: dict[str, bool] = defaultdict(bool)

current_user   = "User1"
confirm_delete = False

# Running display state
last_status    = "—"
last_reason    = "No face detected"
last_conf      = 0
last_login     = "—"
last_scores    = {}


# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────

def draw_bar(frame, label, value, x, y, width=120, height=10, color=(0, 200, 100)):
    """Draw a labelled progress bar on frame."""
    cv2.rectangle(frame, (x, y), (x + width, y + height), (60, 60, 60), -1)
    filled = int(width * max(0, min(value, 100)) / 100)
    cv2.rectangle(frame, (x, y), (x + filled, y + height), color, -1)
    cv2.putText(frame, f"{label}: {int(value)}",
                (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)


def draw_panel(frame, registered, current_user, confirm_delete,
               status, scores, reason, final_conf, login_text, login_color):
    """Draw the left HUD panel."""
    panel_w, panel_h = 210, frame.shape[0]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    y = 20
    # Title
    cv2.putText(frame, "FACE AUTH PRO", (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
    y += 22
    cv2.line(frame, (8, y), (panel_w - 8, y), (60, 60, 60), 1)
    y += 14

    # Current user
    cv2.putText(frame, f"Slot : {current_user}", (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 230, 80), 1)
    y += 16
    user_list = ", ".join(registered.keys()) or "None"
    cv2.putText(frame, f"Reg  : {user_list}", (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.37, (0, 210, 210), 1)
    y += 16
    cv2.putText(frame, "1-4:slot  R:reg  D:del",
                (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    y += 20

    if confirm_delete:
        cv2.putText(frame, "!! Confirm delete? Y/N",
                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 50, 255), 1)
        y += 20

    cv2.line(frame, (8, y), (panel_w - 8, y), (60, 60, 60), 1)
    y += 12

    # Liveness status
    stat_color = (0, 220, 80) if status == "REAL" else (0, 50, 220)
    cv2.putText(frame, f"Liveness: {status}", (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, stat_color, 1)
    y += 16
    cv2.putText(frame, reason, (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 210, 210), 1)
    y += 16

    # Factor bars
    factor_colors = {
        "motion":    (80, 180, 255),
        "texture":   (80, 255, 180),
        "blink":     (255, 200, 50),
        "frequency": (200, 80, 255),
        "skin":      (80, 255, 100),
    }
    for factor, val in scores.items():
        col = factor_colors.get(factor, (200, 200, 200))
        draw_bar(frame, factor, val, 8, y + 6, width=140, color=col)
        y += 22
    y += 6

    cv2.line(frame, (8, y), (panel_w - 8, y), (60, 60, 60), 1)
    y += 12

    # Confidence + login result
    cv2.putText(frame, f"Confidence: {final_conf}%", (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    y += 20
    cv2.putText(frame, login_text, (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, login_color, 2)


# ─────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────

log_event("System started")

while True:
    faces  = detect_face(frame1)
    key    = cv2.waitKey(1) & 0xFF

    # ── USER SLOT SELECT ──────────────────────────────────────────────
    slot_keys = {ord('1'): "User1", ord('2'): "User2",
                 ord('3'): "User3", ord('4'): "User4"}
    if key in slot_keys:
        current_user = slot_keys[key]

    # ── DELETE FLOW ───────────────────────────────────────────────────
    if key == ord('d') and current_user in registered_descriptors:
        confirm_delete = True
    if confirm_delete:
        if key == ord('y'):
            del registered_descriptors[current_user]
            challenge_passed[current_user] = False
            log_event(f"DELETED: {current_user}")
            confirm_delete = False
        elif key == ord('n'):
            confirm_delete = False

    # ── LIVENESS (uses first detected face box if any) ────────────────
    primary_box = faces[0] if len(faces) > 0 else None
    tracker     = blink_trackers[current_user]

    status, scores, brightness, reason, liveness_score = check_liveness(
        frame1, frame2,
        blink_tracker=tracker,
        face_box=primary_box
    )

    # Challenge gate: once blink score >= 80, consider challenge passed
    if scores.get("blink", 0) >= 80:
        challenge_passed[current_user] = True

    # ── PROCESS EACH FACE ─────────────────────────────────────────────
    best_conf  = 0.0
    best_match = None

    for face_box in faces:
        x, y, w, h = face_box
        face_roi = get_face_roi(frame1, face_box)

        # REGISTER
        if key == ord('r'):
            if len(registered_descriptors) < 4:
                descriptor = encode_face(face_roi)
                registered_descriptors[current_user] = descriptor
                challenge_passed[current_user] = False   # reset challenge
                blink_trackers[current_user]   = BlinkTracker()
                log_event(f"REGISTERED: {current_user}")

        # MATCH
        for name, stored_desc in registered_descriptors.items():
            conf = compare_faces(stored_desc, face_roi)
            if conf > best_conf:
                best_conf  = conf
                best_match = name

        # FACE BOX COLOUR
        box_color = (0, 220, 80) if status == "REAL" else (0, 50, 220)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(frame1, status, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2)

    # ── FINAL CONFIDENCE ──────────────────────────────────────────────
    # Challenge-response gate: if user hasn't blinked, cap confidence
    if not challenge_passed.get(current_user, False):
        effective_liveness = liveness_score * 0.4   # penalise
    else:
        effective_liveness = liveness_score

    final_conf = int(0.65 * best_conf + 0.35 * effective_liveness)

    # ── LOGIN DECISION ────────────────────────────────────────────────
    if final_conf > 70 and status == "REAL" and challenge_passed.get(current_user, False):
        login_text  = f"LOGIN: {best_match}"
        login_color = (0, 220, 80)
        log_event(f"LOGIN SUCCESS: {best_match} (conf={final_conf}%)")
    elif final_conf > 70 and not challenge_passed.get(current_user, False):
        login_text  = "BLINK TO CONFIRM"
        login_color = (0, 180, 255)
    else:
        login_text  = "UNKNOWN / DENIED"
        login_color = (0, 50, 220)

    # ── DRAW HUD ──────────────────────────────────────────────────────
    draw_panel(
        frame1,
        registered_descriptors,
        current_user,
        confirm_delete,
        status, scores, reason,
        final_conf,
        login_text, login_color
    )

    # ── XAI HEATMAP (separate window) ─────────────────────────────────
    gray    = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    cv2.putText(heatmap, "XAI: Activation Heatmap",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("XAI Heatmap", heatmap)

    cv2.imshow("Face Authentication Pro", frame1)

    # ── FRAME ADVANCE ─────────────────────────────────────────────────
    frame1 = frame2
    ret, frame2 = cap.read()

    if key == 27:   # ESC
        log_event("System stopped")
        break

cap.release()
cv2.destroyAllWindows()