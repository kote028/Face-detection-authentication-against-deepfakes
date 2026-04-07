# Face Authentication Pro — Anti-Deepfake System

## What's New vs. Original

| Feature | Original | This Version |
|---|---|---|
| Face matching | Raw pixel diff (`absdiff`) | **LBP histogram** (Chi-squared) |
| Liveness factors | Motion + Texture | **5-factor**: motion, texture, blink, DCT frequency, skin tone |
| Anti-deepfake | None | **DCT frequency analysis** catches GAN-generated faces |
| Challenge-response | None | **Blink gate** — must blink before login is accepted |
| Audit trail | None | **Timestamped log** in `logs/auth_log.txt` |
| Web UI | Basic MJPEG | **Real-time dashboard** with factor bars + live log |
| Registration storage | Raw pixel array | **LBP descriptor** (compact + fast) |

---

## How Each Factor Works

### 1. LBP Histogram Face Matching (`face_auth.py`)
Local Binary Patterns encode *texture relationships* between pixels, not raw values.
This makes matching robust to lighting changes, minor pose differences, and JPEG compression.
The face is divided into a 4×4 grid and a histogram is built per cell — capturing both global structure and local detail.

### 2. Motion Score (`liveness.py`)
Inter-frame absolute difference. Real faces move naturally; static images/printed photos have zero motion.

### 3. Texture Score (`liveness.py`)
Laplacian variance measures micro-texture sharpness. Real skin has complex texture; printed/displayed images are flatter.

### 4. Blink Detection / Challenge-Response (`liveness.py`, `app.py`)
An eye cascade detector tracks eye presence per frame.
A blink = eyes detected → not detected → detected again.
The system requires at least one detected blink before granting login — this is the hardest condition for replay attacks and deepfakes to satisfy in real-time.

### 5. DCT Frequency Analysis (`liveness.py`)
GAN-generated deepfake faces are trained to minimise perceptual loss, which makes them unnaturally smooth in the high-frequency DCT domain.
Real camera-captured faces have sensor noise and natural texture in high frequencies.
We measure the ratio of high-frequency energy to total DCT energy.

### 6. Skin Tone Consistency (`liveness.py`)
Deepfake composites often have unnatural colour distributions at face boundaries.
We check what fraction of the face ROI falls within the known YCrCb skin colour range.

---

## Running

```bash
pip install opencv-python flask numpy

# Desktop (OpenCV window)
python app.py

# Web dashboard
python app_web.py
# Then open http://127.0.0.1:5000
```

## Controls (Desktop)
| Key | Action |
|-----|--------|
| 1–4 | Switch user slot |
| R | Register current face |
| D then Y | Delete registered user |
| ESC | Quit |

## Controls (Web)
Use the dropdown to select a user slot, then Register / Delete buttons.
The dashboard auto-refreshes every 500 ms.

---

## Final Confidence Formula

```
effective_liveness = liveness_score           (if blink challenge passed)
                   = liveness_score * 0.4     (if not yet blinked)

final_confidence = 0.65 × face_match + 0.35 × effective_liveness
```

Login is granted only when:
- `final_confidence > 70`
- `liveness_status == "REAL"`
- Blink challenge has been passed
