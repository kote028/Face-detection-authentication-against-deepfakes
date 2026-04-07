def check_liveness(frame1, frame2):
    import cv2
    import numpy as np

    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    movement = cv2.countNonZero(gray)

    gray_face = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    texture = cv2.Laplacian(gray_face, cv2.CV_64F).var()

    brightness = np.mean(gray_face)

    # 🔥 Liveness score (0–100)
    movement_score = min(movement / 10, 100)
    texture_score = min(texture / 2, 100)

    liveness_score = int((movement_score + texture_score) / 2)

    # Decision
    if liveness_score > 40:
        status = "REAL"
        reason = "Natural motion + texture"
    else:
        status = "FAKE"
        reason = "Low motion or flat texture"

    return status, texture, movement, brightness, reason, liveness_score