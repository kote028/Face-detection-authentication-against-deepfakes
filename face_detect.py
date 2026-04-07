"""
face_detect.py — Face + Eye detection module
Uses Haar cascades for face and eye detection.
Eye detection is used for blink-based liveness challenge.
"""

import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)


def detect_face(frame):
    """Detect faces in a frame. Returns list of (x, y, w, h)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)      # ignore tiny detections
    )
    return faces if len(faces) > 0 else []


def detect_eyes(face_roi_gray):
    """
    Detect eyes within a face ROI (grayscale).
    Returns number of eyes found (0 = likely blinking, 2 = eyes open).
    """
    eyes = eye_cascade.detectMultiScale(
        face_roi_gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(20, 20)
    )
    return len(eyes)


def get_face_roi(frame, face_box):
    """Extract and resize a face ROI from a frame."""
    x, y, w, h = face_box
    roi = frame[y:y + h, x:x + w]
    roi = cv2.resize(roi, (100, 100))
    return roi