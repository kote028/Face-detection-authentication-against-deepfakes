import cv2
import numpy as np

def extract_face(frame, face):
    x, y, w, h = face
    face_img = frame[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, (100, 100))
    return face_img

def compare_faces(face1, face2):
    face1 = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
    face2 = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(face1, face2)
    score = np.mean(diff)

    confidence = max(0, 100 - score)

    return confidence