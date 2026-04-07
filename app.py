import cv2
from face_detect import detect_face
from liveness import check_liveness
from face_auth import compare_faces

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
ret, frame2 = cap.read()

registered_faces = {}
current_user = "User1"
confirm_delete = False

while True:

    faces = detect_face(frame1)

    # 🔥 XAI + Liveness
    status, texture, movement, brightness, reason, liveness_score = check_liveness(frame1, frame2)

    key = cv2.waitKey(1) & 0xFF

    # 🎮 USER SELECT
    if key == ord('1'):
        current_user = "User1"
    elif key == ord('2'):
        current_user = "User2"
    elif key == ord('3'):
        current_user = "User3"
    elif key == ord('4'):
        current_user = "User4"

    # 🗑 DELETE FLOW
    if key == ord('d'):
        if current_user in registered_faces:
            confirm_delete = True

    if confirm_delete:
        if key == ord('y'):
            if current_user in registered_faces:
                del registered_faces[current_user]
                print(f"{current_user} deleted")
            confirm_delete = False
        elif key == ord('n'):
            confirm_delete = False

    best_conf = 0
    best_match = None

    for (x, y, w, h) in faces:

        current_face = frame1[y:y+h, x:x+w]
        current_face = cv2.resize(current_face, (100, 100))

        # 📌 REGISTER
        if key == ord('r'):
            if len(registered_faces) < 4:
                registered_faces[current_user] = current_face
                print(f"{current_user} Registered ✅")

        # 🔍 MATCH
        for name, stored in registered_faces.items():
            conf = compare_faces(stored, current_face)

            if conf > best_conf:
                best_conf = conf
                best_match = name

        # 🟢 / 🔴 FACE BOX
        box_color = (0,255,0) if status == "REAL" else (0,0,255)
        cv2.rectangle(frame1, (x,y), (x+w,y+h), box_color, 2)
        cv2.putText(frame1, status, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

    # 🔥 FINAL CONFIDENCE (COMBINED)
    final_conf = int(0.7 * best_conf + 0.3 * liveness_score)

    # 🔐 LOGIN LOGIC
    if final_conf > 70:
        login_text = f"LOGIN: {best_match}"
        login_color = (0,255,0)
    else:
        login_text = "UNKNOWN"
        login_color = (0,0,255)

    # ================= UI =================

    cv2.rectangle(frame1, (0, 0), (320, 180), (0,0,0), -1)

    cv2.putText(frame1, f"User: {current_user}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

    user_list = ", ".join(registered_faces.keys()) or "None"
    cv2.putText(frame1, f"Users: {user_list}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    cv2.putText(frame1, "1-4 | R | D",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    if confirm_delete:
        cv2.putText(frame1, "Confirm: Y/N",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # 🔥 CONFIDENCE DISPLAY
    cv2.putText(frame1, f"Confidence: {final_conf}%",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.putText(frame1, login_text,
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, login_color, 2)

    # 🔥 XAI INFO
    cv2.putText(frame1, f"Face:{int(best_conf)} Live:{liveness_score}",
                (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.putText(frame1, reason,
                (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    # =====================================

    # 🔥 HEATMAP (XAI VISUAL)
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    cv2.imshow("XAI Heatmap", heatmap)

    cv2.imshow("Face Authentication", frame1)

    # UPDATE FRAMES
    frame1 = frame2
    ret, frame2 = cap.read()

    # EXIT
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()