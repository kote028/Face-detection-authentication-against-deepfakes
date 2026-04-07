from flask import Flask, render_template, Response
import cv2
from face_detect import detect_face
from liveness import check_liveness
from face_auth import compare_faces

app = Flask(__name__)

cap = cv2.VideoCapture(0)

registered_faces = {}
current_user = "User1"
last_face = None


def generate_frames():
    global registered_faces, current_user, last_face

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while True:
        last_face = None

        faces = detect_face(frame1)
        status, texture, movement, brightness, reason, liveness_score = check_liveness(frame1, frame2)

        best_conf = 0
        best_match = None

        for (x, y, w, h) in faces:
            face = frame1[y:y+h, x:x+w]
            face = cv2.resize(face, (100, 100))

            last_face = face

            # Auto register first user (demo)
            if current_user not in registered_faces:
                registered_faces[current_user] = face

            for name, stored in registered_faces.items():
                conf = compare_faces(stored, face)
                if conf > best_conf:
                    best_conf = conf
                    best_match = name

            # Draw box
            color = (0, 255, 0) if status == "REAL" else (0, 0, 255)
            cv2.rectangle(frame1, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame1, status, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Final confidence
        final_conf = int(0.7 * best_conf + 0.3 * liveness_score)

        if final_conf > 70:
            text = f"LOGIN: {best_match}"
            col = (0, 255, 0)
        else:
            text = "UNKNOWN"
            col = (0, 0, 255)

        # UI
        cv2.putText(frame1, f"Confidence: {final_conf}%",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.putText(frame1, text,
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

        cv2.putText(frame1, f"Motion:{movement} Bright:{int(brightness)}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.putText(frame1, f"Texture:{int(texture)}",
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.putText(frame1, reason,
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame1)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        frame1 = frame2
        ret, frame2 = cap.read()


# ================= ROUTES =================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/register')
def register():
    global current_user, last_face

    if last_face is not None:
        registered_faces[current_user] = last_face
        print(f"{current_user} Registered via Web")

    return ("", 204)


@app.route('/delete')
def delete():
    global current_user

    if current_user in registered_faces:
        del registered_faces[current_user]
        print(f"{current_user} Deleted via Web")

    return ("", 204)


# ================= RUN SERVER =================

if __name__ == "__main__":
    print("Starting Flask Server...")
    app.run(host="127.0.0.1", port=5000, debug=True)