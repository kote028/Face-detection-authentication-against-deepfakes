"""
app_web.py — Flask Web Interface for Face Authentication Pro
------------------------------------------------------------
Routes:
  GET  /              — main dashboard page
  GET  /video         — MJPEG stream
  POST /set_user      — switch active user slot
  POST /register      — register current face for active user
  POST /delete        — delete active user's registration
  GET  /status        — JSON snapshot of current auth state
  GET  /log           — last 20 lines of audit log
"""

import os
import json
import datetime
from collections import defaultdict
from flask import Flask, render_template_string, Response, jsonify, request
import cv2

from face_detect import detect_face, get_face_roi
from face_auth   import encode_face, compare_faces
from liveness    import check_liveness, BlinkTracker


# ─────────────────────────────────────────
#  SETUP
# ─────────────────────────────────────────

app = Flask(__name__)

LOG_DIR  = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "auth_log.txt")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def log_event(event: str):
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {event}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ─────────────────────────────────────────
#  SHARED STATE  (protected by GIL in CPython)
# ─────────────────────────────────────────

state = {
    "registered":       {},          # name → LBP descriptor
    "blink_trackers":   defaultdict(BlinkTracker),
    "challenge_passed": defaultdict(bool),
    "current_user":     "User1",
    "last_face":        None,        # raw pixel ROI for registration
    "snapshot": {                    # updated every frame, served via /status
        "status":     "—",
        "liveness":   0,
        "confidence": 0,
        "login":      "—",
        "reason":     "—",
        "scores":     {},
    }
}


# ─────────────────────────────────────────
#  FRAME GENERATOR
# ─────────────────────────────────────────

def generate_frames():
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    while True:
        faces   = detect_face(frame1)
        cu      = state["current_user"]
        tracker = state["blink_trackers"][cu]

        primary_box = faces[0] if len(faces) > 0 else None

        live_status, scores, brightness, reason, liveness_score = check_liveness(
            frame1, frame2,
            blink_tracker=tracker,
            face_box=primary_box
        )

        if scores.get("blink", 0) >= 80:
            state["challenge_passed"][cu] = True

        best_conf  = 0.0
        best_match = None

        for face_box in faces:
            x, y, w, h = face_box
            face_roi = get_face_roi(frame1, face_box)
            state["last_face"] = face_roi

            # Auto-register on first detection (demo mode)
            if cu not in state["registered"]:
                state["registered"][cu] = encode_face(face_roi)
                state["challenge_passed"][cu] = False
                state["blink_trackers"][cu]   = BlinkTracker()
                log_event(f"AUTO-REGISTERED: {cu}")

            for name, desc in state["registered"].items():
                conf = compare_faces(desc, face_roi)
                if conf > best_conf:
                    best_conf  = conf
                    best_match = name

            color = (0, 220, 80) if live_status == "REAL" else (0, 50, 220)
            cv2.rectangle(frame1, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame1, live_status, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # Challenge gate
        eff_live = liveness_score if state["challenge_passed"][cu] else liveness_score * 0.4
        final_conf = int(0.65 * best_conf + 0.35 * eff_live)

        if final_conf > 70 and live_status == "REAL" and state["challenge_passed"][cu]:
            login_text  = f"LOGIN: {best_match}"
            login_color = (0, 220, 80)
        elif final_conf > 70 and not state["challenge_passed"][cu]:
            login_text  = "BLINK TO CONFIRM"
            login_color = (0, 180, 255)
        else:
            login_text  = "UNKNOWN / DENIED"
            login_color = (0, 50, 220)

        # Overlay minimal info
        cv2.putText(frame1, f"Confidence: {final_conf}%",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame1, login_text,
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, login_color, 2)
        cv2.putText(frame1, reason,
                    (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 210, 210), 1)

        # Update state snapshot
        state["snapshot"].update({
            "status":     live_status,
            "liveness":   liveness_score,
            "confidence": final_conf,
            "login":      login_text,
            "reason":     reason,
            "scores":     {k: round(v, 1) for k, v in scores.items()},
            "registered": list(state["registered"].keys()),
            "current_user": cu,
            "challenge":  state["challenge_passed"].get(cu, False),
        })

        ret, buffer = cv2.imencode('.jpg', frame1)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        frame1 = frame2
        ret, frame2 = cap.read()


# ─────────────────────────────────────────
#  HTML TEMPLATE
# ─────────────────────────────────────────

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Face Auth Pro</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0d0d0d; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; display: flex; flex-direction: column; align-items: center; padding: 20px; }
    h1  { color: #00cfff; margin-bottom: 16px; letter-spacing: 2px; }
    .layout { display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; justify-content: center; }
    #feed { border: 2px solid #1e1e1e; border-radius: 6px; }
    .panel { background: #151515; border: 1px solid #222; border-radius: 8px; padding: 18px; width: 280px; }
    .panel h3 { color: #00cfff; margin-bottom: 12px; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 1px; }
    .badge { display: inline-block; padding: 3px 10px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; margin-bottom: 8px; }
    .REAL { background: #003d20; color: #00e676; }
    .FAKE { background: #3d0000; color: #ff5252; }
    .bar-row { margin: 6px 0; }
    .bar-label { font-size: 0.75rem; display: flex; justify-content: space-between; margin-bottom: 2px; }
    .bar-bg { background: #2a2a2a; border-radius: 4px; height: 8px; }
    .bar-fill { height: 8px; border-radius: 4px; transition: width 0.3s; }
    .conf-big { font-size: 2.2rem; font-weight: bold; text-align: center; margin: 8px 0; }
    .login-text { text-align: center; font-size: 1rem; font-weight: bold; padding: 8px; border-radius: 6px; margin-bottom: 10px; }
    button { background: #00cfff22; border: 1px solid #00cfff55; color: #00cfff; padding: 8px 14px; border-radius: 6px; cursor: pointer; margin: 4px 2px; font-size: 0.82rem; transition: background 0.2s; }
    button:hover { background: #00cfff44; }
    button.danger { border-color: #ff525555; color: #ff5252; background: #ff525211; }
    button.danger:hover { background: #ff525233; }
    select { background: #1e1e1e; color: #e0e0e0; border: 1px solid #333; padding: 6px 10px; border-radius: 6px; width: 100%; margin-bottom: 8px; }
    .log-box { background: #0a0a0a; border: 1px solid #1e1e1e; border-radius: 6px; padding: 10px; font-size: 0.7rem; font-family: monospace; color: #aaa; max-height: 180px; overflow-y: auto; white-space: pre-wrap; }
    .tag { font-size: 0.7rem; color: #888; margin-top: 6px; }
  </style>
</head>
<body>
  <h1>&#128274; FACE AUTH PRO</h1>
  <div class="layout">
    <img id="feed" src="/video" width="500" alt="Live feed">

    <div style="display:flex;flex-direction:column;gap:16px;">

      <!-- AUTH STATUS -->
      <div class="panel">
        <h3>Auth Status</h3>
        <div id="liveness-badge" class="badge">—</div>
        <div id="reason" class="tag">—</div>
        <div class="conf-big" id="conf">0%</div>
        <div class="login-text" id="login-text">—</div>

        <!-- Factor bars -->
        <div id="bars"></div>
      </div>

      <!-- CONTROLS -->
      <div class="panel">
        <h3>Controls</h3>
        <label style="font-size:0.8rem;color:#aaa;">Active User Slot</label>
        <select id="user-sel" onchange="setUser(this.value)">
          <option>User1</option><option>User2</option>
          <option>User3</option><option>User4</option>
        </select>
        <div id="reg-list" style="font-size:0.75rem;color:#0ff;margin-bottom:8px;">Registered: —</div>
        <button onclick="doRegister()">&#9654; Register Face</button>
        <button class="danger" onclick="doDelete()">&#128465; Delete User</button>
        <div class="tag" id="challenge-status"></div>
      </div>

      <!-- LOG -->
      <div class="panel">
        <h3>Audit Log</h3>
        <div class="log-box" id="log-box">Loading…</div>
      </div>

    </div>
  </div>

<script>
const FACTOR_COLORS = {
  motion:    '#50b4ff',
  texture:   '#50ffc8',
  blink:     '#ffc850',
  frequency: '#c850ff',
  skin:      '#50ff64',
};

function setUser(u) {
  fetch('/set_user', { method:'POST', headers:{'Content-Type':'application/json'},
                       body: JSON.stringify({user: u}) });
}
function doRegister() {
  fetch('/register', { method:'POST' }).then(() => pollStatus());
}
function doDelete() {
  if (confirm('Delete this user?'))
    fetch('/delete', { method:'POST' }).then(() => pollStatus());
}

function pollStatus() {
  fetch('/status').then(r => r.json()).then(d => {
    // Badge
    const badge = document.getElementById('liveness-badge');
    badge.textContent = d.status;
    badge.className   = 'badge ' + d.status;

    document.getElementById('reason').textContent    = d.reason;
    document.getElementById('conf').textContent      = d.confidence + '%';
    document.getElementById('conf').style.color      = d.confidence > 70 ? '#00e676' : '#ff5252';
    document.getElementById('login-text').textContent = d.login;
    document.getElementById('reg-list').textContent  = 'Registered: ' + (d.registered || []).join(', ') || 'None';
    document.getElementById('challenge-status').textContent =
      d.challenge ? '✅ Blink challenge passed' : '👁 Please blink to confirm identity';

    // Factor bars
    const bars = document.getElementById('bars');
    bars.innerHTML = '';
    for (const [k, v] of Object.entries(d.scores || {})) {
      const col = FACTOR_COLORS[k] || '#aaa';
      bars.innerHTML += `
        <div class="bar-row">
          <div class="bar-label"><span>${k}</span><span>${Math.round(v)}</span></div>
          <div class="bar-bg"><div class="bar-fill" style="width:${v}%;background:${col}"></div></div>
        </div>`;
    }
  });
}

function pollLog() {
  fetch('/log').then(r => r.json()).then(d => {
    const box = document.getElementById('log-box');
    box.textContent = d.lines.join('\\n');
    box.scrollTop   = box.scrollHeight;
  });
}

setInterval(pollStatus, 500);
setInterval(pollLog,    2000);
pollStatus(); pollLog();
</script>
</body>
</html>
"""


# ─────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_user', methods=['POST'])
def set_user():
    data = request.get_json(silent=True) or {}
    state["current_user"] = data.get("user", "User1")
    return ("", 204)


@app.route('/register', methods=['POST'])
def register():
    cu   = state["current_user"]
    face = state["last_face"]
    if face is not None:
        state["registered"][cu]       = encode_face(face)
        state["challenge_passed"][cu] = False
        state["blink_trackers"][cu]   = BlinkTracker()
        log_event(f"REGISTERED via web: {cu}")
    return ("", 204)


@app.route('/delete', methods=['POST'])
def delete():
    cu = state["current_user"]
    if cu in state["registered"]:
        del state["registered"][cu]
        state["challenge_passed"][cu] = False
        log_event(f"DELETED via web: {cu}")
    return ("", 204)


@app.route('/status')
def status():
    return jsonify(state["snapshot"])


@app.route('/log')
def get_log():
    try:
        with open(LOG_FILE) as f:
            lines = f.readlines()[-20:]
    except FileNotFoundError:
        lines = []
    return jsonify({"lines": [l.rstrip() for l in lines]})


# ─────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    log_event("Web server started")
    app.run(host="127.0.0.1", port=5000, debug=False)