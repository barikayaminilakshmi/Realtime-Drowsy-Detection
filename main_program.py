import cv2
import mediapipe as mp
import time
import os
import threading

from playsound import playsound
from utils import eye_aspect_ratio, mouth_aspect_ratio

# ================== SETTINGS ==================
EAR_THRESHOLD = 0.20
EYE_CLOSED_FRAMES = 15

MAR_THRESHOLD = 0.60
YAWN_FRAMES = 10

ALARM_COOLDOWN_SEC = 2.0  # play alarm once every 2 seconds
# ==============================================

# Alarm file path (must be in same folder as this script)
ALARM_FILE = os.path.join(os.path.dirname(__file__), "awake.mp3")

def play_alarm_non_blocking(path: str):
    # playsound blocks, so we run it in a background thread (smooth camera)
    threading.Thread(target=playsound, args=(path,), daemon=True).start()

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# FaceMesh landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 308, 13, 14]  # left, right, top, bottom

eye_counter = 0
yawn_counter = 0
alarm_on = False
last_alarm_time = 0.0

def lm_to_point(lm, w, h):
    return (int(lm.x * w), int(lm.y * h))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera open avvaledhu. Try: cv2.VideoCapture(1)")
    raise SystemExit

# Quick check: alarm file exists
if not os.path.exists(ALARM_FILE):
    print(f"❌ Audio file not found: {ALARM_FILE}")
    print("➡️ Put awake.mp3 in the same folder as main_program.py")
    raise SystemExit

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    status = "NO FACE"
    ear = 0.0
    mar = 0.0

    if res.multi_face_landmarks:
        status = "NORMAL"
        face = res.multi_face_landmarks[0]

        left_eye_pts = [lm_to_point(face.landmark[i], w, h) for i in LEFT_EYE]
        right_eye_pts = [lm_to_point(face.landmark[i], w, h) for i in RIGHT_EYE]
        mouth_pts = [lm_to_point(face.landmark[i], w, h) for i in MOUTH]

        # draw small dots (optional)
        for p in left_eye_pts + right_eye_pts + mouth_pts:
            cv2.circle(frame, p, 2, (0, 255, 0), -1)

        # Compute EAR (avg both eyes)
        ear_left = eye_aspect_ratio(left_eye_pts)
        ear_right = eye_aspect_ratio(right_eye_pts)
        ear = (ear_left + ear_right) / 2.0

        # Compute MAR
        mar = mouth_aspect_ratio(mouth_pts)

        # ---------------- Eye-closed detection ----------------
        if ear < EAR_THRESHOLD:
            eye_counter += 1
            if eye_counter >= EYE_CLOSED_FRAMES:
                status = "DROWSY (EYES CLOSED)"
                alarm_on = True
        else:
            eye_counter = 0
            alarm_on = False

        # ---------------- Yawning detection ----------------
        if mar > MAR_THRESHOLD:
            yawn_counter += 1
            if yawn_counter >= YAWN_FRAMES:
                status = "YAWNING"
        else:
            yawn_counter = 0

        # ---------------- Alarm action ----------------
        now = time.time()
        if alarm_on and (now - last_alarm_time) > ALARM_COOLDOWN_SEC:
            last_alarm_time = now
            play_alarm_non_blocking(ALARM_FILE)

    # UI overlay
    cv2.putText(frame, f"STATUS: {status}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
    cv2.putText(frame, f"EAR: {ear:.3f} (thr={EAR_THRESHOLD})", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"MAR: {mar:.3f} (thr={MAR_THRESHOLD})", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Drowsiness Detection (Press Q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()