# src/face_locking.py
"""
Face Locking extension on recognize.py.
- Locks to one hardcoded identity (e.g., "Gabi").
- Tracks stably with centroid + timeout.
- Detects actions: left/right move (nose x delta), blink (EAR), smile (MAR).
- Logs history to data/history/<name>_history_<timestamp>.txt.
Run: python -m src.face_locking
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import paho.mqtt.client as mqtt

try:
    import mediapipe as mp
except Exception as e:
    mp = None
    _MP_IMPORT_ERROR = e

# Reuse from recognize.py
from .haar_5pt import align_face_5pt
from .recognize import HaarFaceMesh5pt, ArcFaceEmbedderONNX, FaceDBMatcher, FaceDet, MatchResult, cosine_distance

# New for full FaceMesh actions (use same mp as in haar_5pt)
mp_face_mesh = mp.solutions.face_mesh

# Config
TARGET_IDENTITY = "Babou"  # Change to your enrolled name
LOCK_TIMEOUT_FRAMES = 10
MOVE_THRESHOLD_PIX = 20
EAR_BLINK_THRESHOLD = 0.2
MAR_SMILE_THRESHOLD = 0.30     # Lowered for more sensitivity
SMILE_WIDTH_THRESHOLD = 1.15   # Mouth width / Eye distance
HISTORY_DIR = Path("data") / "history"

# MQTT Config
MQTT_BROKER = "157.173.101.159"
MQTT_PORT = 1883
MQTT_TOPIC_SERVO = "robotics/team355/servo"
MQTT_TOPIC_EVENTS = "robotics/team355/events"

# Eye landmarks indices (left/right)
LEFT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144]

# Mouth landmarks
MOUTH_CORNER_LEFT = 61
MOUTH_CORNER_RIGHT = 291
MOUTH_TOP = 0
MOUTH_BOTTOM = 17
MOUTH_IDXS = [MOUTH_CORNER_LEFT, MOUTH_CORNER_RIGHT, 13, 14]

def compute_ear(eye_pts):
    vert1 = np.linalg.norm(eye_pts[1] - eye_pts[5])
    vert2 = np.linalg.norm(eye_pts[2] - eye_pts[4])
    horiz = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return (vert1 + vert2) / (2 * horiz + 1e-6)

def compute_mar(mouth_pts):
    vert = np.linalg.norm(mouth_pts[2] - mouth_pts[3])
    horiz = np.linalg.norm(mouth_pts[0] - mouth_pts[1])
    return vert / (horiz + 1e-6)

def main():
    db_path = Path("data/db/face_db.npz")
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    det = HaarFaceMesh5pt(min_size=(70, 70), debug=False)
    embedder = ArcFaceEmbedderONNX("models/embedder_arcface.onnx", (112, 112), debug=False)
    db = load_db_npz(db_path)
    matcher = FaceDBMatcher(db, dist_thresh=0.61)

    full_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # MQTT Client Setup
    client = mqtt.Client()
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        print(f"Connected to MQTT Broker at {MQTT_BROKER}")
    except Exception as e:
        print(f"Failed to connect to MQTT: {e}")
        client = None

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")

    print(f"Face Locking for '{TARGET_IDENTITY}'. q=quit, r=reload DB")

    locked_face: Optional[FaceDet] = None
    lock_timeout = 0
    prev_nose_x: Optional[float] = None
    history_file: Optional[Path] = None
    
    # UI feedback state
    active_actions = {}  # type: Dict[str, float] (action -> expiry_time)

    t0 = time.time()
    frames = 0
    fps: Optional[float] = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        faces = det.detect(frame, max_faces=5)
        vis = frame.copy()
        now = time.time()

        # FPS calc
        frames += 1
        dt = now - t0
        if dt >= 1.0:
            fps = frames / dt
            frames = 0
            t0 = now

        if locked_face:
            # Stable tracking
            if faces:
                prev_center = ((locked_face.x1 + locked_face.x2) / 2, (locked_face.y1 + locked_face.y2) / 2)
                dists = [np.linalg.norm(((f.x1 + f.x2) / 2 - prev_center[0], (f.y1 + f.y2) / 2 - prev_center[1])) for f in faces]
                closest_face = faces[np.argmin(dists)]

                aligned, _ = align_face_5pt(frame, closest_face.kps, (112, 112))
                emb = embedder.embed(aligned)
                mr = matcher.match(emb)

                if mr.name == TARGET_IDENTITY or lock_timeout < LOCK_TIMEOUT_FRAMES:
                    locked_face = closest_face
                    lock_timeout = 0 if mr.name == TARGET_IDENTITY else lock_timeout + 1
                else:
                    locked_face = None
            else:
                lock_timeout += 1
                if lock_timeout >= LOCK_TIMEOUT_FRAMES:
                    locked_face = None
                    if history_file:
                        print(f"Lock released. History saved to {history_file}")
                        history_file = None

            if locked_face:
                # Draw lock
                cv2.rectangle(vis, (locked_face.x1, locked_face.y1), (locked_face.x2, locked_face.y2), (0, 0, 255), 3)
                cv2.putText(vis, f"LOCKED: {TARGET_IDENTITY}", (locked_face.x1, locked_face.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # FaceMesh for actions
                rx1, ry1, rx2, ry2 = locked_face.x1, locked_face.y1, locked_face.x2, locked_face.y2
                roi = frame[ry1:ry2, rx1:rx2]
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                res = full_mesh.process(rgb_roi)

                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0].landmark
                    H, W = roi.shape[:2]
                    pts = np.array([[lm[i].x * W + rx1, lm[i].y * H + ry1] for i in range(len(lm))])

                    # 1. Nose Movement
                    nose_x = pts[1][0]
                    if prev_nose_x is not None:
                        delta_x = nose_x - prev_nose_x
                        if abs(delta_x) > MOVE_THRESHOLD_PIX:
                            action = "MOVE RIGHT" if delta_x > 0 else "MOVE LEFT"
                            log_action(history_file, action.lower(), f"Delta x: {delta_x:.1f}")
                            active_actions[action] = now + 1.0
                            
                            # Publish Servo Angle
                            if client:
                                # Map face position to servo angle (0-180)
                                # W is ROI width. pts[1][0] is nose x in Screen space.
                                # Let's use relative position in ROI.
                                rel_x = (nose_x - rx1) / W # 0.0 (left) to 1.0 (right)
                                angle = int((1.0 - rel_x) * 180) # Invert if needed for servo direction
                                angle = max(0, min(180, angle))
                                payload = json.dumps({"angle": angle, "nose_x": float(nose_x)})
                                client.publish(MQTT_TOPIC_SERVO, payload)
                    prev_nose_x = nose_x

                    # 2. Blink Detection
                    ear = (compute_ear(pts[LEFT_EYE_IDXS]) + compute_ear(pts[RIGHT_EYE_IDXS])) / 2.0
                    if ear < EAR_BLINK_THRESHOLD:
                        log_action(history_file, "eye blink", f"EAR: {ear:.2f}")
                        active_actions["BLINK"] = now + 0.5

                    # 3. Improved Smile Detection
                    # Metric A: MAR (Open mouth)
                    mar = compute_mar(pts[MOUTH_IDXS])
                    
                    # Metric B: Mouth Width ratio (relative to eyes)
                    eye_dist = np.linalg.norm(pts[33] - pts[263])
                    mouth_width = np.linalg.norm(pts[61] - pts[291])
                    smile_width_ratio = mouth_width / (eye_dist + 1e-6)
                    
                    # Metric C: Corner elevation (curvature)
                    mouth_center_y = (pts[0][1] + pts[17][1]) / 2.0
                    corner_y = (pts[61][1] + pts[291][1]) / 2.0
                    smile_curve = mouth_center_y - corner_y # Positive if corners are above mouth center

                    is_smiling = False
                    reason = ""
                    if mar > MAR_SMILE_THRESHOLD:
                        is_smiling, reason = True, f"MAR:{mar:.2f}"
                    elif smile_width_ratio > SMILE_WIDTH_THRESHOLD:
                        is_smiling, reason = True, f"Width:{smile_width_ratio:.2f}"
                    elif smile_curve > 5.0: # Corners are significantly higher than mouth center
                        is_smiling, reason = True, f"Curve:{smile_curve:.1f}"

                    if is_smiling:
                        log_action(history_file, "smile", reason)
                        active_actions["SMILE"] = now + 1.0
                        if client:
                            client.publish(MQTT_TOPIC_EVENTS, json.dumps({"event": "smile", "reason": reason}))
                        # Visual: highlight mouth
                        for idx in [61, 291, 0, 17]:
                            cv2.circle(vis, (int(pts[idx][0]), int(pts[idx][1])), 3, (0, 255, 255), -1)

                if history_file is None:
                    ts = time.strftime("%Y%m%d%H%M%S")
                    history_file = HISTORY_DIR / f"{TARGET_IDENTITY.lower()}_history_{ts}.txt"
                    log_action(history_file, "lock started", "Face locked")

        else:
            # Normal recognition
            for f in faces:
                aligned, _ = align_face_5pt(frame, f.kps, (112, 112))
                emb = embedder.embed(aligned)
                mr = matcher.match(emb)
                if mr.name == TARGET_IDENTITY and mr.accepted:
                    locked_face, lock_timeout = f, 0
                    print(f"Locked onto {TARGET_IDENTITY}")
                    break
                
                color = (0, 255, 0) if mr.accepted else (0, 0, 255)
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
                cv2.putText(vis, mr.name if mr.accepted else "Unknown", (f.x1, f.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Draw active actions UI
        y_offset = 70
        for action, expiry in list(active_actions.items()):
            if now < expiry:
                cv2.putText(vis, action, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                y_offset += 45
            else:
                del active_actions[action]

        # Header info
        header = f"IDs={len(matcher._names)}  thr={matcher.dist_thresh:.2f}"
        if fps: header += f" fps={fps:.1f}"
        cv2.putText(vis, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.imshow("face_locking", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break
        elif key == ord("r"): matcher.reload_from(db_path)

    cap.release()
    cv2.destroyAllWindows()

# Logging helper
def log_action(file_path: Path, action_type: str, desc: str):
    if file_path:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with file_path.open("a") as f:
            f.write(f"{ts} - {action_type} - {desc}\n")

# Load_db_npz from recognize.py (add if not in file)
def load_db_npz(db_path: Path) -> Dict[str, np.ndarray]:
    if not db_path.exists():
        return {}
    data = np.load(str(db_path), allow_pickle=True)
    out = {k: np.asarray(data[k], dtype=np.float32).reshape(-1) for k in data.files}
    return out

if __name__ == "__main__":
    main()