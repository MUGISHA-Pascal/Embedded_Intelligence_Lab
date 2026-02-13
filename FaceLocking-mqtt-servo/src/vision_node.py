"""
vision_node.py
Simulated Vision Node for Distributed Vision-Control System.
Tracks face and publishes movement commands via MQTT.
Topic: vision/team313/movement
"""

import time
import argparse
import cv2
import json
import numpy as np
import paho.mqtt.client as mqtt
from pathlib import Path
import sys
import base64

# Add src to path if needed
sys.path.append(str(Path(__file__).parent.parent))

# Import Face Locking modules
from src.haar_5pt import Haar5ptDetector
from src.recognize import ArcFaceEmbedderONNX, FaceDBMatcher, load_db_npz
from src.face_locking import FaceLockSystem

# Configuration
DEFAULT_BROKER = "localhost" 
PORT = 1883
TEAM_ID = "team313"
TOPIC_MOVEMENT = f"vision/{TEAM_ID}/movement"
TOPIC_SNAPSHOT = f"vision/{TEAM_ID}/snapshot"
TOPIC_HEARTBEAT = f"vision/{TEAM_ID}/heartbeat"

class VisionNode:
    def __init__(self, broker, port, target_name):
        # MQTT Setup
        self.client = mqtt.Client(client_id=f"{TEAM_ID}_vision_node")
        self.client.on_connect = self.on_connect
        self.client.connect(broker, port, 60)
        self.client.loop_start()
        
        # Face Recognition & Locking Setup
        print("Initializing Face Recognition...")
        self.det = Haar5ptDetector(min_size=(70, 70))
        self.embedder = ArcFaceEmbedderONNX(input_size=(112, 112))
        
        # Load Database
        db_path = Path(__file__).parent.parent / "data/db/face_db.npz"
        if not db_path.exists():
            print(f"ERROR: Face DB not found at {db_path}. Run enroll.py first!")
            sys.exit(1)
            
        db = load_db_npz(db_path)
        if target_name not in db:
            print(f"WARNING: Target '{target_name}' not in database. Available: {list(db.keys())}")
        
        self.matcher = FaceDBMatcher(db, dist_thresh=0.60)
        self.system = FaceLockSystem(target_name, self.matcher, self.det)
        
        self.running = True
        self.last_heartbeat = 0
        self.last_publish_time = 0
        self.mqtt_topic = TOPIC_MOVEMENT
        self.snapshot_sent = False  # Track if we've sent the face snapshot

    def on_connect(self, client, userdata, flags, rc):
        print(f"Connected to MQTT Broker with result code {rc}")
        self.publish_heartbeat()

    def publish_movement(self, status, confidence=1.0, target=None, locked=False, face_image=None):
        payload = {
            "status": status,
            "confidence": confidence,
            "target": target,
            "locked": locked,
            "timestamp": time.time()
        }
        
        # Always publish movement command
        self.client.publish(TOPIC_MOVEMENT, json.dumps(payload))
        
        # Publish face image on a SEPARATE topic if available
        if face_image is not None:
            _, buffer = cv2.imencode('.jpg', face_image, [cv2.IMWRITE_JPEG_QUALITY, 70])
            img_payload = {
                "face_image": base64.b64encode(buffer).decode('utf-8'),
                "target": target,
                "timestamp": payload["timestamp"]
            }
            self.client.publish(TOPIC_SNAPSHOT, json.dumps(img_payload))
            print(f"Published: {status} + 📸 Snapshot")
        else:
            print(f"Published: {status}")

    def publish_heartbeat(self):
        payload = {
            "node": "pc_vision",
            "status": "ONLINE",
            "timestamp": time.time()
        }
        self.client.publish(TOPIC_HEARTBEAT, json.dumps(payload))

    def run(self):
        cap = cv2.VideoCapture(0) # Use default camera
        if not cap.isOpened():
             cap = cv2.VideoCapture(1)
        
        print(f"Vision Node Started. Tracking target: {self.system.target_name}")
        print(f"Publishing to {TOPIC_MOVEMENT}")
        
        while self.running:
            ret, frame = cap.read()
            if not ret: break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            H, W = frame.shape[:2]
            
            # Process Frame using FaceLockSystem
            # Note: process_frame now returns (vis_frame, target_face_obj)
            vis, target_face = self.system.process_frame(frame, self.embedder)
            
            status = "NO_FACE"
            face_crop = None
            
            if target_face:
                # Target is found and locked
                f = target_face
                
                # Extract face crop for dashboard (only if not sent yet)
                if not self.snapshot_sent:
                    x1, y1, x2, y2 = int(f.x1), int(f.y1), int(f.x2), int(f.y2)
                    # Add padding
                    pad = 20
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(W, x2 + pad)
                    y2 = min(H, y2 + pad)
                    face_crop = frame[y1:y2, x1:x2]
                    self.snapshot_sent = True  # Mark as sent
                    print("📸 Face snapshot captured and will be sent")
                
                # Calculate Center
                cx = (f.x1 + f.x2) / 2.0
                cx_norm = cx / W
                
                # Movement Logic
                # Deadband: 0.4 to 0.6 is CENTERED
                if cx_norm < 0.4:
                    status = "MOVE_LEFT"
                elif cx_norm > 0.6:
                    status = "MOVE_RIGHT"
                else:
                    status = "CENTERED"
            else:
                # No face detected - reset snapshot flag
                if self.snapshot_sent:
                    self.snapshot_sent = False
                    print("🔓 Target lost - snapshot flag reset")
            
            # --- RATE LIMITING (10Hz) ---
            current_time = time.time()
            if current_time - self.last_publish_time >= 0.1:
                is_locked = (status != "NO_FACE")
                self.publish_movement(status, target=self.system.target_name, locked=is_locked, face_image=face_crop)
                self.last_publish_time = current_time
            
            # Heartbeat every 5s
            if time.time() - self.last_heartbeat > 5:
                self.publish_heartbeat()
                self.last_heartbeat = time.time()
            
            cv2.imshow("Vision Node (Locked)", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.client.loop_stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", type=str, default=DEFAULT_BROKER, help="MQTT Broker Address")
    parser.add_argument("--name", type=str, default="andrew", help="Target name to lock onto")
    args = parser.parse_args()

    node = VisionNode(args.broker, PORT, args.name)
    node.run()
