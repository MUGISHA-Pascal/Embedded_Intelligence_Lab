import cv2
import threading
import time
import json
import argparse
from pathlib import Path
from flask import Flask, render_template, Response
import paho.mqtt.client as mqtt

# === CONFIGURATION ===
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
TOPIC_COMMAND = "embedded_sys/face/command"
TOPIC_STATUS = "embedded_sys/face/status"

FLASK_PORT = 5000

# Paths
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "trained_lbph_face_recognizer_model.yml"
CASCADE_PATH = MODELS_DIR / "haarcascade_frontalface_default.xml"
LABELMAP_PATH = MODELS_DIR / "label_map.json"

# Hardware / Image Params
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CENTER_TOLERANCE = 50 # How close to center before stopping movement

# Global Variables
video_frame = None
lock = threading.Lock()
client = None

# === FLASK APP ===
app = Flask(__name__, template_folder="../web")

@app.route('/')
def index():
    return render_template('dashboard.html')

def generate_frames():
    global video_frame
    while True:
        with lock:
            if video_frame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", video_frame)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# === PROCESSSING LOOP ===
def process_video_loop():
    global video_frame, client
    
    # Load Models
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    recognizer.read(str(MODEL_PATH))
    
    face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
    if face_cascade.empty():
        print("Error: Cascade classifier not found")
        return

    # Load Label Map
    id_to_name = {}
    if LABELMAP_PATH.exists():
        with open(LABELMAP_PATH, "r") as f:
            name_to_id = json.load(f)
        id_to_name = {int(v): k for k, v in name_to_id.items()}
    
    cap = cv2.VideoCapture(0)
    cap.set(3, FRAME_WIDTH)
    cap.set(4, FRAME_HEIGHT)
    
    print("Video Processing Started...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        
        frame_center_x = FRAME_WIDTH // 2
        command = "S"
        person_name = "Unknown"
        confidence = 0
        
        if len(faces) > 0:
            # Find largest face (closest)
            (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
            
            # Predict
            roi_gray = gray[y:y+h, x:x+w]
            try:
                id_, dist = recognizer.predict(roi_gray)
                confidence = max(0, 100 - dist) # Rough confidence estimate
                if confidence > 40: # Threshold
                    person_name = id_to_name.get(id_, f"ID:{id_}")
                else:
                    person_name = "Unknown"
            except Exception as e:
                print(f"Prediction error: {e}")

            # Draw
            color = (0, 255, 0) if person_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{person_name} ({int(confidence)}%)", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Tracking Logic (Center the face)
            cx = x + w // 2
            if cx < frame_center_x - CENTER_TOLERANCE:
                command = "L" # Move Left
            elif cx > frame_center_x + CENTER_TOLERANCE:
                command = "R" # Move Right
            else:
                command = "S" # Stop
                
            # Publish Status
            if client:
                status_payload = json.dumps({
                    "detected": True,
                    "name": person_name,
                    "confidence": int(confidence),
                    "command": command
                })
                client.publish(TOPIC_STATUS, status_payload)
                client.publish(TOPIC_COMMAND, command)
        else:
             # No face
            if client:
                 client.publish(TOPIC_STATUS, json.dumps({"detected": False}))
                 client.publish(TOPIC_COMMAND, "S")

        with lock:
            video_frame = frame.copy()
            
        time.sleep(0.03) # ~30 FPS
        
    cap.release()

# === MQTT SETUP ===
def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT Broker with result code {rc}")

def start_mqtt():
    global client
    client = mqtt.Client()
    client.on_connect = on_connect
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
    except Exception as e:
        print(f"MQTT Connection Failed: {e}")

if __name__ == '__main__':
    # Start MQTT
    start_mqtt()
    
    # Start Video Thread
    t = threading.Thread(target=process_video_loop)
    t.daemon = True
    t.start()
    
    # Start Flask
    print(f"Starting Web Dashboard on http://0.0.0.0:{FLASK_PORT}")
    app.run(host='0.0.0.0', port=FLASK_PORT, debug=False, use_reloader=False)
