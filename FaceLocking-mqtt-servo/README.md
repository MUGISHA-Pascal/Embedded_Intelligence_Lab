# Distributed Vision-Control System (Phase 1)
**Course:** Intelligent Robotics  
**Phase:** Phase 1 (Open-Loop Actuation)  
**Team ID:** team355 

This project implements a distributed system where a PC tracks a face and sends movement commands to an ESP8266-controlled servo motor via an MQTT broker.

---

## 1. Prerequisites

### Hardware
*   **ESP8266 Board** (NodeMCU or similar)
*   **Micro-USB Cable**
*   **Servo Motor** (SG90 or similar)
*   **Jumper Wires**
*   **PC with Webcam**

### Software
*   **Python 3.11+**
*   **Arduino IDE**
*   **MQTT Broker Access** (IP: `157.173.101.159`)

---

## 2. PC Setup (Vision Engine)

### Step 1: Clone and Create Virtual Environment
```bash
# Open PowerShell or Terminal in the project folder
python -m venv .venv
.venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install opencv-python numpy onnxruntime mediapipe==0.10.21 paho-mqtt
```

### Step 3: Download ArcFace Model
The system requires the ArcFace ONNX model for face recognition.
```bash
mkdir models
# Manually download or use curl if available:
curl -L -o models/embedder_arcface.onnx "https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/w600k_r50.onnx"
```

### Step 4: Personnel Enrollment
Before tracking, you must enroll the target face:
1. Run the enrollment script:
   ```bash
   python src/enroll.py
   ```
2. Enter the name (e.g., `Babou`).
3. Press `s` to capture samples and `q` to finish.
4. Verify that `data/db/face_db.npz` is created.

---

## 3. ESP8266 Setup (Edge Controller)

### Step 1: Hardware Wiring
Connect the servo to the ESP8266:
*   **Brown/Black Wire**: `GND`
*   **Red Wire**: `VIN` (5V power from USB)
*   **Orange/Yellow Wire**: `D4` (GPIO2)

### Step 2: Arduino IDE Configuration
1. Install **ESP8266 Board Support**:
   *   Settings -> Additional Boards Manager URLs -> `http://arduino.esp8266.com/stable/package_esp8266com_index.json`
   *   Tools -> Board -> Boards Manager -> Search `esp8266` and install.
2. Install **Required Libraries**:
   *   Tools -> Manage Libraries -> Search and install:
       *   `PubSubClient` (by Nick O'Leary)
       *   `ArduinoJson` (by Benoit Blanchon)

### Step 3: Flash Firmware
1. Open `src/esp8266_servo.ino`.
2. Update the WiFi credentials:
   ```cpp
   const char* ssid = "YOUR_WIFI_NAME";
   const char* password = "YOUR_WIFI_PASSWORD";
   ```
3. Select your board (e.g., `NodeMCU 1.0`) and Port.
4. Click **Upload**.

---

## 4. Running the System

### Step 1: Start the Vision Engine
Ensure your webcam is connected and the ESP8266 is powered on.
```bash
# In your terminal (with .venv active)
python src/face_locking.py
```
*   **Controls**: `q` to quit, `r` to reload the face database.
*   **Logic**: Once the target face is recognized and "LOCKED", the PC will begin publishing angles to `robotics/team355/servo`.

### Step 2: Monitor Output
*   **PC Console**: Should show "Connected to MQTT Broker".
*   **Arduino Serial Monitor**: Should show "WiFi connected" and "Moving servo to: [Angle]".

---

## 5. Technical Details

### Topic Isolation
To avoid interference with other teams, this project uses a dedicated prefix:
*   **Command Topic**: `robotics/team355/servo`
*   **Event Topic**: `robotics/team355/events`

### Data Format (JSON)
The PC sends JSON payloads over MQTT:
```json
{
  "angle": 90,
  "nose_x": 320.5
}
```
## 6. Supplementary (Optional Dashboard)
A real-time web dashboard is available in `dashboard/index.html`. 
To host it persistently on your VPS:
```bash
# Connect to VPS
ssh user355@157.173.101.159
# Run the server in a persistent background process
nohup python3 -m http.server 9335 --directory ~/site >> ~/site/http.log 2>&1 &
```
Access at `http://157.173.101.159:9335`.

---
*Assignment Phase 1 - Distributed Vision-Control*
