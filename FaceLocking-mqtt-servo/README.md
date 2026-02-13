# Face Recognition with ArcFace ONNX and 5-Point Alignment

<img src="https://via.placeholder.com/800x200/007bff/ffffff?text=ArcFace+ONNX+%2B+5-Point+Alignment" alt="Project Banner" width="800"/>

**Author:** Andrew Byukusenge  
**Instructor:** Gabriel Baziramwabo  
**Organization:** Rwanda Coding Academy  

This project implements a **Distributed Face Recognition and Tracking System** for IoT-based servo control using:

- **ArcFace** model (ONNX) for face recognition
- **5-point facial landmark alignment** for precise face detection
- **MQTT** for distributed communication between components
- **ESP8266** microcontroller for edge-based servo control
- **Real-time Web Dashboard** for system monitoring

The system is designed for **embedded systems applications**, demonstrating how computer vision, IoT communication, and edge computing work together in a practical face-tracking servo control system.

## Table of Contents

- [Assessment Details (Week 06)](#assessment-details-week-06)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage](#usage)

## System Architecture

This distributed system consists of four main components:

1. **Vision Node (PC)**: Detects, recognizes, and tracks faces using ArcFace and MediaPipe. Publishes movement commands via MQTT.
2. **MQTT Broker (VPS)**: Central message broker facilitating communication between all components.
3. **ESP8266 (Edge Controller)**: Subscribes to movement commands and controls a servo motor to physically track the detected face.
4. **Web Dashboard**: Real-time visualization of system status, tracking data, and lock status.

## Features

- **Face Recognition & Locking**: Lock onto a specific enrolled identity and track their movements
- **Distributed Architecture**: Components communicate via MQTT, allowing flexible deployment
- **Real-time Servo Control**: ESP8266 controls servo motor based on face position
- **Live Dashboard**: Web-based monitoring with WebSocket updates
- **Action Detection**: Detects blinks, smiles, and head movements
- **CPU-friendly**: Runs on standard laptops without GPU requirements

## Project Structure

```
Face_recognition_with_Arcface/
├── src/
│   ├── vision_node.py       # Main vision processing + MQTT publisher
│   ├── face_locking.py      # Face locking & action detection
│   ├── haar_5pt.py          # Face detection core
│   └── recognize.py         # ArcFace recognition
├── backend/
│   ├── server.js            # MQTT-to-WebSocket relay
│   └── package.json
├── dashboard/
│   └── index.html           # Real-time web dashboard
├── esp8266/
│   └── vision_servo/
│       └── vision_servo.ino # Arduino firmware for ESP8266
├── data/
│   └── db/                  # Face database (face_db.npz)
└── models/
    └── embedder_arcface.onnx
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
cd backend && npm install
```

### 2. Enroll Your Face
```bash
python -m src.enroll --name andrew
```

### 3. Start the System

**On VPS (or local MQTT broker):**
```bash
mosquitto -c mosquitto.conf
```

**On PC - Terminal 1 (Backend):**
```bash
cd backend
npm start
```

**On PC - Terminal 2 (Vision Node):**
```bash
python src/vision_node.py --broker 157.173.101.159 --name andrew
```

### 4. Flash ESP8266
Upload `esp8266/vision_servo/vision_servo.ino` using Arduino IDE.

### 5. Access Dashboard
Open: [http://157.173.101.159:9313]([http://157.173.101.159:9313/])

## Assessment Details (Week 06)

### System Description
This project implements a **Distributed Face Recognition and Locking System** using:
1.  **Vision Node (PC)**: Detects, recognizes, and tracks faces using ArcFace and MediaPipe. Publishes movement commands.
2.  **MQTT Broker (VPS)**: Facilitates communication between the PC, ESP8266, and Dashboard.
3.  **ESP8266 (Edge)**: Subscribes to movement commands and controls a Servo motor to track the face.
4.  **Web Dashboard**: Visualizes the real-time blocking status and tracking info.

### MQTT Topics
-   `vision/team313/movement`: JSON payload with `status` (MOVE_LEFT, MOVE_RIGHT, CENTERED), `target`, and `locked` state.
-   `vision/team313/heartbeat`: System health status.

### Live Dashboard
**URL**: [http://157.173.101.159:8082]([http://157.173.101.159:9313/])

## Face Locking
The new Face Locking feature (`src/face_locking.py` and `vision_node.py`) allows you to track a single enrolled identity continuously.

**How it works:**
1.  **Search**: The system looks for the user using ArcFace recognition.
2.  **Lock**: Once found, it tracks the user's face position.
3.  **Action Detection**: It measures facial landmarks to detect:
    - **Blinks**: Using Eye Aspect Ratio (EAR).
    - **Smiles**: Using mouth width ratios.
    - **Movement**: Using nose position (Left/Right).

**History**:
A file named `<name>_history_<timestamp>.txt` is created to record all detected actions.
