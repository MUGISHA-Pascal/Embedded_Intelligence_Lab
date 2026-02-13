# Face Recognition & Tracking System with MQTT

This project implements a real-time face recognition and tracking system. It consists of three main components:

1.  **Computer Vision Module (PC)**: Detects and recognizes faces, calculates tracking movements.
2.  **Embedded Controller (ESP8266)**: Receives movement commands to control a servo motor for camera tracking.
3.  **Real-time Dashboard**: A web interface to view the live camera feed and system status.

## System Architecture

- **Input**: Webcam video feed processed by OpenCV.
- **Processing**:
  - Face Detection (Haar Cascade).
  - Face Recognition (LBPH Algorithm).
  - Tracking Logic (Calculates offset from center).
- **Communication**: MQTT Protocol (HiveMQ Broker).
- **Output**:
  - Servo Motor Control (via ESP8266).
  - Web Dashboard (Flask + MQTT WebSockets).

## MQTT Topics

The system uses the following topics on `broker.hivemq.com`:

| Topic                       | Direction       | Description            | Payload Format                                                 |
| :-------------------------- | :-------------- | :--------------------- | :------------------------------------------------------------- |
| `embedded_sys/face/command` | PC -> ESP8266   | Motor control commands | `L` (Left), `R` (Right), `S` (Stop)                            |
| `embedded_sys/face/status`  | PC -> Dashboard | Recognition status     | JSON: `{"detected": true, "name": "Pascal", "confidence": 85}` |

## Live Dashboard

The live dashboard is hosted by the Python application.

**URL**: [http://localhost:5000](http://localhost:5000)

(If accessing from another device, replace `localhost` with the PC's IP address).

## Prerequisites

### Hardware

- PC with Webcam.
- ESP8266 (NodeMCU/Wemos).
- Servo Motor (e.g., SG90) connected to Pin D4 (GPIO2).

### Software

- Python 3.x
- Arduino IDE (for flashing ESP8266).

## Installation & Setup

### 1. Python Environment (PC)

Install the required dependencies:

```bash
pip install opencv-contrib-python flask paho-mqtt
```

### 2. ESP8266 Firmware

1.  Open `esp8266_firmware/main_esp8266.ino` in Arduino IDE.
2.  Install `PubSubClient` library by Nick O'Leary.
3.  Update the `ssid` and `password` variables with your WiFi credentials.
4.  Upload the code to your ESP8266 board.

### 3. Run the System

Navigate to the `src` directory and run the main script:

```bash
python main.py
```

The system will:

1.  Connect to the MQTT broker.
2.  Start the video stream server.
3.  Open a window showing the camera feed with recognition overlays.
4.  Start publishing commands to the ESP8266.

Open your browser to `http://localhost:5000` to view the dashboard.
