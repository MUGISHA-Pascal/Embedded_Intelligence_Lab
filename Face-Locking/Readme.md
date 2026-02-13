# Edge ArcFace Face Recognition with 5-Point Alignment

A lightweight, real-time face recognition pipeline optimized for embedded and edge systems (Jetson, Raspberry Pi, edge PCs). It combines classical computer vision with deep learning: Haar Cascade for fast face detection, MediaPipe FaceMesh for 5-point landmark extraction, a 5-point affine alignment to a canonical 112×112 pose, and an ArcFace ONNX embedder to produce identity-preserving embeddings.

This repository focuses on accuracy and efficiency for constrained hardware while keeping the pipeline modular and easy to extend.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Why 5-Point Alignment?](#why-5-point-alignment)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
  - [Face Detection (Haar Cascade)](#face-detection-haar-cascade)
  - [Facial Landmark Detection (MediaPipe FaceMesh)](#facial-landmark-detection-mediapipe-facemesh)
  - [5-Point Face Alignment](#5-point-face-alignment)
  - [ArcFace Embedding (ONNX Runtime)](#arcface-embedding-onnx-runtime)
  - [L2 Normalization & Similarity](#l2-normalization--similarity)
- [Visualization & Demo Features](#visualization--demo-features)
- [Requirements](#requirements)
- [Installation & Run](#installation--run)
- [Usage / Controls](#usage--controls)
- [Target Use Cases](#target-use-cases)
- [Roadmap / Future Improvements](#roadmap--future-improvements)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact / Next Steps](#contact--next-steps)

---

## Project Overview

The pipeline (real-time) performs:

1. Capture frames from a camera
2. Detect faces with Haar Cascade
3. Extract 5 facial landmarks using MediaPipe FaceMesh
4. Align face to canonical pose (112×112)
5. Generate ArcFace embeddings via an ONNX model
6. L2-normalize embeddings
7. Compute cosine similarity between embeddings and visualize results

Designed for:
- Low compute budgets
- Modular experimentation
- Educational clarity

---

## Key Features

- Real-time face detection + embedding extraction
- 5-point landmark-based affine alignment for stable embeddings
- ArcFace embedding using ONNX Runtime for portability
- Lightweight detection (Haar) suitable for edge devices
- Embedding heatmap and similarity visualization for debugging

---

## Why 5-Point Alignment?

ArcFace and similar models expect consistently aligned, frontal faces. Using 5 landmarks (left eye, right eye, nose, left mouth corner, right mouth corner) allows us to:

- Reduce pose variation
- Normalize scale and rotation
- Improve embedding stability and recognition accuracy
- Keep computation minimal (important for embedded devices)

---

## System Architecture

Camera Frame  
↓  
Haar Face Detection  
↓  
MediaPipe FaceMesh (5-point extraction)  
↓  
5-Point Face Alignment (Affine Transform → 112×112)  
↓  
ArcFace ONNX Embedding (embedder_arcface.onnx)  
↓  
L2 Normalization  
↓  
Cosine Similarity + Visualization

---

## Project Structure

ArcFace-based Face Recognition with 5-Point Facial Alignment/
│
├── src/
│   ├── __init__.py
│   ├── embed.py          # Main pipeline (camera → embedding)
│   └── haar_5pt.py       # Haar detection + 5-point extraction & alignment
│
├── models/
│   └── embedder_arcface.onnx
│
├── requirements.txt
└── README.md

---

## Core Components

### Face Detection (Haar Cascade)
- Fast, classical detector (OpenCV)
- Provides a rough bounding box for where to run the landmark detector
- Low computational cost — ideal for real-time performance on edge devices

### Facial Landmark Detection (MediaPipe FaceMesh)
- High-precision landmark detector
- From the full mesh only 5 key points are extracted
- Provides stable landmark positions even with small head movements

### 5-Point Face Alignment
- Affine transformation that maps detected landmarks to a fixed template
- Produces 112×112 aligned RGB face images required by ArcFace
- Normalizes pose, scale, and rotation

### ArcFace Embedding (ONNX Runtime)
- Pretrained ArcFace model loaded via ONNX Runtime
- Input: aligned 112×112 RGB face
- Output: embedding vector (identity-preserving, discriminative)

### L2 Normalization & Similarity
- Embeddings are L2-normalized to unit length
- Cosine similarity computed as dot(embedding_1, embedding_2)
  - Values near 1.0 suggest the same identity
  - Lower values suggest different identities

---

## Visualization & Demo Features

The demo includes:
- Live face bounding box
- 5-point landmark overlay
- Real-time aligned face preview (112×112)
- Embedding heatmap visualization
- FPS counter
- Real-time similarity display between consecutive frames

These make the system both educational and debuggable.

---

## Requirements

- Python 3.10 or 3.11 (MediaPipe may be unstable on 3.12+)
- See `requirements.txt` for exact packages. Typical dependencies include:
  - opencv-python
  - mediapipe
  - onnxruntime
  - numpy
  - matplotlib (optional for heatmap visualization)

---

## Installation & Run

1. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the demo from the project root:
```bash
python -m src.embed
```

Notes:
- Ensure `models/embedder_arcface.onnx` exists (place your ONNX model there).
- Test camera index or video file path if you have multiple cameras.

---

## Usage / Controls

- q → Quit application
- p → Print embedding statistics to terminal

---

## Target Use Cases

- Embedded face recognition systems
- Edge AI identity verification
- Attendance and access control systems
- Research and learning about face recognition pipelines

---

## Roadmap / Future Improvements

- Replace Haar with a lightweight CNN detector (for better multi-face robustness)
- Add an embedding database + matching pipeline
- Quantize ArcFace model for reduced power & faster inference
- Support multi-face tracking
- Hardware acceleration backends (TensorRT, NNAPI, etc.)
- Packaging for specific platforms (Jetson, Raspberry Pi)

---

## License

This project is intended for educational and research purposes. (Add a specific license file like `LICENSE` if you want to choose an OSI-approved license.)

---

## Acknowledgments

- ArcFace / InsightFace
- MediaPipe
- OpenCV
- ONNX Runtime

---