# Face Recognition System - MediaPipe + LBPH

**Week 13 Assignment: AI Without ML - Face Recognition**  
**Deadline: Sunday, 7 December 2025**

A complete face-recognition pipeline using **MediaPipe** for face detection and **LBPH** (Local Binary Patterns Histograms) for classical feature-based recognition.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [Assignment Checklist](#assignment-checklist)

---

## Overview

This project demonstrates a complete face recognition system that combines:

1. **MediaPipe Face Detection** - AI without ML approach for robust face detection
2. **LBPH Face Recognizer** - Classical computer vision algorithm for face recognition

The system consists of three main scripts:

- **1_capture_faces.py** - Capture training images
- **2_train_model.py** - Train the LBPH recognizer
- **3_predict_faces.py** - Real-time face recognition

---

## How It Works

### Detection Stage (MediaPipe)

MediaPipe uses pre-trained models to detect faces in images. It's called "AI without ML" because you don't need to train the detection model - it works out of the box.

**Key Features:**

- Fast and accurate face detection
- Works in various lighting conditions
- Provides bounding box coordinates
- No training required

### Recognition Stage (LBPH)

LBPH is a classical computer vision algorithm that:

1. Divides face images into small regions
2. Extracts local binary patterns from each region
3. Creates histograms of these patterns
4. Compares histograms to recognize faces

**Why LBPH?**

- Classical feature-based approach (not deep learning)
- Works well with limited training data
- Fast inference
- Interpretable results

---

## Project Structure

```
face_recognition_with_mediapipe/
├── 1_capture_faces.py          # Script to capture training images
├── 2_train_model.py             # Script to train LBPH model
├── 3_predict_faces.py           # Script for real-time recognition
├── face_mesh.py                 # Face mesh visualization demo
├── face_recognition.py          # Face mesh detection demo
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore file
├── README.md                    # This file
├── datasets/                    # Training images (created by script 1)
│   ├── Person1/
│   │   ├── Person1_000_*.jpg
│   │   ├── Person1_001_*.jpg
│   │   └── ...
│   ├── Person2/
│   └── ...
└── models/                      # Trained models (created by script 2)
    ├── lbph_face_recognizer.yml # Trained LBPH model
    └── label_map.json           # Label to name mapping
```

---

## Installation

### Prerequisites

- Python 3.8 - 3.11 (MediaPipe doesn't support Python 3.12+)
- Webcam
- Linux, macOS, or Windows

### Step 1: Create Virtual Environment

```bash
# Navigate to project directory
cd face_recognition_with_mediapipe

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**

- `opencv-python` - Computer vision library
- `opencv-contrib-python` - OpenCV with extra modules (includes LBPH)
- `mediapipe` - Google's MediaPipe framework
- `numpy` - Numerical computing

---

## Usage Guide

### Step 1: Capture Face Images

Run the capture script to collect training data:

```bash
python 1_capture_faces.py
```

**Instructions:**

1. Enter the person's name when prompted
2. Choose number of images to capture (default: 50)
3. Position your face in the frame
4. Press **SPACE** to capture each image
5. Move slightly between captures for variety
6. Press **Q** to quit early
7. Repeat for multiple people

**Tips:**

- Capture at least 30-50 images per person
- Vary facial expressions and angles
- Ensure good lighting
- Keep face centered in frame

**Output:**

- Images saved to `datasets/{person_name}/`
- Each image is cropped to face region
- Organized by person name

### Step 2: Train the Model

After capturing images, train the LBPH recognizer:

```bash
python 2_train_model.py
```

**What happens:**

1. Loads all images from `datasets/` folder
2. Converts images to grayscale
3. Trains LBPH Face Recognizer
4. Saves model to `models/lbph_face_recognizer.yml`
5. Saves label mapping to `models/label_map.json`
6. Displays training summary

**Output:**

```
Training Summary
================================================================

Total training images: 150
Number of people: 3
Training time: 0.45 seconds

People in model:
  [0] Alice: 50 images
  [1] Bob: 50 images
  [2] Charlie: 50 images
```

### Step 3: Run Face Recognition

Start real-time face recognition:

```bash
python 3_predict_faces.py
```

**Features:**

- Real-time face detection using MediaPipe
- Face recognition using trained LBPH model
- Color-coded bounding boxes:
  - **Green** = Recognized person
  - **Red** = Unknown person
- Confidence score display (lower = better match)
- FPS counter
- Screenshot capability (press **S**)

**Controls:**

- **Q** or **ESC** - Quit
- **S** - Save screenshot

**Understanding Confidence Scores:**

- **0-50**: Excellent match
- **50-70**: Good match (default threshold)
- **70-100**: Poor match (marked as unknown)
- **100+**: Very poor match

---

## Technical Details

### MediaPipe Face Detection

**Algorithm:** BlazeFace  
**Type:** Pre-trained neural network (AI without ML from user perspective)  
**Speed:** ~30-60 FPS on modern hardware  
**Output:** Bounding box coordinates (x, y, width, height)

**Configuration:**

```python
face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=0,              # 0=short-range (2m), 1=full-range (5m)
    min_detection_confidence=0.5    # Confidence threshold
)
```

### LBPH Face Recognizer

**Algorithm:** Local Binary Patterns Histograms  
**Type:** Classical computer vision (feature-based)  
**Training:** Supervised learning on labeled face images  
**Output:** Label ID and confidence score

**Configuration:**

```python
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,        # Radius of LBP pattern
    neighbors=8,     # Number of sampling points
    grid_x=8,        # Horizontal grid divisions
    grid_y=8,        # Vertical grid divisions
    threshold=100.0  # Recognition threshold
)
```

**How LBPH Works:**

1. **Local Binary Patterns:**

   - Compare each pixel with its neighbors
   - Create binary pattern based on comparisons
   - Convert pattern to decimal number

2. **Histogram Creation:**

   - Divide face into grid (e.g., 8x8)
   - Calculate LBP histogram for each cell
   - Concatenate all histograms

3. **Recognition:**
   - Compare test image histogram with training histograms
   - Use chi-square distance metric
   - Return closest match and confidence

### Pipeline Architecture

```
Input Image
    ↓
[MediaPipe Detection]
    ↓
Face Bounding Box
    ↓
Extract Face Region
    ↓
Convert to Grayscale
    ↓
Resize to 200x200
    ↓
[LBPH Recognition]
    ↓
Label + Confidence
    ↓
Display Result
```

---

## Troubleshooting

### Issue: "No module named 'cv2.face'"

**Solution:** Install opencv-contrib-python

```bash
pip uninstall opencv-python
pip install opencv-contrib-python==4.10.0.84
```

### Issue: "Model not found"

**Solution:** Make sure you've run the scripts in order:

1. First: `1_capture_faces.py`
2. Second: `2_train_model.py`
3. Third: `3_predict_faces.py`

### Issue: "No matching distribution found for mediapipe"

**Solution:** MediaPipe requires Python 3.8-3.11

```bash
# Check Python version
python --version

# If using Python 3.12+, install Python 3.11
# Then create virtual environment with Python 3.11
python3.11 -m venv venv
```

### Issue: "Could not open webcam"

**Solutions:**

- Check if webcam is connected
- Close other applications using the webcam
- Try different camera index: `cv2.VideoCapture(1)` instead of `cv2.VideoCapture(0)`
- On Linux, check permissions: `sudo usermod -a -G video $USER`

### Issue: Poor recognition accuracy

**Solutions:**

- Capture more training images (50+ per person)
- Ensure consistent lighting during capture and recognition
- Capture images with varied expressions and angles
- Adjust confidence threshold in `3_predict_faces.py`:
  ```python
  confidence_threshold = 70  # Lower = stricter, Higher = more lenient
  ```

### Issue: "No face detected" during capture

**Solutions:**

- Ensure good lighting
- Move closer to camera
- Check if face is centered in frame
- Lower detection confidence: `min_detection_confidence=0.3`

---

## Assignment Checklist

### Required Components

- [x] **Code Scripts**

  - [x] Capture script (`1_capture_faces.py`)
  - [x] Training script (`2_train_model.py`)
  - [x] Prediction script (`3_predict_faces.py`)

- [x] **Models Folder**

  - [x] LBPH model (`models/lbph_face_recognizer.yml`)
  - [x] Label map (`models/label_map.json`)

- [x] **Documentation**
  - [x] Clear README explaining how to run the project
  - [x] Installation instructions
  - [x] Usage guide for each script
  - [x] Technical explanation of pipeline

### Understanding Demonstrated

- [x] **Detection Stage:** MediaPipe for face detection (AI without ML)
- [x] **Recognition Stage:** LBPH for classical feature-based recognition
- [x] **Pipeline Integration:** How detection and recognition work together
- [x] **Proper Error Handling:** User-friendly error messages
- [x] **Code Documentation:** Clear comments and docstrings

---

## Additional Resources

### MediaPipe Documentation

- [MediaPipe Face Detection](https://google.github.io/mediapipe/solutions/face_detection.html)
- [MediaPipe Python API](https://google.github.io/mediapipe/getting_started/python.html)

### LBPH Face Recognition

- [OpenCV Face Recognition](https://docs.opencv.org/4.x/dd/d65/classcv_1_1face_1_1LBPHFaceRecognizer.html)
- [Local Binary Patterns](https://en.wikipedia.org/wiki/Local_binary_patterns)

### OpenCV Documentation

- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Face Recognition Module](https://github.com/opencv/opencv_contrib/tree/master/modules/face)

---

## Author

**Assignment:** Week 13 - AI Without ML - Face Recognition  
**Course:** Embedded Systems  
**Date:** December 2025

---

## License

This project is created for educational purposes as part of the Embedded Systems course.

---

## Learning Outcomes

By completing this assignment, you have learned:

1. **Face Detection:** Using MediaPipe's pre-trained models
2. **Classical CV:** Understanding LBPH algorithm
3. **Pipeline Design:** Combining detection and recognition stages
4. **Data Collection:** Capturing and organizing training data
5. **Model Training:** Training classical ML models
6. **Real-time Processing:** Building interactive computer vision applications
7. **Error Handling:** Creating robust, user-friendly applications

---

**Good luck with your assignment!**
