# Face Recognition with ArcFace ONNX and 5-Point Alignment

Reproducing Gabriel Baziramwabo's book project (Rwanda Coding Academy, Jan 2026). CPU-only, webcam-based system for enrollment and live recognition.

## Setup
1. Clone: `git clone https://github.com/Mugisha-pascal/Embedded_Intelligence_Lab/face-recognition-arcface-5pt.git`
2. Install: `pip install opencv-python numpy onnxruntime scipy tqdm mediapipe`
3. Download model: [ArcFace ONNX](https://github.com/deepinsight/insightface/raw/master/models/arcface_r50.onnx) → `models/`
4. Run `python init_project.py` (if needed).

## How to Run
- Test stages: `python -m src.camera`, `python -m src.detect`, etc. (see src/ for details).
- Enroll: `python -m src.enroll` (name → capture 15+ samples → 's' to save).
- Evaluate threshold: `python -m src.evaluate` (uses enroll crops).
- Live recognize: `python -m src.recognize` ('q' quit, +/- adjust thresh).

## How Enrollment Works
- Detects face (Haar) → 5pt landmarks (MediaPipe) → Aligns to 112x112 → Extracts 512D ArcFace embedding (ONNX, L2-normalized).
- Averages multiple samples per identity → Stores in `data/db/face_db.npz`.
- Open-set: Unknowns rejected if cosine distance > threshold.

## Threshold Used
From evaluation: 0.34 (cosine distance; equiv. sim=0.66). FAR~1%, FRR~5%. Adjust in `recognize.py`.

## Notes
- Enrolled 10 identities (e.g.,pascal, classmates).
- Runs on laptop CPU (~15 FPS). For GPU, add ONNX CUDA provider.
- Quiz prep: Pipeline = Detection → Alignment → Embedding → Cosine Match.

Built by Mugisha pascal (Kigali, RW) – Enjoy!