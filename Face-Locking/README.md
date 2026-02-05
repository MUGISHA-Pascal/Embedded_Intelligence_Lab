# Face Locking System

Real-time face tracking that locks onto a specific person and detects their actions (blinking, smiling, head movements).

**See [GUIDE.md](GUIDE.md) for quick start guide**

---

## Quick Setup

### 1. Activate Virtual Environment (if using one)

```bash
source face_env/bin/activate  # macOS/Linux
# OR
face_env\Scripts\activate.bat  # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install opencv-python numpy onnxruntime mediapipe==0.10.9 pillow
```

### 3. Download Model

```bash
curl -L -o models/arcface.onnx \
  https://huggingface.co/onnxmodelzoo/arcfaceresnet100-8/resolve/main/arcfaceresnet100-8.onnx
```

### 4. Run

```bash
python3 main.py
```

That's it! The system will guide you through enrollment and face locking.

---

## Usage

### Simple Workflow

```bash
python3 main.py
```

The system will:

1. Check if you have enrolled faces
2. If not, automatically start enrollment (live camera capture)
3. If yes, let you choose:
   - Lock onto existing face
   - Enroll a new face

### What It Does

- **Locks onto your face** even when others are in frame
- **Detects actions** in real-time:
  - Blinking
  - Smiling
  - Head movements (left/right)
- **Logs everything** to `action_history/` with timestamps

### Visual Indicators

- **Red box** = Locked on you
- **Cyan box** = Other people
- **Green banner** = Lock status + action counts

### Controls

- Press **'q'** to quit

---

## How to Trigger Actions

| Action         | How to Do It                   |
| -------------- | ------------------------------ |
| **Blink**      | Close eyes deliberately        |
| **Smile**      | Smile naturally (lift corners) |
| **Move Left**  | Move head slowly left          |
| **Move Right** | Move head slowly right         |

**All actions now use the same accurate detection method!**

**Tips for Best Results:**

- **Good lighting** - Face should be well-lit from front
- **Face camera** - Look directly at camera
- **Distance** - Stay 2-3 feet from camera
- **Deliberate actions** - Make clear, intentional movements

**Tuning:** See [TUNING.md](TUNING.md) to adjust sensitivity

---

## File Structure

```
Face_Locking/
├── main.py                    # Main entry point (run this!)
├── live_enroll.py             # Live camera enrollment
├── enroll_me.py               # Photo-based enrollment (optional)
├── setup.sh / setup.bat       # Automated setup scripts
├── src/
│   ├── run_pipeline.py        # Face locking system
│   ├── action_detector.py     # Action detection
│   ├── face_locker.py         # Lock management
│   └── ...                    # Other modules
├── models/
│   └── arcface.onnx           # Face recognition model
├── data/identities/           # Enrolled faces
└── action_history/            # Action logs
```

---

## Troubleshooting

**Camera won't open?**

- Check camera permissions
- Close other apps using camera

**Actions not detected?**

- Better lighting
- More deliberate movements
- Face camera directly

**Multiple faces in frame?**

- System locks only on selected person
- Others shown with cyan boxes

---

## Requirements

- Python 3.8+
- Webcam
- No GPU needed (CPU only)

---

**Built for Intelligent Robotics Course**
