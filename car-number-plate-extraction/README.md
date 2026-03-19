# Car Number Plate Extraction

A computer vision project for real-time detection, alignment, OCR, validation, and temporal confirmation of vehicle number plates from video feed.

## Features

- **Plate Detection**: Identifies potential license plate regions using contour analysis and aspect ratio filtering.
- **Alignment**: Warps detected plates to a standardized orientation for accurate OCR.
- **OCR**: Extracts text from aligned plates using Tesseract with character whitelisting.
- **Validation**: Validates extracted text against a regex pattern for standard plate formats (e.g., ABC123D).
- **Temporal Confirmation**: Uses a buffer and majority voting to confirm plates over multiple frames, reducing false positives.

## Project Structure

```
car-number-plate-extraction/
├── src/
│   ├── align.py      # Plate alignment and warping
│   ├── camera.py     # Camera interface utilities
│   ├── detect.py     # Basic plate detection
│   ├── ocr.py        # OCR extraction from plates
│   ├── temporal.py   # Temporal validation with CSV logging
│   └── validate.py   # OCR validation stage
├── data/             # (Optional) Sample images or videos
├── book/             # Documentation or notebooks
├── README.md         # This file
└── requirements.txt  # Dependencies
```

## Requirements

- Python 3.7+
- OpenCV (opencv-python)
- NumPy
- Tesseract OCR engine
- pytesseract (Python wrapper for Tesseract)

## Installation

1. Clone or download the repository.

2. Install Python dependencies:

   ```bash
   pip install opencv-python numpy pytesseract
   ```

3. Install Tesseract OCR:
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`
   - **Windows**: Download from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

4. (Optional) Set up a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

Each script can be run independently to demonstrate different stages of the pipeline:

### Basic Detection (`detect.py`)

Detects and highlights potential plate regions without OCR.

```bash
python src/detect.py
```

### Alignment (`align.py`)

Detects and aligns plates, showing the warped output.

```bash
python src/align.py
```

### OCR (`ocr.py`)

Performs OCR on detected and aligned plates.

```bash
python src/ocr.py
```

### Validation (`validate.py`)

Validates OCR output against plate format regex.

```bash
python src/validate.py
```

### Temporal Confirmation (`temporal.py`)

Runs the full pipeline with temporal validation and logs confirmed plates to `plates_log.csv`.

```bash
python src/temporal.py
```

**Controls:**

- Press `q` to quit any running script.
- Windows will open showing the video feed with overlays and (where applicable) aligned/thresholded plates.

## Configuration

Key parameters (defined in each script):

- `MIN_AREA`: Minimum contour area for plate candidates (default: 600)
- `AR_MIN`, `AR_MAX`: Aspect ratio bounds for plates (default: 2.0 - 8.0)
- `W_OUT`, `H_OUT`: Output dimensions for aligned plates (default: 450x140)
- `BUFFER_SIZE`: Number of frames for temporal buffering (default: 5)
- `COOLDOWN`: Minimum time between logging the same plate (default: 10 seconds)

## Output

- **Console**: Logs confirmed plates when saved (in `temporal.py`).
- **CSV Log**: `plates_log.csv` contains timestamps and confirmed plate numbers.
- **Video Windows**:
  - Main feed with detected plates highlighted
  - Aligned plate image
  - Thresholded image for OCR

## Notes

- Assumes camera index 0 (default webcam). Modify `cv2.VideoCapture(0)` if needed.
- Plate format regex assumes standard format like `ABC123D`. Adjust `PLATE_RE` for other formats.
- Performance depends on lighting, angle, and camera quality.
- For production use, consider GPU acceleration, model-based detection (e.g., YOLO), or cloud OCR services.

## License

[Add license information here, e.g., MIT License]
