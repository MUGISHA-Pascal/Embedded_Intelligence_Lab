# OCR Text Scanner

A GUI-based printed text scanner built with PyQt5 and PyTesseract for optical character recognition with advanced features including ROI selection, live camera input, and text overlay preview.

## Features

- **GUI Interface**: Modern PyQt5-based application window
- **Image Loading**: Support for PNG, JPG, JPEG, BMP, TIFF, GIF formats
- **OCR Processing**: Text extraction using PyTesseract with preprocessing
- **ROI Selection**: Click and drag to select specific regions for OCR
- **Live Camera Input**: Real-time camera feed with OCR capabilities
- **Text Overlay**: Visual preview showing detected text boundaries on images

## Installation

### Prerequisites

- Python 3.7+
- Tesseract OCR engine

### Install Tesseract OCR

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

### Install Dependencies

```bash
cd OCR-Text-Scanner
pip install -r requirements.txt
```

## How to Run

```bash
python src/main.py
```

Or use the launcher script:

```bash
python run_scanner.py
```

## Usage

1. **Load Image**: Click "Load Image" to select an image file
2. **Select ROI**: Click and drag on image to select text region (optional)
3. **Run OCR**: Click "Run OCR" to extract text
4. **View Results**: Extracted text appears in the results panel
5. **Camera Mode**: Click "Start Camera" for live OCR processing

## Project Structure

```
OCR-Text-Scanner/
├── src/
│   └── main.py              # Main GUI application
├── tests/
│   └── test_ocr.py         # Dependency verification
├── assets/
│   └── test_image.png      # Sample test image
├── requirements.txt         # Python dependencies
├── run_scanner.py          # Application launcher
└── README.md               # This file
```

## Dependencies

- **PyQt5**: GUI framework
- **pytesseract**: OCR engine wrapper
- **opencv-python**: Image processing
- **Pillow**: Image handling
- **numpy**: Numerical operations

## Key Components

- **ImageLabel**: Custom widget for image display and ROI selection
- **CameraThread**: Threaded camera input handling
- **OCRProcessor**: Text extraction and image preprocessing
- **MainWindow**: Main application interface and coordination
