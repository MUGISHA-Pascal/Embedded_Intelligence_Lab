#!/bin/bash
# Quick dependency installer

echo "Installing dependencies in virtual environment..."
python3 -m pip install opencv-python numpy onnxruntime mediapipe==0.10.9 pillow

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully!"
else
    echo "✗ Installation failed"
    exit 1
fi
