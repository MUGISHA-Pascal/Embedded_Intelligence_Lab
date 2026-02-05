#!/bin/bash
# Face Locking System - Quick Setup

echo "============================================================"
echo "FACE LOCKING SYSTEM - SETUP"
echo "============================================================"
echo ""

# Check Python
echo "Checking Python..."
if command -v python3 &> /dev/null; then
    echo "✓ Python 3 found"
else
    echo "✗ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install opencv-python numpy onnxruntime mediapipe==0.10.9 pillow

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

# Create directories
echo ""
echo "Creating directories..."
mkdir -p models data/identities action_history
echo "✓ Directories created"

# Download model
echo ""
echo "Downloading ArcFace model (~250MB)..."
if [ -f "models/arcface.onnx" ]; then
    echo "✓ Model already exists"
else
    curl -L -o models/arcface.onnx \
        https://huggingface.co/onnxmodelzoo/arcfaceresnet100-8/resolve/main/arcfaceresnet100-8.onnx
    
    if [ $? -eq 0 ]; then
        echo "✓ Model downloaded"
    else
        echo "✗ Failed to download model"
        exit 1
    fi
fi

echo ""
echo "============================================================"
echo "SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "Run the system:"
echo "  python main.py"
echo ""
