@echo off
REM Face Locking System - Quick Setup

echo ============================================================
echo FACE LOCKING SYSTEM - SETUP
echo ============================================================
echo.

REM Check Python
echo Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [X] Python not found. Please install Python 3.8+
    pause
    exit /b 1
)
echo [OK] Python found

REM Install dependencies
echo.
echo Installing dependencies...
pip install opencv-python numpy onnxruntime mediapipe==0.10.9 pillow

if %errorlevel% equ 0 (
    echo [OK] Dependencies installed
) else (
    echo [X] Failed to install dependencies
    pause
    exit /b 1
)

REM Create directories
echo.
echo Creating directories...
if not exist models mkdir models
if not exist data\identities mkdir data\identities
if not exist action_history mkdir action_history
echo [OK] Directories created

REM Download model
echo.
echo Downloading ArcFace model (~250MB)...
if exist models\arcface.onnx (
    echo [OK] Model already exists
) else (
    curl -L -o models\arcface.onnx https://huggingface.co/onnxmodelzoo/arcfaceresnet100-8/resolve/main/arcfaceresnet100-8.onnx
    
    if %errorlevel% equ 0 (
        echo [OK] Model downloaded
    ) else (
        echo [X] Failed to download model
        pause
        exit /b 1
    )
)

echo.
echo ============================================================
echo SETUP COMPLETE!
echo ============================================================
echo.
echo Run the system:
echo   python main.py
echo.
pause
