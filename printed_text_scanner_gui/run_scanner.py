#!/usr/bin/env python3
"""
Launcher script for OCR Text Scanner
This script provides an easy way to start the application with proper error handling.
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        'PyQt5',
        'pytesseract', 
        'opencv-python',
        'Pillow',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'PyQt5':
                import PyQt5
            elif package == 'pytesseract':
                import pytesseract
            elif package == 'Pillow':
                from PIL import Image
            elif package == 'numpy':
                import numpy
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def check_tesseract():
    """Check if Tesseract OCR is installed"""
    print("\nChecking Tesseract OCR...")
    
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract OCR version: {version}")
        return True
    except Exception as e:
        print(f"✗ Tesseract OCR not found: {e}")
        print("\nPlease install Tesseract OCR:")
        print("  macOS: brew install tesseract")
        print("  Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        print("  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        return False

def main():
    """Main launcher function"""
    print("OCR Text Scanner Launcher")
    print("=" * 30)
    
    # Check if we're in the right directory
    if not os.path.exists('src/main.py'):
        print("Error: Please run this script from the OCR-Text-Scanner directory")
        print("Current directory:", os.getcwd())
        return 1
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check Tesseract
    if not check_tesseract():
        return 1
    
    print("\n" + "=" * 30)
    print("All dependencies verified!")
    print("Starting OCR Text Scanner...")
    print("=" * 30)
    
    # Launch the application
    try:
        # Add src directory to Python path and run main.py
        env = os.environ.copy()
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"src:{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = "src"
        
        subprocess.run([sys.executable, 'src/main.py'], env=env, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error running application: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nApplication closed by user")
        return 0
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
