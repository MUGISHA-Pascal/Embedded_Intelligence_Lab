#!/usr/bin/env python3
"""
Test script for OCR Text Scanner
Verifies that all dependencies are installed and working correctly.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import PyQt5
        print("✓ PyQt5 imported successfully")
    except ImportError as e:
        print(f"✗ PyQt5 import failed: {e}")
        return False
        
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
        
    try:
        import pytesseract
        print("✓ PyTesseract imported successfully")
    except ImportError as e:
        print(f"✗ PyTesseract import failed: {e}")
        return False
        
    try:
        from PIL import Image
        print("✓ Pillow imported successfully")
    except ImportError as e:
        print(f"✗ Pillow import failed: {e}")
        return False
        
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
        
    return True

def test_tesseract():
    """Test if Tesseract OCR is properly installed"""
    print("\nTesting Tesseract OCR...")
    
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract version: {version}")
        return True
    except Exception as e:
        print(f"✗ Tesseract test failed: {e}")
        print("Please install Tesseract OCR:")
        print("  macOS: brew install tesseract")
        print("  Ubuntu: sudo apt-get install tesseract-ocr")
        print("  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        return False

def test_camera():
    """Test if camera is available"""
    print("\nTesting camera availability...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✓ Camera is available")
            cap.release()
            return True
        else:
            print("⚠ Camera not available (this is optional)")
            return True  # Camera is optional
    except Exception as e:
        print(f"⚠ Camera test failed: {e} (this is optional)")
        return True  # Camera is optional

def test_basic_ocr():
    """Test basic OCR functionality"""
    print("\nTesting basic OCR functionality...")
    
    try:
        import numpy as np
        import pytesseract
        from PIL import Image
        
        # Create a simple test image with text
        img_array = np.ones((100, 300, 3), dtype=np.uint8) * 255  # White background
        
        # Convert to PIL Image
        test_image = Image.fromarray(img_array)
        
        # Try OCR (should return empty or minimal text)
        result = pytesseract.image_to_string(test_image)
        print("✓ Basic OCR test completed")
        return True
        
    except Exception as e:
        print(f"✗ Basic OCR test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("OCR Text Scanner - Dependency Test")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("Tesseract Test", test_tesseract),
        ("Camera Test", test_camera),
        ("Basic OCR Test", test_basic_ocr)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 20)
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 40)
    print("TEST SUMMARY:")
    print("=" * 40)
    
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ All tests passed! The OCR Text Scanner should work correctly.")
        print("Run 'python src/main.py' to start the application.")
    else:
        print("✗ Some tests failed. Please check the installation instructions.")
        print("Refer to the README.md file for detailed setup instructions.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
