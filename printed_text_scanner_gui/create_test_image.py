#!/usr/bin/env python3
"""
Script to create a test image with text for OCR testing
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_test_image():
    """Create a test image with sample text"""
    
    # Create a white background image
    width, height = 800, 600
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Sample text to draw
    texts = [
        "OCR Text Scanner Test",
        "This is a sample document for testing",
        "optical character recognition capabilities.",
        "",
        "Features to test:",
        "• Text extraction",
        "• ROI selection", 
        "• Live camera input",
        "• Text overlay preview",
        "",
        "The quick brown fox jumps over the lazy dog.",
        "1234567890 !@#$%^&*()",
        "",
        "Different font sizes and styles can be detected.",
        "PyTesseract works best with clear, high-contrast text."
    ]
    
    # Try to use a system font, fallback to default if not available
    try:
        # Try common system fonts
        font_paths = [
            "/System/Library/Fonts/Arial.ttf",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "C:/Windows/Fonts/arial.ttf"  # Windows
        ]
        
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 24)
                break
        
        if font is None:
            font = ImageFont.load_default()
            
    except Exception:
        font = ImageFont.load_default()
    
    # Draw text on image
    y_position = 50
    line_height = 35
    
    for text in texts:
        if text.strip():  # Skip empty lines for spacing
            draw.text((50, y_position), text, fill='black', font=font)
        y_position += line_height
    
    # Add some geometric shapes for testing
    draw.rectangle([50, y_position + 20, 200, y_position + 60], outline='black', width=2)
    draw.text((60, y_position + 30), "Boxed Text", fill='black', font=font)
    
    # Save the image
    output_path = os.path.join('assets', 'test_image.png')
    os.makedirs('assets', exist_ok=True)
    image.save(output_path)
    
    print(f"Test image created: {output_path}")
    print("You can use this image to test the OCR functionality.")
    
    return output_path

if __name__ == "__main__":
    create_test_image()
