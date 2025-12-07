import cv2
import pytesseract
import sys

# Preprocessing function to enhance image quality for OCR
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    return gray

# Function to select ROI and apply OCR
def recognize_number_plate(image_path):
    # Load the image from the provided file path
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Image file '{image_path}' could not be loaded.")
        return

    # Show the full image and allow the user to select a region of interest (ROI)
    roi = cv2.selectROI("Select ROI", image, False, False)

    # Extract the coordinates of the ROI
    x, y, w, h = roi
    roi_image = image[y:y+h, x:x+w]  # Crop the ROI from the image

    # Preprocess the selected ROI
    processed_roi = preprocess_image(roi_image)

    # Apply OCR on the preprocessed ROI
    custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(processed_roi, config=custom_config)

    # Display the detected text
    print("Number Plate:", text.strip())

    # Display the selected ROI for visual confirmation
    cv2.imshow('Selected ROI', roi_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main block to execute the function using argv
if __name__ == "__main__":
    # Check if image path argument is provided
    if len(sys.argv) != 2:
        print("Usage: python read_number_plate.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    recognize_number_plate(image_path)