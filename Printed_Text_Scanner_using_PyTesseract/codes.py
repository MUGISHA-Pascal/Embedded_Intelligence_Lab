import cv2
import pytesseract
import numpy as np
import os
import time # type: ignore

# Custom save directory
SAVE_DIR = "scanned_texts"
IMAGE_FILENAME = "captured_image.jpg"
ROI_FILENAME = "cropped_roi.jpg"

# Full paths
IMAGE_PATH = os.path.join(SAVE_DIR, IMAGE_FILENAME)
ROI_PATH = os.path.join(SAVE_DIR, ROI_FILENAME)

# Ensure the save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

def select_camera():
    """ Allow the user to choose the camera if multiple are available. """
    print("\n🔍 Detecting available cameras...")
    available_cameras = []
    
    for i in range(5):  # Check up to 5 cameras
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            print(f"✅ Camera found at index {i}")
            cap.release()
        else:
            print(f"❌ No camera at index {i}")

    if not available_cameras:
        print("❌ No cameras detected. Exiting...")
        exit()

    # Let the user choose the camera
    while True:
        cam_index = int(input(f"\n🎥 Select camera index from {available_cameras}: "))
        if cam_index in available_cameras:
            return cam_index
        print("⚠️ Invalid selection! Choose from the detected cameras.")

def capture_image():
    """ Captures an image from the selected camera. """
    cam_index = select_camera()  # Ask user for preferred camera
    cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        print("❌ Error: Camera not accessible. Exiting.") # type: ignore
        cleanup_and_exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Warning: Failed to capture frame. Retrying...") # type: ignore
            continue

        overlay_text(frame, "Instructions:", (20, 50), (255, 0, 0), font_scale=3.2, thickness=3)
        overlay_text(frame, "Step 1: Press 'C' to Capture", (20, 90), (255, 0, 0))
        overlay_text(frame, "Step 2: Select the Region of Interest", (20, 130), (255, 0, 0))
        overlay_text(frame, "Step 3: Confirm OCR Text", (20, 170), (255, 0, 0))
        overlay_text(frame, "Press 'Q' to Quit", (20, 210), (0, 0, 255))
        overlay_text(frame, "_________________________", (20, 230), (255, 0, 0))
        cv2.imshow("Capture Mode", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'): # type: ignore
            cv2.imwrite(IMAGE_PATH, frame)
            cap.release()
            cv2.destroyAllWindows()
            return select_roi(IMAGE_PATH)
        elif key == ord('q'): # type: ignore
            cap.release()
            cv2.destroyAllWindows()
            cleanup_and_exit()

def select_roi(image_path):
    """ Opens the captured image and allows the user to select a Region of Interest (RoI). """
    image = cv2.imread(image_path)

    if image is None:
        print("Warning:: Error: Image could not be loaded.") # type: ignore
        return None

    while True:
        cv2.imshow("Select RoI", image)
        roi = cv2.selectROI("Select RoI", image, fromCenter=False, showCrosshair=True)

        if roi[2] == 0 or roi[3] == 0:
            print("Warning:: No RoI selected. Try again.") # type: ignore
            continue  # Let user retry

        # Crop the selected region
        cropped_roi_path = ROI_PATH
        cropped_roi = image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] # type: ignore
        cv2.imwrite(cropped_roi_path, cropped_roi)
        cv2.destroyAllWindows()
        return extract_text_and_display(cropped_roi_path)

def preprocess_image(image_path):
    """ Preprocesses the image to enhance text recognition. """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text_from_image(image_path):
    """ Extracts text from the cropped RoI using OCR. """
    processed_image = preprocess_image(image_path)
    return pytesseract.image_to_string(processed_image, config="--psm 6")

def overlay_text(image, text, position, color=(0, 255, 0), font_scale=0.3, thickness=1):
    """ Draws overlayed text on the image. """
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def generate_unique_filename():
    """ Generates a unique filename for each saved scan. """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(SAVE_DIR, f"scanned_text_{timestamp}.txt")

def save_text(text):
    """ Saves the extracted text to a unique file. """
    filename = generate_unique_filename()
    with open(filename, "w", encoding="utf-8") as f: # type: ignore
        f.write(text)
    print(f"info:: Extracted text saved at: {filename}") # type: ignore

def extract_text_and_display(image_path):
    """ Extracts text and displays it instead of the cropped image. """
    extracted_text = extract_text_from_image(image_path)
    
    # Create a blank white image for displaying text
    text_display = np.ones((500, 800, 3), dtype=np.uint8) * 255  # White background
    
    # Break the text into multiple lines for display
    lines = extracted_text.split("\n")
    y = 50
    for line in lines:
        if line.strip():
            overlay_text(text_display, line, (20, y), (0, 0, 0), font_scale=0.7)
            y += 30  # Adjust spacing
    
    overlay_text(text_display, "info:: Done! Press 'S' to Save, 'R' to Retake, 'Q' to Quit", (20, 480), (255, 0, 0))

    while True:
        cv2.imshow("OCR Result", text_display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('s'):  # type: ignore # Save extracted text
            save_text(extracted_text)
            overlay_text(text_display, "info:: Saved Successfully!", (20, 420), (0, 255, 0))
            cv2.imshow("OCR Result", text_display)
            cv2.waitKey(1000)  # Show message briefly
            cleanup_and_exit()
            break
        elif key == ord('r'):  # type: ignore # Retake image
            main()
            return
        elif key == ord('q'):  # type: ignore # Quit
            cleanup_and_exit()
            break

    cv2.destroyAllWindows()

def cleanup_and_exit():
    """ Deletes temporary images and exits the program. """
    if os.path.exists(IMAGE_PATH):
        os.remove(IMAGE_PATH)
    if os.path.exists(ROI_PATH):
        os.remove(ROI_PATH)
    print("🧹 Temporary files cleaned. Exiting program.") # type: ignore
    cv2.destroyAllWindows()
    exit() # type: ignore

def main():
    """ Main function that captures an image, allows RoI selection, and performs OCR. """
    capture_image()

if __name__ == "__main__":
    main()