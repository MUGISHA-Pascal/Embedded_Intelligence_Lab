"""
Face Capture Script - AI Without ML Assignment
Uses MediaPipe for face detection to capture training images
Part 1 of 3: Data Collection
"""
import cv2
import mediapipe as mp
import os
import time
from datetime import datetime


class FaceCapture:
    def __init__(self, dataset_dir="datasets"):
        """
        Initialize Face Capture system
        Args:
            dataset_dir: Directory to save captured face images
        """
        self.dataset_dir = dataset_dir
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for short-range (2m), 1 for full-range (5m)
            min_detection_confidence=0.5
        )
        
        # Create dataset directory if it doesn't exist
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
            print(f"✓ Created dataset directory: {self.dataset_dir}")
    
    def detect_face(self, img):
        """
        Detect face in image using MediaPipe
        Args:
            img: Input image (BGR format)
        Returns:
            face_bbox: Bounding box coordinates (x, y, w, h) or None
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.face_detection.process(img_rgb)
        
        if results.detections:
            # Get first detected face
            detection = results.detections[0]
            
            # Get bounding box
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            
            # Convert to pixel coordinates
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            bbox_w = int(bboxC.width * w)
            bbox_h = int(bboxC.height * h)
            
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            bbox_w = min(bbox_w, w - x)
            bbox_h = min(bbox_h, h - y)
            
            return (x, y, bbox_w, bbox_h)
        
        return None
    
    def capture_faces(self, person_name, num_images=50):
        """
        Capture face images for a person
        Args:
            person_name: Name of the person
            num_images: Number of images to capture (default: 50)
        """
        # Create person directory
        person_dir = os.path.join(self.dataset_dir, person_name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
            print(f"✓ Created directory for {person_name}")
        else:
            print(f"⚠ Directory already exists for {person_name}")
            response = input("Continue and add more images? (y/n): ")
            if response.lower() != 'y':
                return
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open webcam!")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print(f"\n{'='*60}")
        print(f"Capturing {num_images} images for: {person_name}")
        print(f"{'='*60}")
        print("\nINSTRUCTIONS:")
        print("  - Position your face in the frame")
        print("  - Move slightly between captures for variety")
        print("  - Press SPACE to capture an image")
        print("  - Press Q to quit early")
        print(f"\nStarting in 3 seconds...\n")
        time.sleep(3)
        
        count = 0
        last_capture_time = 0
        capture_delay = 0.5  # Minimum delay between captures (seconds)
        
        while count < num_images:
            success, img = cap.read()
            if not success:
                print("WARNING: Failed to read frame")
                continue
            
            # Flip for selfie view
            img = cv2.flip(img, 1)
            
            # Detect face
            face_bbox = self.detect_face(img)
            
            # Draw on image
            display_img = img.copy()
            
            if face_bbox:
                x, y, w, h = face_bbox
                # Draw green rectangle around face
                cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw face status
                cv2.putText(display_img, "Face Detected", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # Draw warning
                cv2.putText(display_img, "No Face Detected!", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Draw progress bar
            progress = int((count / num_images) * 100)
            bar_width = 400
            bar_height = 30
            bar_x = 50
            bar_y = display_img.shape[0] - 100
            
            # Background
            cv2.rectangle(display_img, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            # Progress
            cv2.rectangle(display_img, (bar_x, bar_y), 
                         (bar_x + int(bar_width * progress / 100), bar_y + bar_height), 
                         (0, 255, 0), -1)
            # Border
            cv2.rectangle(display_img, (bar_x, bar_y), 
                         (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
            
            # Draw counter
            counter_text = f"{count}/{num_images} images captured ({progress}%)"
            cv2.putText(display_img, counter_text, (bar_x, bar_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw instructions
            cv2.putText(display_img, "Press SPACE to capture | Q to quit", 
                       (50, display_img.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show image
            cv2.imshow(f'Capturing Faces - {person_name}', display_img)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            current_time = time.time()
            
            if key == ord(' '):  # Space bar
                if face_bbox and (current_time - last_capture_time) > capture_delay:
                    # Extract face region
                    x, y, w, h = face_bbox
                    face_img = img[y:y+h, x:x+w]
                    
                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"{person_name}_{count:03d}_{timestamp}.jpg"
                    filepath = os.path.join(person_dir, filename)
                    cv2.imwrite(filepath, face_img)
                    
                    count += 1
                    last_capture_time = current_time
                    print(f"✓ Captured image {count}/{num_images}")
                elif not face_bbox:
                    print("⚠ No face detected! Please position your face in the frame.")
            elif key == ord('q'):
                print("\n⚠ Capture cancelled by user")
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        if count >= num_images:
            print(f"\n{'='*60}")
            print(f"✓ Successfully captured {count} images for {person_name}")
            print(f"✓ Saved to: {person_dir}")
            print(f"{'='*60}\n")
        else:
            print(f"\n✓ Captured {count} images (stopped early)")


def main():
    """Main function to run face capture"""
    print("="*60)
    print("Face Capture Script - AI Without ML Assignment")
    print("Part 1/3: Data Collection using MediaPipe")
    print("="*60)
    
    # Initialize capture system
    capture = FaceCapture(dataset_dir="datasets")
    
    while True:
        print("\n" + "="*60)
        person_name = input("Enter person's name (or 'quit' to exit): ").strip()
        
        if person_name.lower() == 'quit':
            print("\n✓ Exiting capture script")
            break
        
        if not person_name:
            print("⚠ Name cannot be empty!")
            continue
        
        # Sanitize name (remove special characters)
        person_name = "".join(c for c in person_name if c.isalnum() or c in (' ', '_', '-'))
        
        if not person_name:
            print("⚠ Invalid name!")
            continue
        
        # Ask for number of images
        try:
            num_str = input("Number of images to capture (default: 50): ").strip()
            num_images = int(num_str) if num_str else 50
            
            if num_images <= 0:
                print("⚠ Number must be positive! Using default (50)")
                num_images = 50
        except ValueError:
            print("⚠ Invalid number! Using default (50)")
            num_images = 50
        
        # Capture faces
        capture.capture_faces(person_name, num_images)
        
        # Ask if user wants to capture more people
        response = input("\nCapture another person? (y/n): ").strip().lower()
        if response != 'y':
            break
    
    print("\n" + "="*60)
    print("✓ Face capture complete!")
    print("Next step: Run '2_train_model.py' to train the recognizer")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nMake sure you have installed all required packages:")
        print("  pip install -r requirements.txt")
