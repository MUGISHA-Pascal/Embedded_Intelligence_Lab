"""
Face Recognition Script - AI Without ML Assignment
Real-time face recognition using MediaPipe + LBPH
Part 3 of 3: Detection and Recognition Pipeline
"""
import cv2
import mediapipe as mp
import json
import os
import time


class FaceRecognizer:
    def __init__(self, model_dir="models"):
        """
        Initialize Face Recognition System
        Args:
            model_dir: Directory containing trained model and label map
        """
        self.model_dir = model_dir
        
        # Initialize MediaPipe Face Detection (AI without ML)
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )
        
        # Initialize LBPH Face Recognizer (Classical CV)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Load trained model
        self.load_model()
    
    def load_model(self):
        """Load trained LBPH model and label map"""
        model_path = os.path.join(self.model_dir, "lbph_face_recognizer.yml")
        label_map_path = os.path.join(self.model_dir, "label_map.json")
        
        if not os.path.exists(model_path):
            raise Exception(f"Model not found: {model_path}\nPlease run '2_train_model.py' first")
        
        if not os.path.exists(label_map_path):
            raise Exception(f"Label map not found: {label_map_path}\nPlease run '2_train_model.py' first")
        
        # Load model
        self.recognizer.read(model_path)
        print(f"✓ Loaded model from: {model_path}")
        
        # Load label map
        with open(label_map_path, 'r') as f:
            label_map_str = json.load(f)
            # Convert string keys to integers
            self.label_map = {int(k): v for k, v in label_map_str.items()}
        
        print(f"✓ Loaded label map: {len(self.label_map)} person(s)")
        print(f"  People in model: {', '.join(self.label_map.values())}\n")
    
    def detect_face(self, img):
        """
        Detect face using MediaPipe
        Args:
            img: Input image (BGR format)
        Returns:
            face_bbox: Bounding box (x, y, w, h) or None
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
            
            # Ensure coordinates are within bounds
            x = max(0, x)
            y = max(0, y)
            bbox_w = min(bbox_w, w - x)
            bbox_h = min(bbox_h, h - y)
            
            return (x, y, bbox_w, bbox_h)
        
        return None
    
    def recognize_face(self, face_img):
        """
        Recognize face using LBPH
        Args:
            face_img: Face image (BGR format)
        Returns:
            name: Recognized person's name
            confidence: Recognition confidence (lower is better)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Resize to match training size
        gray = cv2.resize(gray, (200, 200))
        
        # Predict
        label, confidence = self.recognizer.predict(gray)
        
        # Get name from label map
        name = self.label_map.get(label, "Unknown")
        
        return name, confidence
    
    def run(self):
        """Run real-time face recognition"""
        print(f"\n{'='*60}")
        print("Starting Real-Time Face Recognition")
        print(f"{'='*60}\n")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open webcam!")
            return
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("KEYBOARD CONTROLS:")
        print("  Q or ESC - Quit")
        print("  S - Save screenshot")
        print("\nStarting recognition...\n")
        
        # FPS tracking
        prev_time = 0
        
        # Recognition threshold (lower confidence = better match)
        confidence_threshold = 70  # Adjust based on your needs
        
        while True:
            success, img = cap.read()
            if not success:
                print("WARNING: Failed to read frame")
                continue
            
            # Flip for selfie view
            img = cv2.flip(img, 1)
            
            # Detect face
            face_bbox = self.detect_face(img)
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            
            # Process detected face
            if face_bbox:
                x, y, w, h = face_bbox
                
                # Extract face region
                face_img = img[y:y+h, x:x+w]
                
                if face_img.size > 0:
                    # Recognize face
                    name, confidence = self.recognize_face(face_img)
                    
                    # Determine if recognized
                    is_recognized = confidence < confidence_threshold
                    
                    # Choose color based on recognition
                    if is_recognized:
                        color = (0, 255, 0)  # Green for recognized
                        status = "Recognized"
                    else:
                        color = (0, 0, 255)  # Red for unknown
                        status = "Unknown"
                        name = "Unknown"
                    
                    # Draw rectangle around face
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    
                    # Draw name and confidence
                    label = f"{name} ({confidence:.1f})"
                    
                    # Background for text
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(img, (x, y - 35), (x + text_w + 10, y), color, -1)
                    
                    # Draw text
                    cv2.putText(img, label, (x + 5, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Draw status
                    cv2.putText(img, status, (x, y + h + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw info overlay
            h, w, c = img.shape
            
            # Semi-transparent overlay
            overlay = img.copy()
            cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
            img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
            
            # Display FPS
            cv2.putText(img, f'FPS: {int(fps)}', (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Display detection status
            detection_text = "Face Detected" if face_bbox else "No Face"
            detection_color = (0, 255, 0) if face_bbox else (0, 0, 255)
            cv2.putText(img, f'Status: {detection_text}', (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, detection_color, 2)
            
            # Display model info
            cv2.putText(img, f'Model: LBPH + MediaPipe', (20, 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display controls
            cv2.putText(img, 'Press Q to quit, S to screenshot', (20, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show image
            cv2.imshow('Face Recognition - MediaPipe + LBPH', img)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                print("\nExiting...")
                break
            elif key == ord('s'):  # Save screenshot
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recognition_{timestamp}.png"
                cv2.imwrite(filename, img)
                print(f"✓ Screenshot saved: {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n✓ Face recognition stopped")


def main():
    """Main function to run face recognition"""
    print("="*60)
    print("Face Recognition Script - AI Without ML Assignment")
    print("Part 3/3: Real-Time Recognition using MediaPipe + LBPH")
    print("="*60)
    
    try:
        # Initialize recognizer
        recognizer = FaceRecognizer(model_dir="models")
        
        # Run recognition
        recognizer.run()
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nMake sure you have:")
        print("  1. Captured faces using '1_capture_faces.py'")
        print("  2. Trained model using '2_train_model.py'")
        print("  3. Installed all dependencies: pip install -r requirements.txt")
        return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
    except Exception as e:
        print(f"\nERROR: {str(e)}")
