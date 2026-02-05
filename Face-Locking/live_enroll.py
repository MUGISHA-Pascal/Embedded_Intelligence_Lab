#!/usr/bin/env python3
"""
Live Face Enrollment System
Captures face images from webcam, enrolls identity, and immediately tests locking.
"""

import os
import sys
import time
from typing import List

import cv2
import numpy as np

from src.camera import camera_stream
from src.detect import detect_faces
from src.align import align_face
from src.embed import ArcFaceEmbedder
from src.enroll import enroll_identity
from src.utils import ensure_dir


class LiveEnrollmentCapture:
    """Captures face images from live camera for enrollment."""
    
    def __init__(self, target_samples: int = 5):
        self.target_samples = target_samples
        self.captured_frames: List[np.ndarray] = []
        self.capture_cooldown = 0
        self.cooldown_frames = 15  # ~0.5 seconds at 30fps
        
    def capture_frame(self, frame: cv2.Mat, boxes: List) -> bool:
        """
        Attempt to capture a frame for enrollment.
        
        Returns:
            True if frame was captured, False otherwise
        """
        if self.capture_cooldown > 0:
            self.capture_cooldown -= 1
            return False
        
        if len(boxes) != 1:
            return False
        
        # Capture the frame
        self.captured_frames.append(frame.copy())
        self.capture_cooldown = self.cooldown_frames
        return True
    
    def is_complete(self) -> bool:
        """Check if we have enough samples."""
        return len(self.captured_frames) >= self.target_samples
    
    def get_progress(self) -> tuple:
        """Get current progress (captured, target)."""
        return len(self.captured_frames), self.target_samples


def draw_capture_ui(frame: cv2.Mat, progress: tuple, boxes: List, message: str = "") -> None:
    """Draw capture interface on frame."""
    height, width = frame.shape[:2]
    captured, target = progress
    
    # Draw header
    cv2.rectangle(frame, (0, 0), (width, 120), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (width, 120), (0, 255, 255), 3)
    
    # Title
    cv2.putText(frame, "LIVE FACE ENROLLMENT", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Progress
    progress_text = f"Captured: {captured}/{target}"
    cv2.putText(frame, progress_text, (20, 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Progress bar
    bar_width = width - 40
    bar_x = 20
    bar_y = 90
    bar_height = 20
    
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
    if target > 0:
        filled_width = int((captured / target) * bar_width)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), (0, 255, 0), -1)
    
    # Message
    if message:
        cv2.putText(frame, message, (20, height - 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Instructions
    instructions = [
        "Position your face in the frame",
        "Look at the camera naturally",
        "Press 'q' to cancel",
    ]
    
    y_offset = height - 50
    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (20, y_offset + i * 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Draw face boxes
    for box in boxes:
        x1, y1, x2, y2 = box
        color = (0, 255, 0) if len(boxes) == 1 else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        if len(boxes) > 1:
            cv2.putText(frame, "Multiple faces detected!", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)


def capture_live_samples(target_samples: int = 5) -> List[np.ndarray]:
    """
    Capture face samples from live camera.
    
    Args:
        target_samples: Number of samples to capture
        
    Returns:
        List of captured frames
    """
    print("\n" + "=" * 60)
    print("LIVE FACE CAPTURE")
    print("=" * 60)
    print(f"\nWe'll capture {target_samples} images of your face.")
    print("Position yourself in front of the camera.")
    print("Images will be captured automatically with slight delays.")
    print("\nPress any key to start...")
    input()
    
    capturer = LiveEnrollmentCapture(target_samples)
    last_capture_time = 0
    
    try:
        for frame in camera_stream():
            # Detect faces
            boxes = detect_faces(frame, min_confidence=0.7)
            
            # Attempt capture
            current_time = time.time()
            captured = False
            message = ""
            
            if len(boxes) == 0:
                message = "No face detected - move closer"
            elif len(boxes) > 1:
                message = "Multiple faces detected - ensure only you are visible"
            elif current_time - last_capture_time > 0.5:
                if capturer.capture_frame(frame, boxes):
                    captured = True
                    last_capture_time = current_time
                    progress = capturer.get_progress()
                    message = f"Captured {progress[0]}/{progress[1]}!"
                    print(f"✓ Captured sample {progress[0]}/{progress[1]}")
            
            # Draw UI
            draw_capture_ui(frame, capturer.get_progress(), boxes, message)
            
            cv2.imshow("Live Enrollment Capture", frame)
            
            # Check for completion
            if capturer.is_complete():
                print(f"\n✓ Successfully captured {target_samples} samples!")
                time.sleep(1)
                break
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n✗ Capture cancelled by user")
                cv2.destroyAllWindows()
                return []
        
        cv2.destroyAllWindows()
        return capturer.captured_frames
    
    except Exception as e:
        print(f"\n✗ Error during capture: {e}")
        cv2.destroyAllWindows()
        return []


def save_captured_frames(frames: List[np.ndarray], temp_dir: str = "temp_enrollment") -> List[str]:
    """Save captured frames to temporary directory."""
    ensure_dir(temp_dir)
    paths = []
    
    for idx, frame in enumerate(frames):
        path = os.path.join(temp_dir, f"capture_{idx:04d}.jpg")
        cv2.imwrite(path, frame)
        paths.append(path)
    
    return paths


def cleanup_temp_files(temp_dir: str) -> None:
    """Remove temporary enrollment files."""
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)


def main():
    """Main enrollment workflow."""
    print("\n" + "=" * 60)
    print("LIVE FACE ENROLLMENT SYSTEM")
    print("=" * 60)
    
    # Get identity name
    print("\nEnter your name for enrollment:")
    name = input("Name: ").strip()
    
    if not name:
        print("✗ Name cannot be empty")
        return
    
    # Check if already enrolled
    identities_dir = "data/identities"
    identity_path = os.path.join(identities_dir, name)
    if os.path.exists(identity_path):
        print(f"\n⚠ Identity '{name}' already exists!")
        response = input("Overwrite? (yes/no): ").strip().lower()
        if response != "yes":
            print("✗ Enrollment cancelled")
            return
    
    # Capture samples
    print(f"\n✓ Starting live capture for: {name}")
    frames = capture_live_samples(target_samples=5)
    
    if not frames:
        print("✗ No frames captured. Enrollment failed.")
        return
    
    # Save temporary files
    print("\n✓ Processing captured images...")
    temp_dir = "temp_enrollment"
    image_paths = save_captured_frames(frames, temp_dir)
    
    # Enroll identity
    print(f"✓ Enrolling identity: {name}")
    embedder = ArcFaceEmbedder("models/arcface.onnx")
    
    try:
        count, folder = enroll_identity(
            name=name,
            image_paths=image_paths,
            embedder=embedder,
            output_dir=identities_dir,
            detection_confidence=0.6,
        )
        
        if count > 0:
            print(f"\n✓ Successfully enrolled {count} samples for '{name}'")
            print(f"✓ Data saved to: {folder}")
            
            # Ask if user wants to test
            print("\n" + "=" * 60)
            response = input("Start face locking test now? (yes/no): ").strip().lower()
            if response == "yes":
                cleanup_temp_files(temp_dir)
                print("\n✓ Starting face locking system...")
                print("=" * 60 + "\n")
                
                # Import and run the pipeline
                from src.run_pipeline import main as run_pipeline
                run_pipeline()
            else:
                print("\n✓ Enrollment complete!")
                print(f"Run 'python -m src.run_pipeline' to test face locking.")
        else:
            print("\n✗ Enrollment failed - no valid faces detected")
    
    except Exception as e:
        print(f"\n✗ Enrollment error: {e}")
    
    finally:
        # Cleanup
        cleanup_temp_files(temp_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        cleanup_temp_files("temp_enrollment")
        sys.exit(0)
