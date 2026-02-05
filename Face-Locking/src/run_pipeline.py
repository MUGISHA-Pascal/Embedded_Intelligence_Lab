import os
import time
from typing import List, Tuple

import cv2

from .action_detector import ActionDetector
from .action_logger import ActionLogger
from .camera import camera_stream
from .embed import ArcFaceEmbedder
from .face_locker import FaceLocker
from .recognize import load_identity_database, recognize_frame


def draw_label(frame: cv2.Mat, box: Tuple[int, int, int, int], text: str, color: Tuple[int, int, int]) -> None:
    """Draw a bounding box and label on the frame."""
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(frame, (x1, y1 - 24), (x2, y1), color, -1)
    cv2.putText(
        frame,
        text,
        (x1 + 4, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_lock_indicator(frame: cv2.Mat, identity: str, duration: float, actions: List[str], action_counts: dict) -> None:
    """Draw lock status and recent actions on the frame."""
    height, width = frame.shape[:2]
    
    # Draw lock status banner
    banner_height = 140
    cv2.rectangle(frame, (10, 10), (width - 10, banner_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (width - 10, banner_height), (0, 255, 0), 3)
    
    # Lock status text with icon
    status_text = f"🔒 LOCKED: {identity}"
    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Duration text
    duration_text = f"Duration: {duration:.1f}s"
    cv2.putText(frame, duration_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Recent actions with visual feedback
    if actions:
        actions_text = f"Detected: {', '.join(actions).upper()}"
        cv2.putText(frame, actions_text, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Action counts
    counts_text = f"Blinks: {action_counts.get('blink', 0)} | Smiles: {action_counts.get('smile', 0)} | Moves: {action_counts.get('move', 0)}"
    cv2.putText(frame, counts_text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def draw_instructions(frame: cv2.Mat, num_faces: int) -> None:
    """Draw usage instructions on the frame."""
    height, width = frame.shape[:2]
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, height - 110), (width - 10, height - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    instructions = [
        "Press 'q' to quit | Try: Blink, Smile, Move head left/right",
        f"Faces detected: {num_faces} | Lock activates on target face",
    ]
    
    y_offset = height - 90
    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (20, y_offset + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def select_target_identity(database: dict) -> str:
    """
    Allow user to select which identity to lock onto.
    
    Args:
        database: Dictionary of enrolled identities
        
    Returns:
        Selected identity name
    """
    if not database:
        print("No enrolled identities found!")
        print("Please enroll at least one identity first.")
        exit(1)
    
    print("\n" + "=" * 60)
    print("FACE LOCKING SYSTEM")
    print("=" * 60)
    print("\nEnrolled identities:")
    identities = list(database.keys())
    for i, name in enumerate(identities, 1):
        print(f"  {i}. {name}")
    
    while True:
        try:
            choice = input(f"\nSelect identity to lock (1-{len(identities)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(identities):
                selected = identities[idx]
                print(f"\n✓ Target identity set to: {selected}")
                print("Starting camera...\n")
                return selected
            else:
                print(f"Please enter a number between 1 and {len(identities)}")
        except (ValueError, KeyboardInterrupt):
            print("\nExiting...")
            exit(0)


def main() -> None:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(base_dir, "models", "arcface.onnx")
    identities_dir = os.path.join(base_dir, "data", "identities")
    history_dir = os.path.join(base_dir, "action_history")

    embedder = ArcFaceEmbedder(model_path)
    database = load_identity_database(identities_dir)
    
    # Select target identity
    target_identity = select_target_identity(database)
    
    # Initialize face locking components
    locker = FaceLocker(target_identity=target_identity, lock_threshold=0.45, unlock_timeout=2.0)
    action_detector = ActionDetector()
    action_logger = ActionLogger(output_dir=history_dir)
    
    logging_started = False
    recent_actions: List[str] = []
    action_display_frames = 0
    action_counts = {"blink": 0, "smile": 0, "move": 0}

    print("\n" + "=" * 60)
    print("FACE LOCKING SYSTEM ACTIVE")
    print("=" * 60)
    print(f"Target: {target_identity}")
    print("Waiting for target face to appear...")
    print("\nActions to try:")
    print("  • Blink your eyes")
    print("  • Smile")
    print("  • Move your head left/right")
    print("\nPress 'q' to quit\n")

    for frame in camera_stream():
        current_time = time.time()
        
        # Recognize all faces in frame
        results = recognize_frame(frame, embedder, database)
        
        # Update face lock
        locked_box = locker.update(results, current_time)
        
        # Draw all detected faces (non-locked faces in cyan)
        for box, name, score in results:
            if box != locked_box:
                color = (255, 200, 0) if name != "Unknown" else (100, 100, 100)
                draw_label(frame, box, f"{name} ({score:.2f})", color)
        
        # Handle locked face
        if locked_box is not None:
            # Start logging if not already started
            if not logging_started:
                log_file = action_logger.start_logging(target_identity)
                action_logger.log_lock_event("LOCK ESTABLISHED", f"Target: {target_identity}")
                print(f"\n✓ Lock established on {target_identity}")
                print(f"✓ Logging to: {log_file}")
                print("✓ Action detection active\n")
                logging_started = True
            
            # Detect actions
            detected_actions = action_detector.detect_actions(frame, locked_box)
            
            # Log actions and update counts
            if detected_actions:
                action_logger.log_multiple_actions(detected_actions)
                recent_actions = detected_actions
                action_display_frames = 30  # Display for 30 frames (~1 second)
                
                # Update counts
                for action in detected_actions:
                    if action == "blink":
                        action_counts["blink"] += 1
                        print(f"  👁️  Blink detected! (Total: {action_counts['blink']})")
                    elif action == "smile":
                        action_counts["smile"] += 1
                        print(f"  😊 Smile detected! (Total: {action_counts['smile']})")
                    elif action in ["moved_left", "moved_right"]:
                        action_counts["move"] += 1
                        direction = "left" if action == "moved_left" else "right"
                        print(f"  ← → Head moved {direction}! (Total moves: {action_counts['move']})")
            
            # Draw locked face with special color (red)
            lock_duration = locker.get_lock_duration(current_time)
            draw_label(frame, locked_box, f"🔒 {target_identity}", (0, 0, 255))
            
            # Draw lock indicator
            display_actions = recent_actions if action_display_frames > 0 else []
            draw_lock_indicator(frame, target_identity, lock_duration, display_actions, action_counts)
            action_display_frames = max(0, action_display_frames - 1)
        
        else:
            # Lock released
            if logging_started:
                action_logger.log_lock_event("LOCK RELEASED", f"Target lost for {locker.unlock_timeout}s")
                action_logger.stop_logging()
                print(f"\n✗ Lock released on {target_identity}")
                print(f"✓ Session summary:")
                print(f"  - Blinks: {action_counts['blink']}")
                print(f"  - Smiles: {action_counts['smile']}")
                print(f"  - Head movements: {action_counts['move']}")
                print(f"✓ Log saved to: {action_logger.get_current_file()}\n")
                print("Waiting for target face to reappear...")
                logging_started = False
                action_detector.reset()
                recent_actions = []
                action_counts = {"blink": 0, "smile": 0, "move": 0}
        
        # Draw instructions
        draw_instructions(frame, len(results))
        
        cv2.imshow("Face Locking System", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    if logging_started:
        action_logger.stop_logging()
        print(f"\n✓ Final log saved to: {action_logger.get_current_file()}")
    
    cv2.destroyAllWindows()
    print("\n" + "=" * 60)
    print("Face Locking System terminated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
