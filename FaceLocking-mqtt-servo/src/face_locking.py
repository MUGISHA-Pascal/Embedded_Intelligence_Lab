"""
face_locking.py
"""
import time
import argparse
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from enum import Enum
import mediapipe as mp

# Import existing modules
# We need to ensure we can import from . if run as a module or direct
try:
    from .haar_5pt import Haar5ptDetector, align_face_5pt, _bbox_from_5pt, _clip_box_xyxy
    from .recognize import ArcFaceEmbedderONNX, FaceDBMatcher, load_db_npz
except ImportError:
    # If run directly: python src/face_locking.py
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from src.haar_5pt import Haar5ptDetector, align_face_5pt, _bbox_from_5pt, _clip_box_xyxy
    from src.recognize import ArcFaceEmbedderONNX, FaceDBMatcher, load_db_npz

# ---------------------------------------------------------
# Action Logic
# ---------------------------------------------------------
@dataclass
class FaceAction:
    timestamp: float
    action_type: str
    details: str

class FaceActionDetector:
    def __init__(self):
        # MediaPipe Landmark Indices
        # Left Eye (for EAR)
        self.P_LEFT_EYE = [33, 160, 158, 133, 153, 144] 
        # Right Eye (for EAR)
        self.P_RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        # Mouth (for SMILE/MAR) - 61=left corner, 291=right corner, 0=upper lip, 17=lower lip
        self.P_MOUTH = [61, 291, 0, 17]
        # Nose for pose
        self.P_NOSE_TIP = 1
        
        # Thresholds
        self.EAR_THRESH = 0.22  # Below this -> closed
        self.MAR_THRESH = 0.45  # Above this -> smile/open (simplified smile detection)
        # Smile can also be detected by mouth corner width relative to face width

        self.last_blink_time = 0.0
        self.blink_cooldown = 0.3
        
        self.last_nose_x = None

    def _ear(self, lm, idxs):
        # eye aspect ratio
        # vertical dists
        v1 = np.linalg.norm(lm[idxs[1]] - lm[idxs[5]])
        v2 = np.linalg.norm(lm[idxs[2]] - lm[idxs[4]])
        # horizontal
        h = np.linalg.norm(lm[idxs[0]] - lm[idxs[3]])
        return (v1 + v2) / (2.0 * h + 1e-6)

    def detect(self, mp_landmarks, frame_w, frame_h) -> List[Tuple[str, str]]:
        """
        Input: mp_landmarks (list of normalized x,y,z) from MediaPipe
        Returns: list of (ActionType, Description)
        """
        actions = []
        now = time.time()
        
        # Convert necessary landmarks to np arrays for calculation
        coords = np.array([[p.x, p.y] for p in mp_landmarks])
        
        # 1. Blink Detection
        left_ear = self._ear(coords, self.P_LEFT_EYE)
        right_ear = self._ear(coords, self.P_RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0
        
        if avg_ear < self.EAR_THRESH:
            if (now - self.last_blink_time) > self.blink_cooldown:
                actions.append(("BLINK", f"EAR={avg_ear:.2f}"))
                self.last_blink_time = now

        # 2. Smile Detection (Simple width checks or mouth alignment)
        # Check if mouth corners are 'wide' or mouth is open
        # Better simple smile: check if corners (61, 291) are higher than usual relative to upper lip (0)?
        # Or just use mouth width / jaw width ratio?
        # Let's use simple aspect ratio of mouth for "laugh/smile" (open mouth)
        # and maybe specific corner comparison for closed smile.
        # Simplest: Mouth width (61-291) vs Face Width (234-454 for cheeks)
        left_cheek = coords[234]
        right_cheek = coords[454]
        face_width = np.linalg.norm(right_cheek - left_cheek)
        
        mouth_l = coords[61]
        mouth_r = coords[291]
        mouth_width = np.linalg.norm(mouth_r - mouth_l)
        
        ratio = mouth_width / (face_width + 1e-6)
        if ratio > 0.45: # Tweak this
             actions.append(("SMILE", f"ratio={ratio:.2f}"))

        # 3. Head Movement (Left/Right)
        # Check nose x relative to frame center (0.5 in normalized coords)
        nose = coords[self.P_NOSE_TIP]
        if nose[0] < 0.50:
             actions.append(("MOVE_LEFT", f"nose_x={nose[0]:.2f}"))
        elif nose[0] > 0.60:
             actions.append(("MOVE_RIGHT", f"nose_x={nose[0]:.2f}"))
             
        return actions

# ---------------------------------------------------------
# Face Locking System
# ---------------------------------------------------------
class LockState(Enum):
    SEARCHING = 0
    LOCKED = 1
    # Could add LOST_RECOVERING state if we want hysteresis

class FaceLockSystem:
    def __init__(self, target_name: str, matcher: FaceDBMatcher, detector: Haar5ptDetector):
        self.target_name = target_name
        self.matcher = matcher
        self.det = detector
        self.state = LockState.SEARCHING
        
        self.action_det = FaceActionDetector()
        self.history: List[FaceAction] = []
        
        self.locked_frames = 0
        self.lost_frames = 0
        self.MAX_LOST_FRAMES = 10  # Tolerance before unlocking
        
        # We need to store the session file name
        ts = time.strftime("%Y%m%d%H%M%S")
        safe_name = "".join(c for c in target_name if c.isalnum())
        self.history_file = Path(f"{safe_name}_history_{ts}.txt")
        
        print(f"[FaceLock] Initialized. Target: {target_name}. Log: {self.history_file}")

    def log_action(self, atype: str, details: str):
        now = time.time()
        # Avoid spamming movement logs? Maybe only log on change?
        # For assignment, "record a history" is key.
        # We can implement a simple deduplication: don't log same action within 0.5s
        if self.history:
            last = self.history[-1]
            if last.action_type == atype and (now - last.timestamp) < 1.0:
                return

        act = FaceAction(timestamp=now, action_type=atype, details=details)
        self.history.append(act)
        
        line = f"{time.strftime('%H:%M:%S', time.localtime(now))} | {atype} | {details}\n"
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(line)
        print(f">> ACTION: {atype} ({details})")

    def process_frame(self, frame: np.ndarray, embedder: ArcFaceEmbedderONNX) -> Tuple[np.ndarray, Optional[object]]:
        vis = frame.copy()
        H, W = vis.shape[:2]

        faces, mp_res = self.det.detect_with_mesh(frame, max_faces=5)

        # 1. Process all faces to find matches
        # We want to identify everyone, but only "lock" on the target.
        target_face = None
        target_sim = 0.0

        for f in faces:
            # Default gray box for unknown/processing
            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (100, 100, 100), 1)

            aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
            emb = embedder.embed(aligned)
            mr = self.matcher.match(emb)

            if mr.accepted:
                # It is a known person
                is_target = (mr.name == self.target_name)
                
                if is_target:
                    # Keep track of the best target candidate
                    if mr.similarity > target_sim:
                        target_sim = mr.similarity
                        target_face = f
                else:
                    # Just label other known people immediately
                    cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (255, 200, 0), 2) # Cyan/Gold
                    cv2.putText(
                        vis,
                        mr.name,
                        (f.x1, f.y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 200, 0),
                        2
                    )

        # 2. State Machine Logic for Target
        
        # Handle state transitions based on whether target was found this frame
        if self.state == LockState.SEARCHING:
            cv2.putText(
                vis,
                f"SEARCHING: {self.target_name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2
            )

            if target_face is not None:
                self.state = LockState.LOCKED
                self.lost_frames = 0
                self.log_action("LOCK_ACQUIRED", f"sim={target_sim:.2f}")

        # Note: If we just transitioned to LOCKED, we fall through to this block if we use 'if' 
        # But usually we wait for next frame or handle it now. 
        # Let's handle it now (or next frame). The original code used elif, so it waited.
        # Let's use 'if' so we immediately start tracking if found.
        if self.state == LockState.LOCKED:
            if target_face is not None:
                self.lost_frames = 0
                f = target_face
                
                # Highlight Target
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 3)
                cv2.putText(
                    vis,
                    f"TARGET: {self.target_name}",
                    (f.x1, f.y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                
                # Action Detection on Target
                if mp_res and mp_res.multi_face_landmarks:
                    fw_x, fw_y = (f.x1 + f.x2) / 2, (f.y1 + f.y2) / 2
                    best_lm = None
                    min_dist = float("inf")

                    for lm_list in mp_res.multi_face_landmarks:
                        nose = lm_list.landmark[1]
                        nx, ny = nose.x * W, nose.y * H
                        dist = ((nx - fw_x) ** 2 + (ny - fw_y) ** 2) ** 0.5

                        if dist < min_dist:
                            min_dist = dist
                            best_lm = lm_list.landmark

                    if best_lm and min_dist < max(f.x2 - f.x1, f.y2 - f.y1):
                        actions = self.action_det.detect(best_lm, W, H)
                        for atype, desc in actions:
                            self.log_action(atype, desc)
                            cv2.putText(
                                vis,
                                f"ACT: {atype}",
                                (10, H - 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 255),
                                2
                            )
            else:
                # Target not found this frame
                self.lost_frames += 1
                cv2.putText(
                    vis,
                    f"LOCKED: {self.target_name}", # Header matches
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    vis,
                    f"LOST ({self.lost_frames}/{self.MAX_LOST_FRAMES})",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                if self.lost_frames > self.MAX_LOST_FRAMES:
                    self.state = LockState.SEARCHING
                    self.log_action("LOCK_LOST", "Target disappeared")

        return vis, target_face


def main():
    cfg = argparse.ArgumentParser()
    cfg.add_argument("--name", type=str, default="andrew", help="Target identity to lock onto")
    args = cfg.parse_args()
    
    # Init
    db_path = Path("data/db/face_db.npz")
    if not db_path.exists():
        print("No database found! Please run enroll.py first.")
        return

    det = Haar5ptDetector(min_size=(70, 70), debug=False)
    embedder = ArcFaceEmbedderONNX(input_size=(112, 112))
    
    db = load_db_npz(db_path)
    if args.name not in db:
        print(f"Warning: '{args.name}' not in database. Available: {list(db.keys())}")
        # Proceed anyway? No, impossible to lock.
        # But let's allow it to start scanning so user can see failures.
    
    matcher = FaceDBMatcher(db, dist_thresh=0.60)
    
    system = FaceLockSystem(args.name, matcher, det)
    
    cap = cv2.VideoCapture(1)
    print("Mask Locking System Started. Press 'q' to quit.")
    
    while True:
        ok, frame = cap.read()
        if not ok: break
        
        # Mirror the frame (user requested to "remove" the flip, implying they want the opposite of current)
        frame = cv2.flip(frame, 1)
        
        vis, _ = system.process_frame(frame, embedder)
        
        cv2.imshow("Face Locking", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
