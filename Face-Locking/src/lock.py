# src/lock.py
"""
Face Locking: lock onto one enrolled identity, track them, detect actions, record history.

Based on Term-02 Week-04 Learning Guide:
- Manual face selection (one identity to lock)
- When selected face is confidently recognized -> lock onto it
- Stable tracking (don't jump to other faces; tolerate brief recognition failures)
- Action detection: face moved left/right, eye blink, smile
- Record actions to <face>_history_<timestamp>.txt

Run:
python -m src.lock

Keys:
q : quit
r : reload DB
l : release lock (unlock)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .haar_5pt import align_face_5pt
from .embed import ArcFaceEmbedderONNX
from .recognize import (
    HaarFaceMesh5pt,
    FaceDet,
    MatchResult,
    FaceDBMatcher,
    load_db_npz,
    cosine_similarity,
)

# ----------------------------------
# Config
# ----------------------------------

DB_PATH = Path("data/db/face_db.npz")
HISTORY_DIR = Path("data/lock_history")
LOCK_SIM_THRESHOLD = 0.55   # min similarity to acquire lock
TRACK_SIM_THRESHOLD = 0.45  # min similarity to keep track (tolerate brief failures)
LOCK_RELEASE_SEC = 4.0     # release lock if face not seen for this long
MOVE_PX_THRESHOLD = 15     # min horizontal movement to record "moved left/right"
BLINK_RATIO_THRESHOLD = 0.75   # eye-nose dist ratio drop = blink
SMILE_RATIO_BASELINE = 1.0     # mouth_width/eye_dist above baseline * 1.2 = smile
ACTION_COOLDOWN_MOVE = 0.5     # sec between move actions
ACTION_COOLDOWN_BLINK = 0.4
ACTION_COOLDOWN_SMILE = 1.0

# ----------------------------------
# Lock state
# ----------------------------------

@dataclass
class LockState:
    identity: str = ""
    embedding: Optional[np.ndarray] = None
    lock_time: float = 0.0
    last_seen_time: float = 0.0
    history_path: Optional[Path] = None
    history_file: Optional[object] = None  # open file handle
    prev_center_x: float = 0.0
    prev_eye_nose_dist: float = 0.0
    prev_mouth_ratio: float = 0.0
    last_action_time: Dict[str, float] = field(default_factory=dict)

    def is_locked(self) -> bool:
        return bool(self.identity and self.embedding is not None)

    def release(self):
        if self.history_file is not None:
            try:
                self.history_file.close()
            except Exception:
                pass
            self.history_file = None
        self.identity = ""
        self.embedding = None
        self.history_path = None

    def write_action(self, action_type: str, description: str):
        if self.history_file is None:
            return
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        line = f"{ts}\t{action_type}\t{description}\n"
        try:
            self.history_file.write(line)
            self.history_file.flush()
        except Exception:
            pass

# ----------------------------------
# Action detection (simple heuristics from 5pt)
# ----------------------------------
# kps order: [left_eye, right_eye, nose, left_mouth, right_mouth] -> indices 0,1,2,3,4

def _face_center_x(kps: np.ndarray) -> float:
    return float(np.mean(kps[:, 0]))

def _eye_nose_dist(kps: np.ndarray) -> float:
    eye_y = (kps[0, 1] + kps[1, 1]) / 2.0
    return float(kps[2, 1] - eye_y)  # nose below eyes -> positive

def _mouth_width_ratio(kps: np.ndarray) -> float:
    mouth_w = float(kps[4, 0] - kps[3, 0])
    eye_dist = float(np.linalg.norm(kps[1] - kps[0]))
    if eye_dist < 1e-6:
        return 0.0
    return mouth_w / eye_dist

def _bbox_size(kps: np.ndarray) -> float:
    return float(np.max(kps[:, 0]) - np.min(kps[:, 0]) + np.max(kps[:, 1]) - np.min(kps[:, 1]))

def detect_actions(
    lock: LockState,
    f: FaceDet,
    frame_time: float,
) -> None:
    kps = f.kps
    center_x = _face_center_x(kps)
    eye_nose = _eye_nose_dist(kps)
    mouth_ratio = _mouth_width_ratio(kps)
    size = _bbox_size(kps)

    # Face moved left
    if lock.prev_center_x != 0:
        dx = center_x - lock.prev_center_x
        if dx < -MOVE_PX_THRESHOLD and (frame_time - lock.last_action_time.get("moved_left", 0)) >= ACTION_COOLDOWN_MOVE:
            lock.write_action("moved_left", f"face moved left (dx={dx:.0f})")
            lock.last_action_time["moved_left"] = frame_time
        elif dx > MOVE_PX_THRESHOLD and (frame_time - lock.last_action_time.get("moved_right", 0)) >= ACTION_COOLDOWN_MOVE:
            lock.write_action("moved_right", f"face moved right (dx={dx:.0f})")
            lock.last_action_time["moved_right"] = frame_time

    # Blink: brief drop in eye-nose distance (eyes "squeeze" toward nose)
    if lock.prev_eye_nose_dist > 1e-6 and size > 1e-6:
        ratio = eye_nose / (lock.prev_eye_nose_dist + 1e-6)
        if ratio < BLINK_RATIO_THRESHOLD and (frame_time - lock.last_action_time.get("blink", 0)) >= ACTION_COOLDOWN_BLINK:
            lock.write_action("blink", "eye blink detected")
            lock.last_action_time["blink"] = frame_time

    # Smile: mouth wider relative to eyes
    if lock.prev_mouth_ratio > 0.3 and mouth_ratio > lock.prev_mouth_ratio * 1.15:
        if (frame_time - lock.last_action_time.get("smile", 0)) >= ACTION_COOLDOWN_SMILE:
            lock.write_action("smile", "smile or laugh detected")
            lock.last_action_time["smile"] = frame_time

    lock.prev_center_x = center_x
    if lock.prev_eye_nose_dist <= 0:
        lock.prev_eye_nose_dist = eye_nose
    else:
        lock.prev_eye_nose_dist = 0.85 * lock.prev_eye_nose_dist + 0.15 * eye_nose
    if lock.prev_mouth_ratio <= 0:
        lock.prev_mouth_ratio = mouth_ratio
    else:
        lock.prev_mouth_ratio = 0.9 * lock.prev_mouth_ratio + 0.1 * mouth_ratio

# ----------------------------------
# Main
# ----------------------------------

def main():
    db_path = DB_PATH
    db = load_db_npz(db_path)
    if not db:
        print("No enrolled identities. Run: python -m src.enroll")
        return

    names = sorted(db.keys())
    print("Enrolled identities:", ", ".join(names))
    target = input("Enter identity to lock (e.g. Beni): ").strip()
    if not target or target not in db:
        print(f"Unknown identity '{target}'. Choose from: {', '.join(names)}")
        return

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    det = HaarFaceMesh5pt(min_size=(70, 70), debug=False)
    embedder = ArcFaceEmbedderONNX(input_size=(112, 112), debug=False)
    matcher = FaceDBMatcher(db=db, dist_thresh=0.34)

    lock = LockState()
    locked_embedding = db[target].reshape(-1).astype(np.float32)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")

    print("Face Locking. q=quit, r=reload DB, l=release lock")
    print(f"Waiting for '{target}' to appear...")

    t0 = time.time()
    fps_t0 = t0
    fps_n = 0
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        faces = det.detect(frame, max_faces=5)
        vis = frame.copy()
        h, w = vis.shape[:2]

        best_face: Optional[Tuple[FaceDet, float, MatchResult, np.ndarray]] = None  # (face, sim, mr, emb)

        for f in faces:
            aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
            emb = embedder.embed(aligned).embedding
            mr = matcher.match(emb)
            sim_to_target = float(np.dot(emb, locked_embedding))

            if not lock.is_locked():
                if mr.name == target and mr.accepted and sim_to_target >= LOCK_SIM_THRESHOLD:
                    if best_face is None or sim_to_target > best_face[1]:
                        best_face = (f, sim_to_target, mr, emb)
            else:
                if sim_to_target >= TRACK_SIM_THRESHOLD:
                    if best_face is None or sim_to_target > best_face[1]:
                        best_face = (f, sim_to_target, mr, emb)

        if lock.is_locked():
            if best_face is not None:
                f, sim, mr, _ = best_face
                lock.last_seen_time = now
                detect_actions(lock, f, now)
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 255), 3)
                cv2.putText(vis, f"LOCKED: {lock.identity}", (f.x1, max(0, f.y1 - 35)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(vis, f"sim={sim:.2f}", (f.x1, max(0, f.y1 - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                for (x, y) in f.kps.astype(int):
                    cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 255), -1)
            else:
                if now - lock.last_seen_time > LOCK_RELEASE_SEC:
                    lock.release()
                    print("Lock released (face not seen for too long).")
                else:
                    cv2.putText(vis, f"LOCKED: {lock.identity} (searching...)", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        else:
            if best_face is not None:
                f, sim, mr, emb = best_face
                lock.identity = target
                lock.embedding = emb.copy()
                lock.lock_time = now
                lock.last_seen_time = now
                lock.prev_center_x = _face_center_x(f.kps)
                lock.prev_eye_nose_dist = _eye_nose_dist(f.kps)
                lock.prev_mouth_ratio = _mouth_width_ratio(f.kps)
                ts = time.strftime("%Y%m%d%H%M%S", time.localtime())
                lock.history_path = HISTORY_DIR / f"{target.lower()}_history_{ts}.txt"
                lock.history_file = open(lock.history_path, "w", encoding="utf-8")
                lock.history_file.write("timestamp\taction_type\tdescription\n")
                lock.write_action("lock_acquired", f"locked onto {target}")
                print(f"Locked onto {target}. History: {lock.history_path}")
            else:
                for f in faces:
                    aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                    emb = embedder.embed(aligned).embedding
                    mr = matcher.match(emb)
                    label = mr.name if mr.name else "Unknown"
                    color = (0, 255, 0) if mr.accepted else (0, 0, 255)
                    cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
                    cv2.putText(vis, label, (f.x1, max(0, f.y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        fps_n += 1
        if now - fps_t0 >= 1.0:
            fps = fps_n / (now - fps_t0)
            fps_n = 0
            fps_t0 = now
        cv2.putText(vis, f"fps: {fps:.1f} | target: {target}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if lock.is_locked():
            cv2.putText(vis, "LOCKED - actions recorded", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        cv2.imshow("Face Lock", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            db = load_db_npz(db_path)
            matcher.reload_from(db_path)
            print(f"DB reloaded: {len(matcher._names)} identities")
        elif key == ord("l"):
            if lock.is_locked():
                lock.release()
                print("Lock released (manual).")

    lock.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
