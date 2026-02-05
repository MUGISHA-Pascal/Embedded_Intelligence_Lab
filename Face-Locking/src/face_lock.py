# src/face_lock.py
from __future__ import annotations
import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as e:  # pragma: no cover
    mp = None
    _MP_IMPORT_ERROR = e

from .recognize import (
    HaarFaceMesh5pt,
    ArcFaceEmbedderONNX,
    FaceDBMatcher,
    load_db_npz,
)
from .haar_5pt import align_face_5pt


@dataclass
class FaceBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def center(self) -> Tuple[float, float]:
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))

    def area(self) -> float:
        return float(max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1))


def _clip_xyxy(x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(W - 1, x1)))
    y1 = int(max(0, min(H - 1, y1)))
    x2 = int(max(0, min(W - 1, x2)))
    y2 = int(max(0, min(H - 1, y2)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _iou(a: FaceBox, b: FaceBox) -> float:
    ax1, ay1, ax2, ay2 = a.as_tuple()
    bx1, by1, bx2, by2 = b.as_tuple()
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    ua = a.area() + b.area() - inter
    return float(inter) / float(ua + 1e-6)


def _dist_c(a: FaceBox, b: FaceBox) -> float:
    ax, ay = a.center()
    bx, by = b.center()
    return float(np.hypot(ax - bx, ay - by))


class ActionLogger:
    def __init__(self, base_dir: Path, person: str):
        self.base_dir = base_dir
        self.person = person
        self.file_path = self._create_file()

    def _create_file(self) -> Path:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d%H%M%S")
        ms = int((time.time() - int(time.time())) * 1000)
        fname = f"{self.person.lower()}_history_{ts}{ms:03d}.txt"
        path = self.base_dir / fname
        path.touch(exist_ok=False)
        return path

    def log(self, action: str, desc: str = "") -> None:
        iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        ms = int((time.time() - int(time.time())) * 1000)
        line = f"{iso}.{ms:03d}Z, {action}, {desc}\n"
        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(line)


class FaceLocker:
    def __init__(
        self,
        lock_name: str,
        unlock_timeout_s: float = 2.5,
        dist_thresh: float = 0.62,
        blink_thr: float = 0.20,
        smile_thr: float = 0.60,
        debug: bool = False,
    ):
        if mp is None:
            raise RuntimeError(
                f"mediapipe import failed: {_MP_IMPORT_ERROR}\nInstall: pip install mediapipe==0.10.21"
            )
        self.lock_name = lock_name
        self.unlock_timeout_s = float(unlock_timeout_s)
        self.debug = bool(debug)

        # Components
        self.det = HaarFaceMesh5pt(min_size=(70, 70), debug=False)
        self.embedder = ArcFaceEmbedderONNX(
            model_path="models/embedder_arcface.onnx",
            input_size=(112, 112),
            debug=False,
        )
        db_path = Path("data/db/face_db.npz")
        self.db = load_db_npz(db_path)
        self.matcher = FaceDBMatcher(self.db, dist_thresh=dist_thresh)
        self.db_path = db_path

        # Landmarks model for action detection on locked ROI only
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # State
        self.state = "IDLE"  # or "LOCKED"
        self.last_box: Optional[FaceBox] = None
        self.last_seen_time: float = 0.0
        self.baseline_cx: Optional[float] = None
        self.logger: Optional[ActionLogger] = None

        # Movement detection params
        self.move_trigger_frac = 0.03  # 3% of frame width
        self.move_reset_frac = 0.015   # 1.5% to reset
        self.move_cooldown_s = 0.5
        self._last_move_t = 0.0

        # Blink/Smile params
        self.blink_thr = float(blink_thr)
        self.smile_thr = float(smile_thr)
        self.blink_cooldown_s = 0.6
        self.smile_cooldown_s = 0.8
        self._last_blink_t = 0.0
        self._last_smile_t = 0.0

    def _choose_locked_face(self, candidates: List[FaceBox], W: int) -> Optional[FaceBox]:
        if not candidates or self.last_box is None:
            return None
        # Prefer highest IoU; fallback to smallest center distance
        ious = [(_iou(self.last_box, c), i) for i, c in enumerate(candidates)]
        best_iou, idx_iou = max(ious, key=lambda t: t[0])
        if best_iou >= 0.10:
            return candidates[idx_iou]
        # Fallback by center distance threshold (10% of image diagonal)
        dists = [(_dist_c(self.last_box, c), i) for i, c in enumerate(candidates)]
        idx_min = min(dists, key=lambda t: t[0])[1]
        c = candidates[idx_min]
        diag = np.hypot(W, W)  # approximate with W (square-ish)
        if _dist_c(self.last_box, c) <= 0.12 * diag:
            return c
        return None

    @staticmethod
    def _to_box(x1: int, y1: int, x2: int, y2: int) -> FaceBox:
        return FaceBox(int(x1), int(y1), int(x2), int(y2))

    def _detect_actions(self, frame: np.ndarray, box: FaceBox) -> List[Tuple[str, str]]:
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = _clip_xyxy(box.x1, box.y1, box.x2, box.y2, W, H)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return []
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        if not res.multi_face_landmarks:
            return []
        lm = res.multi_face_landmarks[0].landmark
        rh, rw = roi.shape[:2]

        def P(idx: int) -> np.ndarray:
            p = lm[idx]
            return np.array([p.x * rw, p.y * rh], dtype=np.float32)

        # Blink via normalized vertical distances of eyelids
        # Left: upper 159, lower 145, corners 33 & 133
        L_up, L_lo = P(159), P(145)
        L_c1, L_c2 = P(33), P(133)
        L_w = float(np.linalg.norm(L_c2 - L_c1) + 1e-6)
        L_h = float(np.linalg.norm(L_up - L_lo))
        # Right: upper 386, lower 374, corners 263 & 362
        R_up, R_lo = P(386), P(374)
        R_c1, R_c2 = P(263), P(362)
        R_w = float(np.linalg.norm(R_c2 - R_c1) + 1e-6)
        R_h = float(np.linalg.norm(R_up - R_lo))
        ear = 0.5 * ((L_h / L_w) + (R_h / R_w))

        # Smile via MAR: inner lip 13/14 vs corners 61/291
        U, D = P(13), P(14)
        ML, MR = P(61), P(291)
        mar = float(np.linalg.norm(U - D) / (np.linalg.norm(MR - ML) + 1e-6))

        now = time.time()
        actions: List[Tuple[str, str]] = []
        if ear < self.blink_thr and (now - self._last_blink_t) >= self.blink_cooldown_s:
            actions.append(("eye_blink", f"EAR={ear:.2f}"))
            self._last_blink_t = now
        if mar > self.smile_thr and (now - self._last_smile_t) >= self.smile_cooldown_s:
            actions.append(("smile", f"MAR={mar:.2f}"))
            self._last_smile_t = now
        return actions

    def _movement_action(self, box: FaceBox, W: int) -> Optional[Tuple[str, str]]:
        now = time.time()
        if (now - self._last_move_t) < self.move_cooldown_s:
            return None
        cx, _ = box.center()
        if self.baseline_cx is None:
            self.baseline_cx = float(cx)
            return None
        dx = float(cx - self.baseline_cx)
        trig = self.move_trigger_frac * W
        reset = self.move_reset_frac * W
        if dx >= trig:
            self.baseline_cx = float(cx)  # hysteresis: reset to current after event
            self._last_move_t = now
            return ("moved_right", f"dx=+{dx:.0f}px")
        elif dx <= -trig:
            self.baseline_cx = float(cx)
            self._last_move_t = now
            return ("moved_left", f"dx={dx:.0f}px")
        else:
            # slow drift reset to keep baseline in sync
            if abs(dx) <= reset:
                self.baseline_cx = 0.5 * (self.baseline_cx + cx)
            return None

    def run(self, default_window: str = "face_lock") -> None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Camera not available")
        print("Face Locking. q=quit, u=unlock, r=reload DB, +/- threshold")
        t0 = time.time()
        frames = 0
        fps: Optional[float] = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            faces_det = self.det.detect(frame, max_faces=5)
            H, W = frame.shape[:2]
            vis = frame.copy()

            # Convert detections to boxes list for tracking selection
            boxes = [self._to_box(f.x1, f.y1, f.x2, f.y2) for f in faces_det]

            locked_idx: Optional[int] = None
            chosen_box: Optional[FaceBox] = None
            chosen_name: Optional[str] = None
            chosen_accepted = False
            chosen_dist = 1.0
            chosen_sim = 0.0

            # Recognition for all faces, but we will pick one to lock/track
            for i, f in enumerate(faces_det):
                aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                emb = self.embedder.embed(aligned)
                mr = self.matcher.match(emb)

                # IDLE -> acquire lock when selected identity appears
                if self.state == "IDLE" and mr.accepted and mr.name == self.lock_name:
                    self.state = "LOCKED"
                    chosen_box = boxes[i]
                    chosen_name = mr.name
                    chosen_accepted = True
                    chosen_dist = mr.distance
                    chosen_sim = mr.similarity
                    self.last_box = chosen_box
                    self.last_seen_time = time.time()
                    self.baseline_cx = chosen_box.center()[0]
                    if self.logger is None:
                        self.logger = ActionLogger(Path("data/history"), self.lock_name)
                    self.logger.log("lock_acquired", f"sim={mr.similarity:.3f} dist={mr.distance:.3f}")

                # If already locked, try to update chosen box to the best matching spatial face
                if self.state == "LOCKED":
                    # select candidate by spatial continuity, but prefer if this face is the lock_name when recognized
                    if mr.accepted and mr.name == self.lock_name:
                        # Strong candidate: recognized as target
                        cand = boxes[i]
                        chosen_box = cand
                        chosen_name = mr.name
                        chosen_accepted = True
                        chosen_dist = mr.distance
                        chosen_sim = mr.similarity
                        locked_idx = i

            # If locked but not recognized in this frame or we didn't see it during recognition loop,
            # choose by spatial continuity among boxes
            if self.state == "LOCKED" and (chosen_box is None) and boxes:
                cb = self._choose_locked_face(boxes, W)
                if cb is not None:
                    chosen_box = cb

            # State maintenance
            now = time.time()
            if self.state == "LOCKED":
                if chosen_box is not None:
                    self.last_box = chosen_box
                    self.last_seen_time = now
                else:
                    # no plausible box this frame
                    if (now - self.last_seen_time) >= self.unlock_timeout_s:
                        # auto-unlock
                        if self.logger is not None:
                            self.logger.log("lock_released", "timeout")
                        self.state = "IDLE"
                        self.last_box = None
                        self.baseline_cx = None

            # Draw and actions
            for i, f in enumerate(faces_det):
                c = (0, 0, 255)
                thick = 2
                if self.state == "LOCKED" and self.last_box is not None:
                    lb = self.last_box
                    if f.x1 == lb.x1 and f.y1 == lb.y1 and f.x2 == lb.x2 and f.y2 == lb.y2:
                        c = (0, 255, 0)
                        thick = 3
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), c, thick)

            header = f"IDs={len(self.matcher._names)} thr={self.matcher.dist_thresh:.2f}"
            if fps is not None:
                header += f" fps={fps:.1f}"
            header += f" Locked={'None' if self.state=='IDLE' else self.lock_name}"
            cv2.putText(vis, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            # Action detection & UI label for locked box
            if self.state == "LOCKED" and self.last_box is not None:
                lb = self.last_box
                cv2.putText(
                    vis,
                    f"LOCKED: {self.lock_name}",
                    (lb.x1, max(0, lb.y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                # Movement
                mv = self._movement_action(lb, W)
                if mv and self.logger is not None:
                    self.logger.log(mv[0], mv[1])
                # Blink & Smile
                acts = self._detect_actions(frame, lb)
                for act, desc in acts:
                    if self.logger is not None:
                        self.logger.log(act, desc)

            # FPS bookkeeping
            frames += 1
            dt = time.time() - t0
            if dt >= 1.0:
                fps = frames / dt
                frames = 0
                t0 = time.time()

            cv2.imshow(default_window, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("u"):
                if self.state == "LOCKED":
                    if self.logger is not None:
                        self.logger.log("lock_released", "manual")
                self.state = "IDLE"
                self.last_box = None
                self.baseline_cx = None
            elif key == ord("r"):
                self.matcher.reload_from(self.db_path)
                print(f"[face_lock] DB reloaded: {len(self.matcher._names)} identities")
            elif key in (ord("+"), ord("=")):
                self.matcher.dist_thresh = float(min(1.20, self.matcher.dist_thresh + 0.01))
                print(
                    f"[face_lock] thr(dist)={self.matcher.dist_thresh:.2f} (sim~{1.0-self.matcher.dist_thresh:.2f})"
                )
            elif key == ord("-"):
                self.matcher.dist_thresh = float(max(0.05, self.matcher.dist_thresh - 0.01))
                print(
                    f"[face_lock] thr(dist)={self.matcher.dist_thresh:.2f} (sim~{1.0-self.matcher.dist_thresh:.2f})"
                )

        cap.release()
        cv2.destroyAllWindows()


def _select_lock_name_from_db(default: str, db: dict) -> str:
    names = sorted(list(db.keys()))
    if not names:
        raise RuntimeError("No identities in DB. Please enroll first.")
    if default and default in db:
        return default
    print("Available identities:")
    for i, n in enumerate(names):
        print(f"  [{i+1}] {n}")
    try:
        s = input(f"Select identity to lock (default '{default or names[0]}'): ").strip()
        if not s:
            return default or names[0]
        if s.isdigit():
            j = int(s) - 1
            if 0 <= j < len(names):
                return names[j]
        if s in db:
            return s
    except KeyboardInterrupt:
        pass
    return default or names[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Face Locking with action history")
    p.add_argument("--lock-name", type=str, default="Joyeuse", help="Identity to lock")
    p.add_argument("--unlock-timeout", type=float, default=2.5, help="Seconds to auto-unlock after disappearance")
    p.add_argument("--blink-thr", type=float, default=0.20, help="EAR threshold for blink detection")
    p.add_argument("--smile-thr", type=float, default=0.60, help="MAR threshold for smile detection")
    p.add_argument("--dist-thr", type=float, default=0.62, help="Matcher distance threshold")
    return p.parse_args()


def main():
    args = parse_args()
    db = load_db_npz(Path("data/db/face_db.npz"))
    lock_name = _select_lock_name_from_db(args.lock_name, db)
    fl = FaceLocker(
        lock_name=lock_name,
        unlock_timeout_s=args.unlock_timeout,
        dist_thresh=args.dist_thr,
        blink_thr=args.blink_thr,
        smile_thr=args.smile_thr,
        debug=False,
    )
    fl.run()


if __name__ == "__main__":
    main()
