# src/align.py
"""
Alignment demo using your WORKING pipeline:
- Haar face detection (fast)
- MediaPipe FaceMesh -> 5 keypoints (stable)
- ArcFace-style 5pt alignment -> 112x112 (or any size you set)

This avoids the bug in haar 5pt.py where the aligned window was shown only after the loop and using stale variables.

Run:
python -m src.align

Keys:
q quit
s save current aligned face to data/debug_aligned/<timestamp>.jpg
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# Import from your existing script
from .haar_5pt import Haar5ptDetector, align_face_5pt

def _put_text(img, text: str, xy=(10, 30), scale=0.8, thickness=2):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def _safe_imshow(win: str, img: np.ndarray):
    if img is None:
        return
    cv2.imshow(win, img)

def main(
    cam_index: int = 0,
    out_size: Tuple[int, int] = (112, 112),
    mirror: bool = True,
):
    cap = cv2.VideoCapture(cam_index)

    det = Haar5ptDetector(
        min_size=(70, 70),
        smooth_alpha=0.80,
        debug=True,
    )

    out_w, out_h = int(out_size[0]), int(out_size[1])
    blank = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # Where to save aligned snapshots
    save_dir = Path("data/debug_aligned")
    save_dir.mkdir(parents=True, exist_ok=True)

    last_aligned = blank.copy()
    fps_t0 = time.time()
    fps_n = 0
    fps = 0.0

    print("align running. Press 'q' to quit, 's' to save aligned face.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        faces = det.detect(frame, max_faces=1)

        vis = frame.copy()
        aligned = None

        if faces:
            f = faces[0]

            # Draw box + 5 pts
            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
            for (x, y) in f.kps.astype(int):
                cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)

            # Align (this is the whole point)
            aligned, _M = align_face_5pt(frame, f.kps, out_size=out_size)

            # Keep last good aligned (so window doesn't go black on brief misses)
            if aligned is not None and aligned.size:
                last_aligned = aligned

            _put_text(vis, "OK (Haar + FaceMesh 5pt)", (10, 30), 0.75, 2)
        else:
            _put_text(vis, "no face", (10, 30), 0.9, 2)

        # FPS
        fps_n += 1
        dt = time.time() - fps_t0
        if dt >= 1.0:
            fps = fps_n / dt
            fps_n = 0
            fps_t0 = time.time()
        _put_text(vis, f"FPS: {fps:.1f}", (10, 60), 0.75, 2)
        _put_text(vis, f"warp: 5pt -> {out_w}:{out_h}", (10, 90), 0.75, 2)

        _safe_imshow("align - camera", vis)
        _safe_imshow("align - aligned", last_aligned)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            ts = int(time.time() * 1000)
            out_path = save_dir / f"{ts}.jpg"
            cv2.imwrite(str(out_path), last_aligned)
            print(f"aligned saved: {out_path}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()