# FaceLocking — README

A short reference for how face locking works, which actions are detected, and how history files are named and stored.

---

## How face locking works 

- Pipeline: **camera → Haar face detection → FaceMesh (5pt) → alignment (112×112)** → **ArcFace ONNX** embedding → **cosine-distance match** against `data/db/face_db.npz`.
- Lock acquisition: when the matcher finds the configured identity (the `--lock-name`) with distance below the matching threshold (see `--dist-thr`), the system transitions from **IDLE** → **LOCKED** and begins tracking that face.
- Locked state: the face box is tracked by spatial continuity; the system checks for actions inside the locked ROI and logs them. If the locked face disappears for more than `--unlock-timeout` seconds, the lock is released (auto-unlock). Manual unlock can also be triggered by pressing `u`.

> Tip: recognition thresholds and timeouts can be tuned via command-line args to `face_lock.py` (e.g., `--dist-thr`, `--blink-thr`, `--smile-thr`, `--unlock-timeout`).

---

## Which actions are detected 

All actions are detected only for the currently **LOCKED** ROI.

- `lock_acquired` – when the configured identity is recognized and a lock is established (logs similarity and distance).
- `lock_released` – when the lock is released (reason: `manual` when pressing `u`, or `timeout` when the person disappears).
- `eye_blink` – detected via Eye Aspect Ratio (EAR) falling below `--blink-thr` (default 0.20). Description includes `EAR=<value>`.
- `smile` – detected via Mouth Aspect Ratio (MAR) exceeding `--smile-thr` (default 0.60). Description includes `MAR=<value>`.
- `moved_left` / `moved_right` – detected when the face center displaces horizontally beyond a fraction of the frame width (defaults: trigger 3% of frame width, reset 1.5%). Description includes `dx=<pixels>`.

Each detected action is immediately appended to the active history file with a short textual description.

---

## How history files are named and stored 

- Directory: `data/history/` (created automatically when a lock is acquired).
- File name format: `<person>_history_<YYYYMMDDHHMMSS><mmm>.txt` — for example: `joyeuse_history_20260131214841300.txt` (where the last three digits are milliseconds).
- File contents / per-line format: `YYYY-MM-DDTHH:MM:SS.mmmZ, <action>, <desc>` (UTC; `Z` denotes UTC)
  - Example line: `2026-01-31T21:48:41.300Z, eye_blink, EAR=0.12`
- A new file is created on the first lock acquisition for the session and events are appended throughout the session.

---

## Related files & locations 

- Enrollment outputs: `data/db/face_db.npz` (binary embeddings) and `data/db/face_db.json` (metadata). See `src/enroll.py`.
- Saved aligned crops: `data/enroll/<name>/*.jpg` — capture filenames use millisecond timestamps (e.g., `1650000000000.jpg`).
- History files: `data/history/` (see above).
- Model: `models/embedder_arcface.onnx` used to produce embeddings.

---
Made by Mugisha Pascal