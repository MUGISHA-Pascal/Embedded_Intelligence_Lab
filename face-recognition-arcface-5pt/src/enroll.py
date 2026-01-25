# src/enroll.py
"""
enroll.py
Enrollment tool using your working pipeline:

camera -> Haar detection -> FaceMesh 5pt -> align_face_5pt (112x112) -> ArcFace embedding
Stores template per identity (mean embedding, L2-normalized).

Re-enroll behavior:
- If data/enroll/<name> already contains aligned crops, those are loaded,
  embedded again, and INCLUDED in the template. New captures are appended.

Outputs:
- data/db/face_db.npz   (name -> embedding vector)
- data/db/face_db.json  (metadata)
Optional:
- data/enroll/<name>/*.jpg aligned face crops

Controls:
- SPACE: capture one sample (if face found)
- a: auto-capture toggle (captures periodically)
- s: save enrollment (after enough total samples)
- r: reset NEW samples (keeps existing crops on disk)
- q: quit
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .haar_5pt import Haar5ptDetector, align_face_5pt
from .embed import ArcFaceEmbedderONNX


# -------------------------
# Config
# -------------------------

@dataclass
class EnrollConfig:
    out_db_npz: Path = Path("data/db/face_db.npz")
    out_db_json: Path = Path("data/db/face_db.json")

    save_crops: bool = True
    crops_dir: Path = Path("data/enroll")

    samples_needed: int = 15
    auto_capture_every_s: float = 0.25
    max_existing_crops: int = 300

    # UI
    window_main: str = "enroll"
    window_aligned: str = "aligned_112"


# -------------------------
# DB helpers
# -------------------------

def ensure_dirs(cfg: EnrollConfig) -> None:
    cfg.out_db_npz.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_db_json.parent.mkdir(parents=True, exist_ok=True)
    if cfg.save_crops:
        cfg.crops_dir.mkdir(parents=True, exist_ok=True)


def load_db(cfg: EnrollConfig) -> Dict[str, np.ndarray]:
    if cfg.out_db_npz.exists():
        data = np.load(cfg.out_db_npz, allow_pickle=True)
        return {k: data[k].astype(np.float32) for k in data.files}
    return {}


def save_db(cfg: EnrollConfig, db: Dict[str, np.ndarray], meta: dict) -> None:
    ensure_dirs(cfg)
    np.savez(cfg.out_db_npz, **{k: v.astype(np.float32) for k, v in db.items()})
    cfg.out_db_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def mean_embedding(embeddings: List[np.ndarray]) -> np.ndarray:
    """Mean + L2 normalize."""
    E = np.stack([e.reshape(-1) for e in embeddings], axis=0).astype(np.float32)
    m = E.mean(axis=0)
    m = m / (np.linalg.norm(m) + 1e-12)
    return m.astype(np.float32)


# -------------------------
# Crops loader
# -------------------------

def _list_existing_crops(person_dir: Path, max_count: int) -> List[Path]:
    if not person_dir.exists():
        return []
    files = sorted([p for p in person_dir.glob("*.jpg") if p.is_file()])
    if len(files) > max_count:
        files = files[-max_count:]
    return files


def load_existing_samples_from_crops(
    cfg: EnrollConfig,
    emb: ArcFaceEmbedderONNX,
    person_dir: Path,
) -> List[np.ndarray]:
    """
    Reads aligned crops from disk and re-embeds them.
    """
    if not cfg.save_crops:
        return []

    crops = _list_existing_crops(person_dir, cfg.max_existing_crops)
    base: List[np.ndarray] = []

    for p in crops:
        img = cv2.imread(str(p))
        if img is None:
            continue
        try:
            r = emb.embed(img)
            base.append(r.embedding)
        except Exception:
            continue

    return base


# -------------------------
# UI helpers
# -------------------------

def draw_status(
    frame: np.ndarray,
    name: str,
    base_count: int,
    new_count: int,
    needed: int,
    auto: bool,
    msg: str = "",
) -> None:
    total = base_count + new_count
    lines = [
        f"ENROLL: {name}",
        f"Existing: {base_count} | New: {new_count} | Total: {total} / {needed}",
        f"Auto: {'ON' if auto else 'OFF'}  (toggle: a)",
        "SPACE=capture | s=save | r=reset NEW | q=quit",
    ]
    if msg:
        lines.insert(0, msg)

    # draw with black shadow for readability
    y = 30
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        y += 26


# -------------------------
# Main
# -------------------------

def main():
    cfg = EnrollConfig()
    ensure_dirs(cfg)

    name = input("Enter person name to enroll (e.g., Alice): ").strip()
    if not name:
        print("No name provided. Exiting.")
        return

    # Pipeline (your working practical stack)
    det = Haar5ptDetector(min_size=(70, 70), smooth_alpha=0.80, debug=False)
    emb = ArcFaceEmbedderONNX(model_path="models/embedder_arcface.onnx", input_size=(112, 112),
                              debug=False)

    db = load_db(cfg)

    person_dir = cfg.crops_dir / name
    if cfg.save_crops:
        person_dir.mkdir(parents=True, exist_ok=True)

    base_samples: List[np.ndarray] = load_existing_samples_from_crops(cfg, emb, person_dir)
    new_samples: List[np.ndarray] = []

    status_msg = ""
    if base_samples:
        status_msg = f"Loaded {len(base_samples)} existing samples from disk."

    auto = False
    last_auto = 0.0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera.")

    cv2.namedWindow(cfg.window_main, cv2.WINDOW_NORMAL)
    cv2.namedWindow(cfg.window_aligned, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cfg.window_aligned, 240, 240)

    print("\nEnrollment started.")
    if base_samples:
        print(f"Re-enroll mode: found {len(base_samples)} existing samples in {person_dir}/")
    print("Tip: stable lighting, move slightly left/right, different expressions.")
    print("Controls: SPACE=capture, a=auto, s=save, r=reset NEW, q=quit\n")

    t0 = time.time()
    frames = 0
    fps: Optional[float] = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            vis = frame.copy()
            faces = det.detect(frame, max_faces=1)

            aligned: Optional[np.ndarray] = None

            if faces:
                f = faces[0]

                # draw bbox + kps
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
                for (x, y) in f.kps.astype(int):
                    cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)

                aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                cv2.imshow(cfg.window_aligned, aligned)
            else:
                cv2.imshow(cfg.window_aligned, np.zeros((112, 112, 3), dtype=np.uint8))

            # auto capture
            now = time.time()
            if auto and aligned is not None and (now - last_auto) >= cfg.auto_capture_every_s:
                r = emb.embed(aligned)
                new_samples.append(r.embedding)
                last_auto = now
                status_msg = f"Auto captured NEW ({len(new_samples)})"

                if cfg.save_crops:
                    fn = person_dir / f"{int(now * 1000)}.jpg"
                    cv2.imwrite(str(fn), aligned)

            # FPS
            frames += 1
            dt = time.time() - t0
            if dt >= 1.0:
                fps = frames / dt
                frames = 0
                t0 = time.time()

            if fps is not None:
                cv2.putText(vis, f"FPS: {fps:.1f}", (10, vis.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            draw_status(
                vis,
                name=name,
                base_count=len(base_samples),
                new_count=len(new_samples),
                needed=cfg.samples_needed,
                auto=auto,
                msg=status_msg,
            )

            cv2.imshow(cfg.window_main, vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("a"):
                auto = not auto
                status_msg = f"Auto mode {'ON' if auto else 'OFF'}"

            if key == ord("r"):
                new_samples.clear()
                status_msg = "NEW samples reset (existing kept)."

            if key == ord(" "):  # SPACE
                if aligned is None:
                    status_msg = "No face detected. Not captured."
                else:
                    r = emb.embed(aligned)
                    new_samples.append(r.embedding)
                    status_msg = f"Captured NEW ({len(new_samples)})"

                    if cfg.save_crops:
                        fn = person_dir / f"{int(time.time() * 1000)}.jpg"
                        cv2.imwrite(str(fn), aligned)

            if key == ord("s"):
                total = len(base_samples) + len(new_samples)
                if total < max(3, cfg.samples_needed // 2):
                    status_msg = f"Not enough total samples to save (have {total})."
                    continue

                all_samples = base_samples + new_samples
                template = mean_embedding(all_samples)
                db[name] = template

                meta = {
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "embedding_dim": int(template.size),
                    "names": sorted(db.keys()),
                    "samples_existing_used": int(len(base_samples)),
                    "samples_new_used": int(len(new_samples)),
                    "samples_total_used": int(len(all_samples)),
                    "note": "Embeddings are L2-normalized vectors. Matching uses cosine similarity.",
                }
                save_db(cfg, db, meta)

                status_msg = f"Saved '{name}' to DB. Total identities: {len(db)}"
                print(status_msg)

                # reload base from disk so UI matches reality
                base_samples = load_existing_samples_from_crops(cfg, emb, person_dir)
                new_samples.clear()

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()