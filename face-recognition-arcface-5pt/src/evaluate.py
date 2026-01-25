# src/evaluate.py
"""
evaluate.py
Threshold tuning / evaluation using enrollment crops (aligned 112x112).
Assumptions:
- Enrollment crops exist under: data/enroll/<name>/*.jpg
- Crops are aligned (112x112) already (as saved by enroll.py / haar_5pt pipeline)
- Uses ArcFaceEmbedderONNX from embed.py (your working embedder)
Outputs:
- Prints summary stats for genuine/impostor cosine distances
- Suggests a threshold based on a target FAR
Run:
 python -m src.evaluate
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .embed import ArcFaceEmbedderONNX


# -------------------------
# Config
# -------------------------

@dataclass
class EvalConfig:
    enroll_dir: Path = Path("data/enroll")
    min_imgs_per_person: int = 5
    max_imgs_per_person: int = 80      # cap for speed
    target_far: float = 0.01           # 1% FAR target
    thresholds: Tuple[float, float, float] = (0.10, 1.20, 0.01)  # start, end, step

    # Optional sanity constraints
    require_size: Tuple[int, int] = (112, 112)


# -------------------------
# Math
# -------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    # embeddings are already L2-normalized in embed, so dot is cosine
    return float(np.dot(a, b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # distance = 1 - cosine similarity
    return 1.0 - cosine_similarity(a, b)


# -------------------------
# IO
# -------------------------

def list_people(cfg: EvalConfig) -> List[Path]:
    if not cfg.enroll_dir.exists():
        raise FileNotFoundError(f"Enroll dir not found: {cfg.enroll_dir}. Run enroll.py first.")
    return sorted([p for p in cfg.enroll_dir.iterdir() if p.is_dir()])


def _is_aligned_crop(img: np.ndarray, req: Tuple[int, int]) -> bool:
    h, w = img.shape[:2]
    return (w, h) == (int(req[0]), int(req[1]))


def load_embeddings_for_person(
    embedder: ArcFaceEmbedderONNX,
    person_dir: Path,
    cfg: EvalConfig,
) -> List[np.ndarray]:
    imgs = sorted(list(person_dir.glob("*.jpg")))[: cfg.max_imgs_per_person]
    embs: List[np.ndarray] = []

    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # If someone accidentally saved non-aligned crops, skip them (keeps eval clean)
        if cfg.require_size is not None and not _is_aligned_crop(img, cfg.require_size):
            continue

        res = embedder.embed(img)
        embs.append(res.embedding)

    return embs


# -------------------------
# Eval
# -------------------------

def pairwise_distances(embs_a: List[np.ndarray], embs_b: List[np.ndarray], same: bool) -> List[float]:
    dists: List[float] = []
    if same:
        for i in range(len(embs_a)):
            for j in range(i + 1, len(embs_a)):
                dists.append(cosine_distance(embs_a[i], embs_a[j]))
    else:
        for ea in embs_a:
            for eb in embs_b:
                dists.append(cosine_distance(ea, eb))
    return dists


def sweep_thresholds(genuine: np.ndarray, impostor: np.ndarray, cfg: EvalConfig):
    t0, t1, step = cfg.thresholds
    thresholds = np.arange(t0, t1 + 1e-9, step, dtype=np.float32)

    # FAR: impostor accepted => dist <= thr |  FRR: genuine rejected  => dist > thr
    results = []
    for thr in thresholds:
        far = float(np.mean(impostor <= thr)) if impostor.size else 0.0
        frr = float(np.mean(genuine > thr)) if genuine.size else 0.0
        results.append((float(thr), far, frr))
    return results


def describe(arr: np.ndarray) -> str:
    if arr.size == 0:
        return "n=0"
    return (
        f"n={arr.size}  mean={arr.mean():.3f}  std={arr.std():.3f}  "
        f"p05={np.percentile(arr, 5):.3f}  p50={np.percentile(arr, 50):.3f}  p95={np.percentile(arr,
95):.3f}"
    )


def main():
    cfg = EvalConfig()

    embedder = ArcFaceEmbedderONNX(
        model_path="models/embedder_arcface.onnx",
        input_size=(112, 112),
        debug=False,
    )

    people_dirs = list_people(cfg)
    if len(people_dirs) < 1:
        print("No enrolled people found.")
        return

    # Load embeddings per person
    per_person: Dict[str, List[np.ndarray]] = {}
    for pdir in people_dirs:
        name = pdir.name
        embs = load_embeddings_for_person(embedder, pdir, cfg)
        if len(embs) >= cfg.min_imgs_per_person:
            per_person[name] = embs
        else:
            print(f"Skipping {name}: only {len(embs)} valid aligned crops (need >= {cfg.min_imgs_per_person}).")

    names = sorted(per_person.keys())
    if len(names) < 1:
        print("Not enough data to evaluate. Enroll more samples.")
        return

    # Genuine
    genuine_all: List[float] = []
    for name in names:
        genuine_all.extend(pairwise_distances(per_person[name], per_person[name], same=True))

    # Impostor
    impostor_all: List[float] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            impostor_all.extend(pairwise_distances(per_person[names[i]], per_person[names[j]],
same=False))

    genuine = np.array(genuine_all, dtype=np.float32)
    impostor = np.array(impostor_all, dtype=np.float32)

    print("\n=== Distance Distributions (cosine distance = 1 - cosine similarity) ===")
    print(f"Genuine (same person):   {describe(genuine)}")
    print(f"Impostor (diff persons): {describe(impostor)}")

    results = sweep_thresholds(genuine, impostor, cfg)

    # Choose threshold with FAR <= target_far and minimal FRR
    best = None
    for thr, far, frr in results:
        if far <= cfg.target_far:
            if best is None or frr < best[2]:
                best = (thr, far, frr)

    print("\n=== Threshold Sweep ===")
    stride = max(1, len(results) // 10)
    for thr, far, frr in results[::stride]:
        print(f"thr={thr:.2f}  FAR={far*100:5.2f}%  FRR={frr*100:5.2f}%")

    if best is not None:
        thr, far, frr = best
        print(
            f"\nSuggested threshold (target FAR {cfg.target_far*100:.1f}%): "
            f"thr={thr:.2f}  FAR={far*100:.2f}%  FRR={frr*100:.2f}%"
        )
    else:
        print(
            f"\nNo threshold in range met FAR <= {cfg.target_far*100:.1f}%. "
            "Try widening threshold sweep range or collecting more varied samples."
        )

    # Extra: recommend a similarity-style threshold too
    if best is not None:
        sim_thr = 1.0 - best[0]
        print(f"\n(Equivalent cosine similarity threshold ~ {sim_thr:.3f}, since sim = 1 - dist)")

    print()


if __name__ == "__main__":

    main()