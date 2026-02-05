from __future__ import annotations

import os
from typing import Iterable, List, Tuple

import cv2
import numpy as np

from .align import align_face
from .detect import detect_faces
from .embed import ArcFaceEmbedder
from .utils import ensure_dir, l2_normalize, save_image


def enroll_identity(
    name: str,
    image_paths: Iterable[str],
    embedder: ArcFaceEmbedder,
    output_dir: str,
    detection_confidence: float = 0.6,
) -> Tuple[int, str]:
    """Enroll a single identity from image paths.

    Saves aligned 112x112 crops and embeddings to disk.
    Returns number of samples enrolled and the identity folder.
    """
    identity_dir = os.path.join(output_dir, name)
    crops_dir = os.path.join(identity_dir, "crops")
    ensure_dir(crops_dir)

    embeddings: List[np.ndarray] = []
    sample_count = 0

    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        if image is None:
            continue

        boxes = detect_faces(image, min_confidence=detection_confidence)
        if not boxes:
            continue

        # Use the largest box for enrollment
        boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        aligned = align_face(image, boxes[0])
        if aligned is None:
            continue

        aligned_face, _ = aligned
        embedding = embedder.embed(aligned_face)
        embeddings.append(embedding)

        crop_path = os.path.join(crops_dir, f"{idx:04d}.jpg")
        save_image(crop_path, aligned_face)
        sample_count += 1

    if embeddings:
        embeddings_array = l2_normalize(np.stack(embeddings), axis=1)
        np.save(os.path.join(identity_dir, "embeddings.npy"), embeddings_array)

    return sample_count, identity_dir
