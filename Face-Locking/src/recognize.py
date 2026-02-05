from __future__ import annotations

import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .align import align_face
from .detect import detect_faces
from .embed import ArcFaceEmbedder


def load_identity_database(identities_dir: str) -> Dict[str, np.ndarray]:
    """Load embeddings for each identity into memory."""
    database: Dict[str, np.ndarray] = {}
    if not os.path.isdir(identities_dir):
        return database

    for name in os.listdir(identities_dir):
        identity_dir = os.path.join(identities_dir, name)
        embedding_path = os.path.join(identity_dir, "embeddings.npy")
        if os.path.isfile(embedding_path):
            embeddings = np.load(embedding_path)
            database[name] = embeddings

    return database


def match_identity(embedding: np.ndarray, database: Dict[str, np.ndarray]) -> Tuple[str, float]:
    """Find best matching identity and similarity score."""
    best_name = "Unknown"
    best_score = -1.0

    for name, embeddings in database.items():
        scores = np.dot(embeddings, embedding)
        score = float(np.max(scores))
        if score > best_score:
            best_score = score
            best_name = name

    return best_name, best_score


def recognize_frame(
    frame: cv2.Mat,
    embedder: ArcFaceEmbedder,
    database: Dict[str, np.ndarray],
    threshold: float = 0.45,
) -> List[Tuple[Tuple[int, int, int, int], str, float]]:
    """Recognize faces in a frame and return list of (box, name, score)."""
    results: List[Tuple[Tuple[int, int, int, int], str, float]] = []
    boxes = detect_faces(frame)

    for box in boxes:
        aligned = align_face(frame, box)
        if aligned is None:
            continue
        aligned_face, _ = aligned
        embedding = embedder.embed(aligned_face)
        name, score = match_identity(embedding, database)
        if score < threshold:
            name = "Unknown"
        results.append((box, name, score))

    return results
