import os
from typing import Optional

import cv2
import numpy as np


def l2_normalize(vec: np.ndarray, axis: int = -1, epsilon: float = 1e-12) -> np.ndarray:
    """L2-normalize a vector or batch of vectors."""
    norm = np.linalg.norm(vec, axis=axis, keepdims=True)
    return vec / np.maximum(norm, epsilon)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    a_norm = l2_normalize(a)
    b_norm = l2_normalize(b)
    return float(np.dot(a_norm, b_norm))


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_image(path: str, image_bgr: np.ndarray) -> None:
    """Save a BGR image to disk, creating parent dirs if needed."""
    ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, image_bgr)


def clip_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    """Clip bounding box to image bounds."""
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    return x1, y1, x2, y2


def to_int_tuple(point: np.ndarray) -> tuple[int, int]:
    """Convert float point to int tuple."""
    return int(round(point[0])), int(round(point[1]))
