from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

ARC_FACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

LANDMARK_INDEXES = [33, 263, 1, 61, 291]  # left eye, right eye, nose, mouth left, mouth right


def _extract_landmarks(landmarks, width: int, height: int) -> np.ndarray:
    points = []
    for idx in LANDMARK_INDEXES:
        lm = landmarks.landmark[idx]
        points.append([lm.x * width, lm.y * height])
    return np.array(points, dtype=np.float32)


def _landmarks_bbox(points: np.ndarray) -> Tuple[int, int, int, int]:
    x1, y1 = np.min(points, axis=0)
    x2, y2 = np.max(points, axis=0)
    return int(x1), int(y1), int(x2), int(y2)


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union == 0:
        return 0.0
    return inter_area / union


def align_face(
    image_bgr: cv2.Mat,
    face_box: Tuple[int, int, int, int],
    output_size: int = 112,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Align a face to 112x112 using 5-point landmarks from MediaPipe FaceMesh."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_bgr.shape[:2]

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None

    best_points = None
    best_iou = 0.0
    for landmarks in results.multi_face_landmarks:
        points = _extract_landmarks(landmarks, width, height)
        lm_box = _landmarks_bbox(points)
        iou = _bbox_iou(face_box, lm_box)
        if iou > best_iou:
            best_iou = iou
            best_points = points

    if best_points is None:
        return None

    src = best_points.astype(np.float32)
    dst = ARC_FACE_TEMPLATE.copy()
    dst[:, 0] = dst[:, 0] * (output_size / 112)
    dst[:, 1] = dst[:, 1] * (output_size / 112)

    transform, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if transform is None:
        return None

    aligned = cv2.warpAffine(image_bgr, transform, (output_size, output_size), borderValue=0.0)
    return aligned, src
