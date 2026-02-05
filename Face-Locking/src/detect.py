from typing import List, Tuple

import cv2
import mediapipe as mp


def detect_faces(image_bgr: cv2.Mat, min_confidence: float = 0.5) -> List[Tuple[int, int, int, int]]:
    """Detect faces and return bounding boxes (x1, y1, x2, y2)."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_bgr.shape[:2]

    with mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=min_confidence,
    ) as detector:
        results = detector.process(image_rgb)

    boxes: List[Tuple[int, int, int, int]] = []
    if not results.detections:
        return boxes

    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        x1 = int(bbox.xmin * width)
        y1 = int(bbox.ymin * height)
        x2 = int((bbox.xmin + bbox.width) * width)
        y2 = int((bbox.ymin + bbox.height) * height)
        boxes.append((x1, y1, x2, y2))

    return boxes
