from typing import Generator

import cv2


def camera_stream(camera_index: int = 0) -> Generator[cv2.Mat, None, None]:
    """Yield frames from the webcam until it is closed."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()
