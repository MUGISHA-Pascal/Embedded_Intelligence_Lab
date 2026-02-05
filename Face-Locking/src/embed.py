from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from .utils import l2_normalize


class ArcFaceEmbedder:
    """ArcFace ONNX embedder running on CPU."""

    def __init__(self, model_path: str) -> None:
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, image_bgr: cv2.Mat) -> np.ndarray:
        """Prepare aligned 112x112 BGR image to model input."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = image_rgb.astype(np.float32)
        image = (image - 127.5) / 128.0
        image = np.transpose(image, (2, 0, 1))
        return np.expand_dims(image, axis=0)

    def embed(self, aligned_face_bgr: cv2.Mat) -> np.ndarray:
        """Generate L2-normalized embedding for an aligned face."""
        input_tensor = self.preprocess(aligned_face_bgr)
        embedding = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        embedding = embedding.reshape(-1)
        return l2_normalize(embedding)
