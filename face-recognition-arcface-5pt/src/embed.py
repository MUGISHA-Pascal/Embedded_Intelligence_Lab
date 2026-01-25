# src/embed.py
"""
Embedding stage (ArcFace ONNX) using your working pipeline:

camera
-> Haar detection
-> FaceMesh 5pt
-> align_face_5pt (112x112)
-> ArcFace embedding
-> vector visualization (education)

Run:
 python -m src.embed

Keys:
 q : quit
 p : print embedding stats to terminal
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import time
import cv2
import numpy as np
import onnxruntime as ort

from .haar_5pt import Haar5ptDetector, align_face_5pt


# -------------------------
# Data
# -------------------------

@dataclass
class EmbeddingResult:
    embedding: np.ndarray   # (D,) float32, L2-normalized
    norm_before: float
    dim: int


# -------------------------
# Embedder
# -------------------------

class ArcFaceEmbedderONNX:
    """
    ArcFace / InsightFace-style ONNX embedder.
    Input: aligned 112x112 BGR image.
    Output: L2-normalized embedding vector.
    """

    def __init__(
        self,
        model_path: str = "models/embedder_arcface.onnx",
        input_size: Tuple[int, int] = (112, 112),
        debug: bool = False,
    ):
        self.in_w, self.in_h = input_size
        self.debug = debug

        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

        if debug:
            print("[embed] model loaded")
            print("[embed] input:", self.sess.get_inputs()[0].shape)

    def _preprocess(self, aligned_bgr: np.ndarray) -> np.ndarray:
        if aligned_bgr.shape[:2] != (self.in_h, self.in_w):
            aligned_bgr = cv2.resize(aligned_bgr, (self.in_w, self.in_h))

        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        x = np.transpose(rgb, (2, 0, 1))[None, ...]
        return x.astype(np.float32)

    @staticmethod
    def _l2_normalize(v: np.ndarray, eps: float = 1e-12):
        n = float(np.linalg.norm(v) + eps)
        return (v / n).astype(np.float32), n

    def embed(self, aligned_bgr: np.ndarray) -> EmbeddingResult:
        x = self._preprocess(aligned_bgr)
        y = self.sess.run([self.out_name], {self.in_name: x})[0]
        v = y.reshape(-1).astype(np.float32)
        v_norm, n0 = self._l2_normalize(v)
        return EmbeddingResult(v_norm, n0, v_norm.size)


# -------------------------
# Visualization helpers
# -------------------------

def draw_text_block(img, lines, origin=(10, 30), scale=0.7, color=(0, 255, 0)):
    x, y = origin
    for line in lines:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
        y += int(28 * scale)


def draw_embedding_matrix(
    img: np.ndarray,
    emb: np.ndarray,
    top_left=(10, 220),
    cell_scale: int = 6,
    title: str = "embedding"
):
    """
    Visualize embedding vector as a heatmap matrix.
    """
    D = emb.size
    cols = int(np.ceil(np.sqrt(D)))
    rows = int(np.ceil(D / cols))

    mat = np.zeros((rows, cols), dtype=np.float32)
    mat.flat[:D] = emb

    norm = (mat - mat.min()) / (mat.max() - mat.min() + 1e-6)
    gray = (norm * 255).astype(np.uint8)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    heat = cv2.resize(
        heat,
        (cols * cell_scale, rows * cell_scale),
        interpolation=cv2.INTER_NEAREST,
    )

    x, y = top_left
    h, w = heat.shape[:2]
    ih, iw = img.shape[:2]

    if x + w > iw or y + h > ih:
        return 0, 0

    img[y:y+h, x:x+w] = heat
    cv2.putText(
        img,
        title,
        (x, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2,
    )
    return w, h


def emb_preview_str(emb: np.ndarray, n: int = 8) -> str:
    vals = " ".join(f"{v:+.3f}" for v in emb[:n])
    return f"vec[0:{n}]: {vals} ..."


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# -------------------------
# Demo
# -------------------------

def main():
    cap = cv2.VideoCapture(0)

    det = Haar5ptDetector(
        min_size=(70, 70),
        smooth_alpha=0.80,
        debug=False,
    )

    emb_model = ArcFaceEmbedderONNX(
        model_path="models/embedder_arcface.onnx",
        debug=False,
    )

    prev_emb: Optional[np.ndarray] = None

    print("Embedding Demo running. Press 'q' to quit, 'p' to print embedding.")

    t0 = time.time()
    frames = 0
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vis = frame.copy()
        faces = det.detect(frame, max_faces=1)
        info = []

        if faces:
            f = faces[0]

            # draw detection
            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
            for (x, y) in f.kps.astype(int):
                cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)

            # align + embed
            aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
            res = emb_model.embed(aligned)

            info.append(f"embedding dim: {res.dim}")
            info.append(f"norm(before L2): {res.norm_before:.2f}")

            if prev_emb is not None:
                sim = cosine_similarity(prev_emb, res.embedding)
                info.append(f"cos(prev,this): {sim:.3f}")

            prev_emb = res.embedding

            # aligned preview (top-right)
            aligned_small = cv2.resize(aligned, (160, 160))
            h, w = vis.shape[:2]
            vis[10:170, w-170:w-10] = aligned_small

            # --------- VISUALIZATION LAYOUT ---------
            draw_text_block(vis, info, origin=(10, 30))

            HEAT_X, HEAT_Y = 10, 220
            CELL_SCALE = 6

            ww, hh = draw_embedding_matrix(
                vis,
                res.embedding,
                top_left=(HEAT_X, HEAT_Y),
                cell_scale=CELL_SCALE,
                title="embedding heatmap",
            )

            if ww > 0:
                cv2.putText(
                    vis,
                    emb_preview_str(res.embedding),
                    (HEAT_X, HEAT_Y + hh + 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 200),
                    2,
                )

        else:
            draw_text_block(vis, ["no face"], origin=(10, 30), color=(0, 0, 255))

        # FPS
        frames += 1
        dt = time.time() - t0
        if dt >= 1.0:
            fps = frames / dt
            frames = 0
            t0 = time.time()
        cv2.putText(vis, f"fps: {fps:.1f}", (10, vis.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Face Embedding", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("p") and prev_emb is not None:
            print("[embedding]")
            print(" dim:", prev_emb.size)
            print(" min/max:", prev_emb.min(), prev_emb.max())
            print(" first10:", prev_emb[:10])

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()