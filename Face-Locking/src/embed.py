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
from pathlib import Path
from typing import Tuple, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

import time
import cv2
import numpy as np
import onnxruntime as ort
from math import ceil

from .haar_5pt import Haar5ptDetector, align_face_5pt

# Resolve model path: project_root/models/embedder_arcface.onnx (works from any cwd)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "embedder_arcface.onnx"

# ArcFace ONNX 112x112 embedder – auto-download if missing or empty (e.g. after init_project)
_ARCFACE_ONNX_URL = "https://huggingface.co/garavv/arcface-onnx/resolve/main/arc.onnx"


def _download_arcface_model(dest: Path) -> bool:
    """Download ArcFace ONNX to dest. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = Request(_ARCFACE_ONNX_URL, headers={"User-Agent": "Python-urllib"})
        with urlopen(req, timeout=60) as resp:
            data = resp.read()
        if len(data) < 100:
            return False
        dest.write_bytes(data)
        return True
    except (URLError, OSError, Exception):
        return False

# ---
# Data
# ---

@dataclass
class EmbeddingResult:
    embedding: np.ndarray  # (D,) float32, L2-normalized
    norm_before: float
    dim: int

# ---
# Embedder
# ---

class ArcFaceEmbedderONNX:
    """
    ArcFace / InsightFace-style ONNX embedder.
    Input: aligned 112x112 BGR image.
    Output: L2-normalized embedding vector.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: Tuple[int, int] = (112, 112),
        debug: bool = False,
    ):
        self.in_w, self.in_h = input_size
        self.debug = debug

        path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        path = path.resolve()

        # Auto-download if missing or empty (e.g. placeholder from init_project)
        if not path.exists() or path.stat().st_size < 100:
            if self.debug:
                print("[embed] Model missing or empty, attempting download...")
            if _download_arcface_model(path):
                if self.debug:
                    print("[embed] Downloaded ArcFace ONNX to", path)
            else:
                if not path.exists():
                    raise FileNotFoundError(
                        f"ArcFace ONNX model not found: {path}\n"
                        "Download failed. Get an ArcFace ONNX (112x112) and save as models/embedder_arcface.onnx\n"
                        f"Or try manually: {_ARCFACE_ONNX_URL}"
                    )
                raise ValueError(
                    f"ArcFace model file is empty or invalid: {path} ({path.stat().st_size} bytes).\n"
                    "Delete it and run again to auto-download, or download a valid ONNX and save as models/embedder_arcface.onnx"
                )
        try:
            self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        except Exception as e:
            err = str(e)
            if "ModelProto does not have a graph" in err or "does not have a graph" in err.lower():
                raise ValueError(
                    f"Invalid or corrupted ONNX model: {path}\n"
                    "The file is not a valid ONNX model (empty or wrong format). "
                    "Download a valid ArcFace ONNX embedder and save as models/embedder_arcface.onnx"
                ) from e
            raise
        inp = self.sess.get_inputs()[0]
        self.in_name = inp.name
        self.out_name = self.sess.get_outputs()[0].name
        # Many ArcFace ONNX (e.g. garavv) expect NHWC (batch, H, W, C); others NCHW (batch, C, H, W)
        shape = getattr(inp, "shape", []) or []
        try:
            # shape like [1, 112, 112, 3] -> NHWC; [1, 3, 112, 112] -> NCHW
            dim1 = int(shape[1]) if len(shape) > 1 else None
            self._input_nhwc = dim1 == self.in_h or dim1 == 112  # second dim is height -> NHWC
        except (TypeError, ValueError):
            self._input_nhwc = True  # default NHWC for garavv/arcface-onnx

        if debug:
            print("[embed] model loaded")
            print("[embed] input:", inp.shape, "layout:", "NHWC" if self._input_nhwc else "NCHW")
            print("[embed] output:", self.sess.get_outputs()[0].shape)

    def _preprocess(self, aligned_bgr: np.ndarray) -> np.ndarray:
        if aligned_bgr.shape[:2] != (self.in_h, self.in_w):
            aligned_bgr = cv2.resize(aligned_bgr, (self.in_w, self.in_h))

        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        if self._input_nhwc:
            x = rgb[None, ...]  # (1, H, W, C) NHWC
        else:
            x = np.transpose(rgb, (2, 0, 1))[None, ...]  # (1, C, H, W) NCHW
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

# ---
# Visualization helpers
# ---

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
    cols = int(ceil(np.sqrt(D)))
    rows = int(ceil(D / cols))

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
    vals = " ".join(f"{v:.3f}" for v in emb[:n])
    return f"vec[0:{n}]: {vals} ..."

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

# ---
# Demo
# ---

def main():
    cap = cv2.VideoCapture(0)

    det = Haar5ptDetector(
        min_size=(70, 70),
        smooth_alpha=0.80,
        debug=False,
    )

    emb_model = ArcFaceEmbedderONNX(debug=False)

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

            # --- VISUALIZATION LAYOUT ---
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
            print("dim:", prev_emb.size)
            print("min/max:", prev_emb.min(), prev_emb.max())
            print("first10:", prev_emb[:10])

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()