"""
Face Engine for Portable Clinic.
Wraps InsightFace (SCRFD detection + ArcFace recognition).
"""

import cv2
import numpy as np
from .config import DET_SIZE, DET_THRESH, _DET_SIZE_EXPLICIT, INFER_MAX_WIDTH, MIN_FACE_PX


def get_onnx_providers() -> list[str]:
    """Detect available ONNX Runtime execution providers (GPU / DirectML / CPU)."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "DmlExecutionProvider" in available:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
    except ImportError:
        pass
    return ["CPUExecutionProvider"]


class FaceEngine:
    """Manages SCRFD detector and ArcFace buffalo_l recognition model."""

    def __init__(self):
        from insightface.app import FaceAnalysis

        self.providers = get_onnx_providers()
        self.provider = self.providers[0]
        self.gpu_in_use = self.provider != "CPUExecutionProvider"

        det_size = DET_SIZE
        if not self.gpu_in_use and not _DET_SIZE_EXPLICIT:
            # Auto-lower detection size to 384x384 for smooth real-time CPU performance
            det_size = (384, 384)
            print(f"[FaceEngine] CPU Mode — auto-lowering det_size to {det_size} for real-time throughput.")

        print(f"[FaceEngine] Provider: {self.provider} (GPU: {self.gpu_in_use})")
        self.app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=self.providers,
        )
        ctx_id = 0 if self.gpu_in_use else -1
        self.app.prepare(ctx_id=ctx_id, det_thresh=DET_THRESH, det_size=det_size)

    def embed_image(self, img: np.ndarray) -> np.ndarray | None:
        """Extract normalized embedding from the largest face in the image."""
        faces = self.app.get(img)
        if not faces:
            return None
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return largest.normed_embedding

    def detect_faces(self, img: np.ndarray) -> list[dict]:
        """
        Detect all faces in the frame.
        Returns list of dicts: {'box': [x1, y1, x2, y2], 'embedding': np.ndarray, 'score': float}
        """
        infer_img, scale = self._resize_max_width(img, INFER_MAX_WIDTH)
        faces = self.app.get(infer_img)
        results = []
        for face in faces or []:
            x1, y1, x2, y2 = (face.bbox.astype(np.float32) / scale).astype(int)
            if (x2 - x1) < MIN_FACE_PX or (y2 - y1) < MIN_FACE_PX:
                continue
            emb = face.normed_embedding
            if emb is None:
                continue
            results.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "embedding": emb,
                "det_score": float(face.det_score) if hasattr(face, "det_score") else 1.0,
            })
        return results

    @staticmethod
    def _resize_max_width(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
        """Resize frame if wider than max_width."""
        if max_width <= 0:
            return frame, 1.0
        h, w = frame.shape[:2]
        if w <= max_width:
            return frame, 1.0
        scale = max_width / w
        return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_LINEAR), scale
