"""InsightFace buffalo_sc MobileFaceNet, using its official ONNX pipeline."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from advface.config import DEFAULT_DET_SIZE, project_root


class MobileFaceNetEmbedder:
    name = "mobilefacenet_buffalo_sc"
    threshold = None  # Set only after independent Calibration v2 training.

    def __init__(self, det_size: tuple[int, int] = DEFAULT_DET_SIZE) -> None:
        from insightface.model_zoo import get_model

        folder = project_root() / "models/insightface/buffalo_sc"
        detector, recognizer = folder / "det_500m.onnx", folder / "w600k_mbf.onnx"
        if not detector.is_file() or not recognizer.is_file():
            raise FileNotFoundError("official buffalo_sc detector and recognizer are required")
        self.detector = get_model(str(detector), providers=["CPUExecutionProvider"])
        self.recognizer = get_model(str(recognizer), providers=["CPUExecutionProvider"])
        self.detector.prepare(ctx_id=-1, input_size=det_size)
        self.recognizer.prepare(ctx_id=-1)

    def get_embedding(self, image_bgr: np.ndarray, label: str = "image") -> np.ndarray:
        from insightface.utils import face_align

        boxes, keypoints = self.detector.detect(image_bgr, max_num=1)
        if len(boxes) != 1 or keypoints is None:
            raise ValueError(f"no face detected in {label}")
        crop = face_align.norm_crop(image_bgr, landmark=keypoints[0],
                                    image_size=self.recognizer.input_size[0])
        embedding = np.asarray(self.recognizer.get_feat(crop), dtype=np.float32).reshape(-1)
        if embedding.shape != (512,) or not np.isfinite(embedding).all() or np.linalg.norm(embedding) == 0:
            raise ValueError(f"invalid MobileFaceNet embedding for {label}")
        return embedding / np.linalg.norm(embedding)
