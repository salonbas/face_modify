"""
FaceNet（InceptionResnetV1）— 黑箱 Transfer 的 Victim 模型。

與 InsightFace ArcFace 完全分離：
- 自己的人臉偵測（MTCNN）
- 自己的對齊／裁切（160×160）
- 自己的正規化
不得與 ArcFace embedding 直接比較。
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from advface.config import FACENET_SIMILARITY_THRESHOLD, project_root


def _ensure_torch_cache_dir() -> None:
    """權重下載目錄改到專案內，避免依賴不可寫的 ~/.cache。"""
    cache = project_root() / ".cache" / "torch"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(cache))


class FaceNetEmbedder:
    """facenet-pytorch InceptionResnetV1（VGGFace2）評估介面。"""

    name: str = "facenet_vggface2"
    threshold: Optional[float] = FACENET_SIMILARITY_THRESHOLD

    def __init__(self, device: Optional[str] = None) -> None:
        _ensure_torch_cache_dir()
        try:
            import torch
            from facenet_pytorch import InceptionResnetV1, MTCNN
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "需要 facenet-pytorch。請執行：pip install facenet-pytorch --no-deps"
            ) from e

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._torch = torch
        # keep_all=False：取最大臉；post_process=True：輸出已正規化到約 [-1,1]
        self._mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=device,
            keep_all=False,
        )
        self._model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        for p in self._model.parameters():
            p.requires_grad_(False)

    def get_embedding(self, image_bgr: np.ndarray, label: str = "image") -> np.ndarray:
        """BGR uint8 → FaceNet 自有 preprocessing → L2-normalized 512-d embedding。"""
        torch = self._torch
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError(f"空白影像：{label}")

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # MTCNN 接受 PIL 或 ndarray (H,W,3) uint8 RGB
        face = self._mtcnn(rgb)
        if face is None:
            raise ValueError(f"FaceNet/MTCNN 未偵測到人臉：{label}")

        if face.dim() == 3:
            face = face.unsqueeze(0)
        face = face.to(self.device)

        with torch.no_grad():
            emb = self._model(face)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        out = emb.detach().cpu().numpy().reshape(-1).astype(np.float32)
        if out.size == 0:
            raise ValueError(f"FaceNet embedding 為空：{label}")
        return out
