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
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from advface.config import FACENET_SIMILARITY_THRESHOLD


FACENET_WEIGHT_FILENAME = "20180402-114759-vggface2.pt"


def facenet_cache_dir() -> Path:
    """Return the repository-local Torch cache, independent of ADVFACE_ROOT."""
    return project_root_from_package() / ".cache" / "torch"


def project_root_from_package():
    # facenet.py -> models -> advface -> repository root
    return Path(__file__).resolve().parents[2]


def facenet_pretrained_weight_path() -> Path:
    return facenet_cache_dir() / "checkpoints" / FACENET_WEIGHT_FILENAME


def facenet_pretrained_available() -> bool:
    return facenet_pretrained_weight_path().is_file()


def _ensure_torch_cache_dir() -> None:
    """權重下載目錄固定於 repository，避免測試 root 改變 cache。"""
    cache = facenet_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(cache)


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

    def get_embeddings(self, images_bgr: list[np.ndarray], label: str = "images") -> np.ndarray:
        """Evaluate a trajectory in one MTCNN/model batch."""
        if not images_bgr:
            return np.empty((0, 512), dtype=np.float32)
        rgbs = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images_bgr]
        faces = self._mtcnn(rgbs)
        if faces is None or any(face is None for face in faces):
            raise ValueError(f"FaceNet/MTCNN 未偵測到人臉：{label}")
        if isinstance(faces, list):
            faces = self._torch.stack(faces)
        faces = faces.to(self.device)
        with self._torch.no_grad():
            emb = self._model(faces)
            emb = self._torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.detach().cpu().numpy().astype(np.float32)
