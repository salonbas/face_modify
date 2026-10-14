"""模型載入：評估用 InsightFace FaceAnalysis，攻擊用可微分 ArcFace Torch，Transfer victim 用 FaceNet。"""

from __future__ import annotations

from typing import Optional

from advface.models.arcface_torch import load_arcface_torch
from advface.models.facenet import FaceNetEmbedder
from advface.models.insightface_app import (
    InsightFaceEmbedder,
    create_face_app,
    get_embedding_from_bgr,
    get_embedding_from_path,
    pick_best_face,
)


def load_embedder(
    name: str,
    *,
    device: Optional[str] = None,
    det_size: tuple[int, int] | None = None,
):
    """依名稱建立評估用 EmbeddingModel。新增 victim 時在此加一分支即可。"""
    key = name.strip().lower().replace("-", "_")
    if key in ("insightface", "buffalo_l", "insightface_buffalo_l", "arcface"):
        return InsightFaceEmbedder(det_size=det_size)
    if key in ("facenet", "facenet_vggface2"):
        return FaceNetEmbedder(device=device)
    raise ValueError(f"未知 embedding 模型：{name}（支援：insightface_buffalo_l, facenet_vggface2）")


__all__ = [
    "create_face_app",
    "pick_best_face",
    "get_embedding_from_bgr",
    "get_embedding_from_path",
    "InsightFaceEmbedder",
    "FaceNetEmbedder",
    "load_arcface_torch",
    "load_embedder",
]
