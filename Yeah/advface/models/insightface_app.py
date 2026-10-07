"""
評估時完整 InsightFace FaceAnalysis pipeline。

路徑：偵測 → 對齊 → embedding（`face.normed_embedding`）。
與攻擊時可微分 ArcFace Torch 路徑不同；驗收應走此模組。
"""
from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
from typing import Any

import numpy as np
from insightface.app import FaceAnalysis

from advface.config import (
    DEFAULT_DET_SIZE,
    EMBEDDING_DIM,
    FACE_MODEL_NAME,
    insightface_providers,
    project_root,
)


BUFFALO_L_FILENAMES = (
    "w600k_r50.onnx",
    "det_10g.onnx",
    "1k3d68.onnx",
    "2d106det.onnx",
    "genderage.onnx",
)


def repo_insightface_root() -> Path:
    """Return the repository-owned InsightFace root (not a cache root)."""
    return project_root() / "models" / "insightface"


def insightface_model_root() -> Path:
    """Prefer the reviewed repository-local package; retain legacy fallback."""
    local_package = repo_insightface_root() / "buffalo_l"
    if all((local_package / name).is_file() for name in BUFFALO_L_FILENAMES):
        return repo_insightface_root()
    return Path.home() / ".insightface"


def _insightface_verbose_stdout() -> bool:
    """設 ADVFACE_VERBOSE=1 / true / yes 時保留 InsightFace 的 print 洗版（除錯用）。"""
    return os.environ.get("ADVFACE_VERBOSE", "").strip().lower() in ("1", "true", "yes")


def create_face_app(det_size: tuple[int, int] | None = None) -> FaceAnalysis:
    det = det_size or DEFAULT_DET_SIZE
    if _insightface_verbose_stdout():
        app = FaceAnalysis(
            name=FACE_MODEL_NAME, root=str(insightface_model_root()),
            providers=insightface_providers(),
        )
        app.prepare(ctx_id=0, det_size=det)
        return app

    # InsightFace / model_zoo 用 print 印載入過程；預設關掉以免洗版
    with contextlib.redirect_stdout(io.StringIO()):
        app = FaceAnalysis(
            name=FACE_MODEL_NAME, root=str(insightface_model_root()),
            providers=insightface_providers(),
        )
        app.prepare(ctx_id=0, det_size=det)
    return app


def pick_best_face(faces: list[Any], label: str) -> Any:
    if not faces:
        raise ValueError(f"圖片中沒有偵測到人臉：{label}")
    return max(faces, key=lambda f: getattr(f, "det_score", 0.0))


def get_embedding_from_path(app: FaceAnalysis, img_path: str | Path) -> np.ndarray:
    """由路徑讀圖並抽 embedding。目前 scripts 未直接呼叫；保留供外部／notebook 使用。"""
    from advface.image_io import load_bgr

    path = Path(img_path)
    img = load_bgr(path)
    face = pick_best_face(app.get(img), str(path))
    emb = np.asarray(face.normed_embedding, dtype=np.float32)
    if emb.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"Embedding 維度異常：{path}={emb.shape}（預期 {EMBEDDING_DIM}）")
    return emb


def get_embedding_from_bgr(app: FaceAnalysis, img_bgr: np.ndarray, label: str) -> np.ndarray:
    face = pick_best_face(app.get(img_bgr), label)
    emb = np.asarray(face.normed_embedding, dtype=np.float32)
    if emb.shape[0] != EMBEDDING_DIM:
        raise ValueError(f"Embedding 維度異常：{label}={emb.shape}（預期 {EMBEDDING_DIM}）")
    return emb


class InsightFaceEmbedder:
    """
    FaceAnalysis 評估包裝：符合 EmbeddingModel 契約。
    供 Transfer Evaluation 當 surrogate（或同介面的 victim）使用。
    不改變既有 get_embedding_from_bgr 行為。
    """

    name: str = f"insightface_{FACE_MODEL_NAME}"
    threshold: float | None = None

    def __init__(
        self,
        det_size: tuple[int, int] | None = None,
        app: FaceAnalysis | None = None,
    ) -> None:
        from advface.config import SIMILARITY_THRESHOLD

        self.app = app if app is not None else create_face_app(det_size=det_size)
        self.threshold = float(SIMILARITY_THRESHOLD)

    def get_embedding(self, image_bgr: np.ndarray, label: str = "image") -> np.ndarray:
        return get_embedding_from_bgr(self.app, image_bgr, label)
