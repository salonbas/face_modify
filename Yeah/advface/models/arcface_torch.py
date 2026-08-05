"""
攻擊時可微分 ArcFace（ONNX → PyTorch）。

將 InsightFace buffalo_l 的 `w600k_r50.onnx` 經 onnx2torch 轉換並快取，
供 FGSM / PGD 對輸入求梯度。與評估時完整 FaceAnalysis 路徑分離。
"""
from __future__ import annotations

import os
from typing import Optional

from advface.config import ARCFACE_ONNX_FILENAME, FACE_MODEL_NAME


def _recognition_onnx_path() -> str:
    from insightface.utils import ensure_available

    root = os.path.expanduser("~/.insightface")
    model_dir = ensure_available("models", FACE_MODEL_NAME, root=root)
    path = os.path.join(model_dir, ARCFACE_ONNX_FILENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到 ArcFace ONNX：{path}")
    return path


_TORCH_ARC: Optional[object] = None


def load_arcface_torch(device: str):
    """
    把 InsightFace buffalo_l 的 w600k_r50.onnx 轉成可微分 PyTorch 模型並快取。

    需要：pip install onnx2torch
    注意：onnx2torch 轉換時需要在 ONNX 所在目錄建立暫存檔，
          但 ~/.insightface/ 通常沒有寫入權限，
          故先把 ONNX 複製到系統 tmp 目錄再轉換。
    """
    global _TORCH_ARC
    if _TORCH_ARC is not None:
        return _TORCH_ARC

    try:
        import onnx2torch
    except ImportError as e:
        raise RuntimeError("需要 onnx2torch。請執行：pip install onnx2torch") from e

    import shutil
    import tempfile

    import torch

    src_path = _recognition_onnx_path()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_onnx = shutil.copy2(src_path, tmpdir)
        model = onnx2torch.convert(tmp_onnx)

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _TORCH_ARC = model
    return _TORCH_ARC
