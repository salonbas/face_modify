import os
from pathlib import Path

# InsightFace
FACE_MODEL_NAME = "buffalo_l"
DEFAULT_DET_SIZE = (640, 640)
EMBEDDING_DIM = 512

# 常用預設圖（可改）
DEFAULT_IMAGE = "data/raw/sun.png"
DEFAULT_IMAGE2 = "data/raw/face2.jpg"

# 高斯實驗
DEFAULT_GAUSSIAN_EPS_START = 10
DEFAULT_GAUSSIAN_EPS_END = 100
DEFAULT_GAUSSIAN_EPS_STEP = 5
GAUSSIAN_FAIL_THRESHOLD = 0.4

# FGSM（ArcFace ONNX→Torch 梯度，eps 語意 [0,1] 對應約 ×255 像素域）
DEFAULT_FGSM_EPS_LIST = "0.01,0.03,0.07,0.1"


def insightface_providers() -> list[str]:
    """
    預設僅 CPU，方便 Colab / 無 NVIDIA / AMD 等環境重現。
    設環境變數 ADVFACE_PROVIDERS=cuda,cpu 可改為優先 GPU。
    """
    raw = os.environ.get("ADVFACE_PROVIDERS", "cpu").strip().lower()
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    mapping = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
    }
    out: list[str] = []
    for p in parts:
        if p in mapping and mapping[p] not in out:
            out.append(mapping[p])
    return out or ["CPUExecutionProvider"]


def project_root() -> Path:
    """假設從 repo 根目錄執行；否則為 cwd。"""
    return Path.cwd()


def ensure_project_dirs(base: Path | None = None) -> None:
    base = base or project_root()
    for rel in ("data/raw", "data/adversarial", "models",
                "results/fgsm", "results/pgd", "results/gaussian", "results/compare",
                "notebooks"):
        (base / rel).mkdir(parents=True, exist_ok=True)
