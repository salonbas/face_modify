import os
from pathlib import Path

# InsightFace
FACE_MODEL_NAME = "buffalo_l"
ARCFACE_ONNX_FILENAME = "w600k_r50.onnx"
DEFAULT_DET_SIZE = (640, 640)
EMBEDDING_DIM = 512

# 研究上重複使用的常數
ALIGNED_FACE_SIZE = 112
PIXEL_MIN = 0.0
PIXEL_MAX = 255.0
ARCFACE_INPUT_MEAN = 127.5
ARCFACE_INPUT_SCALE = 127.5
SIMILARITY_THRESHOLD = 0.4
# FaceNet 尚無正式 verification threshold；勿硬套 ArcFace 0.4。
# None → 報告顯示 N/A；transfer 改以觀測性準則判定（見 evaluation.transfer）。
FACENET_SIMILARITY_THRESHOLD = None

# Transfer Evaluation Experiment v1 正式預設（與既有 PGD 研究設定一致）
DEFAULT_TRANSFER_EPS = "0.040"
DEFAULT_TRANSFER_STEPS = 200
DEFAULT_VICTIM_MODEL = "facenet"
# 無正式 victim threshold 時，觀測 transfer 的 cosine delta 下限
TRANSFER_OBSERVED_DELTA = 0.2

# 常用預設圖（可改）
DEFAULT_IMAGE = "data/raw/sun.png"
# 預留給未來 pair／targeted 實驗；目前 CLI 未使用
DEFAULT_IMAGE2 = "data/raw/face2.jpg"

# 高斯實驗
DEFAULT_GAUSSIAN_EPS_START = 10
DEFAULT_GAUSSIAN_EPS_END = 100
DEFAULT_GAUSSIAN_EPS_STEP = 5
GAUSSIAN_FAIL_THRESHOLD = SIMILARITY_THRESHOLD

# FGSM（ArcFace ONNX→Torch 梯度，eps 語意 [0,1] 對應約 ×255 像素域）
DEFAULT_FGSM_EPS_LIST = "0.01,0.03,0.07,0.1"

DEFAULT_ATTACK_SEED = 0


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
    """
    專案根目錄。優先順序：
      1. 環境變數 ADVFACE_ROOT
      2. 本套件檔案位置推導（advface/ 的上一層）
    不再假設一定要從 repo root 當 cwd 執行。
    """
    env = os.environ.get("ADVFACE_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    # advface/config.py → parents[1] = repository root
    return Path(__file__).resolve().parents[1]


def ensure_project_dirs(base: Path | None = None) -> None:
    base = base or project_root()
    for rel in (
        "data/raw",
        "results/fgsm",
        "results/pgd",
        "results/gaussian",
        "results/compare",
        "results/transfer",
    ):
        (base / rel).mkdir(parents=True, exist_ok=True)
