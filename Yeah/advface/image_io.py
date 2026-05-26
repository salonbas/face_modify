from pathlib import Path

import cv2
import numpy as np


def load_bgr(path: str | Path) -> np.ndarray:
    p = Path(path)
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"無法讀取圖片：{p}")
    return img


def image_stem(path: str | Path) -> str:
    return Path(path).stem
