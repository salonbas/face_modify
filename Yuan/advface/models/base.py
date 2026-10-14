"""Embedding 模型薄介面：各模型自行 preprocessing，只比較自己的 original vs adversarial。"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingModel(Protocol):
    """
    統一契約：輸入 BGR uint8 圖，輸出 1-D float embedding。

    約束：
    - 每個實作負責自己的偵測／對齊／正規化。
    - 不同模型之間不得直接比較 embedding。
    - 僅在同一模型內比較 original vs adversarial。
    """

    name: str
    threshold: Optional[float]

    def get_embedding(self, image_bgr: np.ndarray, label: str = "image") -> np.ndarray:
        """回傳 1-D float32 embedding（建議已 L2-normalize）。"""
        ...
