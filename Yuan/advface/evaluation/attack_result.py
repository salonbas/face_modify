"""統一的白盒攻擊結果欄位（不含黑箱專屬欄位）。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

from advface.config import SIMILARITY_THRESHOLD


def is_attack_success(cosine: float, threshold: float = SIMILARITY_THRESHOLD) -> bool:
    if cosine != cosine:  # NaN
        return False
    return float(cosine) < threshold


@dataclass
class AttackResult:
    """單次 eps 的驗收結果；欄位名稱固定以便 CSV／verify 共用。"""

    eps: float
    cosine: float
    success: bool
    attack_mode: str
    output_image_path: Optional[str] = None
    steps: Optional[int] = None
    euclidean: Optional[float] = None
    eps_255: Optional[float] = None
    # 攻擊過程暫存影像（不寫入 CSV）
    attacked_bgr: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def to_metrics_row(self) -> dict[str, Any]:
        return {
            "eps": self.eps,
            "eps_255": self.eps_255 if self.eps_255 is not None else self.eps * 255.0,
            "steps": self.steps if self.steps is not None else "",
            "cosine": self.cosine,
            "success": int(self.success),
        }

    def to_public_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("attacked_bgr", None)
        return d
