"""
White-box → Black-box Transfer Evaluation。

只做評估：給定 original / adversarial / model，比較該模型自己的 embedding。
不含任何攻擊邏輯。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from advface.config import TRANSFER_OBSERVED_DELTA
from advface.evaluation.attack_result import is_attack_success
from advface.evaluation.similarity import cosine_similarity, euclidean_distance
from advface.models.base import EmbeddingModel


@dataclass
class TransferEvalResult:
    """單一模型上的 original vs adversarial 評估結果。"""

    model_name: str
    cosine_before: float
    cosine_after: float
    delta: float
    success: bool
    threshold: Optional[float] = None
    euclidean_distance: Optional[float] = None
    embedding_original: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    embedding_adversarial: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    error: Optional[str] = None
    # 無正式 threshold 時的觀測性 transfer 標記（不影響 surrogate success）
    transfer_observed: Optional[bool] = None

    def to_metrics_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "cosine_before": self.cosine_before,
            "cosine_after": self.cosine_after,
            "delta": self.delta,
            "success": bool(self.success),
            "threshold": self.threshold,
            "euclidean_distance": self.euclidean_distance,
            "transfer_observed": self.transfer_observed,
            "error": self.error,
        }


def _is_transfer_observed(result: TransferEvalResult) -> bool:
    """
    Victim transfer 是否觀測到。

    - 有正式 threshold：以 cosine_after < threshold 為準（success）
    - 無 threshold：以 delta >= TRANSFER_OBSERVED_DELTA 為觀測準則（非驗證門檻）
    """
    if result.error:
        return False
    if result.threshold is not None:
        return bool(result.success)
    if result.delta != result.delta:  # NaN
        return False
    return float(result.delta) >= float(TRANSFER_OBSERVED_DELTA)


def evaluate_on_model(
    original_bgr: np.ndarray,
    adversarial_bgr: np.ndarray,
    model: EmbeddingModel,
) -> TransferEvalResult:
    """
    在單一模型上比較 original vs adversarial。

    cosine_before：乾淨圖自我相似度（理論上 ≈ 1）
    cosine_after ：original vs adversarial
    delta        ：cosine_before - cosine_after
    success      ：cosine_after < model.threshold（若有門檻；否則 False）
    """
    name = getattr(model, "name", type(model).__name__)
    threshold = getattr(model, "threshold", None)

    try:
        emb_o = np.asarray(model.get_embedding(original_bgr, label="original"), dtype=np.float32).reshape(-1)
        emb_a = np.asarray(model.get_embedding(adversarial_bgr, label="adversarial"), dtype=np.float32).reshape(-1)
    except Exception as e:
        return TransferEvalResult(
            model_name=name,
            cosine_before=float("nan"),
            cosine_after=float("nan"),
            delta=float("nan"),
            success=False,
            threshold=float(threshold) if threshold is not None else None,
            euclidean_distance=None,
            transfer_observed=False,
            error=str(e),
        )

    cos_before = float(cosine_similarity(emb_o, emb_o))
    cos_after = float(cosine_similarity(emb_o, emb_a))
    delta = float(cos_before - cos_after)
    euc = float(euclidean_distance(emb_o, emb_a))

    if threshold is None:
        success = False
    else:
        success = is_attack_success(cos_after, threshold=float(threshold))

    result = TransferEvalResult(
        model_name=name,
        cosine_before=cos_before,
        cosine_after=cos_after,
        delta=delta,
        success=success,
        threshold=float(threshold) if threshold is not None else None,
        euclidean_distance=euc,
        embedding_original=emb_o,
        embedding_adversarial=emb_a,
        error=None,
    )
    result.transfer_observed = _is_transfer_observed(result)
    return result


def evaluate_models(
    original_bgr: np.ndarray,
    adversarial_bgr: np.ndarray,
    models: Mapping[str, EmbeddingModel],
) -> dict[str, TransferEvalResult]:
    """
    對任意數量模型做 original vs adversarial 評估。
    每個模型只用自己的 embedding；不跨模型比較。
    """
    return {
        role: evaluate_on_model(original_bgr, adversarial_bgr, model)
        for role, model in models.items()
    }


def evaluate_transfer(
    original_bgr: np.ndarray,
    adversarial_bgr: np.ndarray,
    *,
    surrogate: EmbeddingModel,
    victim: EmbeddingModel | None = None,
    victims: Mapping[str, EmbeddingModel] | None = None,
) -> dict[str, TransferEvalResult]:
    """分別在 surrogate / victim(s) 上評估；不跨模型比較 embedding。"""
    models: dict[str, EmbeddingModel] = {"surrogate": surrogate}
    if victims:
        models.update(victims)
    elif victim is not None:
        models["victim"] = victim
    return evaluate_models(original_bgr, adversarial_bgr, models)


def build_conclusion(surrogate: TransferEvalResult, victim: TransferEvalResult) -> dict[str, str]:
    """正式實驗報告用結論字串（固定用語）。"""
    whitebox = "Successful" if surrogate.success else "Failed"
    observed = _is_transfer_observed(victim)
    transfer = "Observed" if observed else "Not Observed"

    if whitebox == "Successful" and observed:
        body = (
            "The generated adversarial example successfully fools the surrogate model "
            "and the perturbation also transfers to the victim model."
        )
    elif whitebox == "Successful":
        body = (
            "The generated adversarial example successfully fools the surrogate model. "
            "However, the perturbation is not yet transferable to the victim model."
        )
    else:
        body = (
            "The generated adversarial example does not fool the surrogate model; "
            "transfer evaluation is inconclusive."
        )
    closing = "This experiment establishes the first transfer evaluation baseline."
    return {
        "whitebox": whitebox,
        "transfer": transfer,
        "body": body,
        "closing": closing,
        "transfer_label": transfer,
    }
