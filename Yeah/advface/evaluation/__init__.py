"""評估指標與攻擊結果資料結構。"""

from advface.evaluation.attack_result import AttackResult, is_attack_success
from advface.evaluation.similarity import (
    cosine_similarity,
    cosine_similarity_or_nan,
    euclidean_distance,
)
from advface.evaluation.transfer import (
    TransferEvalResult,
    build_conclusion,
    evaluate_on_model,
    evaluate_transfer,
)

__all__ = [
    "AttackResult",
    "is_attack_success",
    "cosine_similarity",
    "cosine_similarity_or_nan",
    "euclidean_distance",
    "TransferEvalResult",
    "evaluate_on_model",
    "evaluate_transfer",
    "build_conclusion",
]
