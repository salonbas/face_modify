"""
單次 Transfer experiment unit 的薄封裝。

實際執行走 advface.experiments.runner.run_experiment。
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from advface.config import DEFAULT_ATTACK_SEED
from advface.experiments.runner import apply_attack_seed, run_experiment
from advface.models.base import EmbeddingModel


def run_canonical_whitebox_attack(
    img_bgr: np.ndarray,
    *,
    attack: str,
    eps: float,
    steps: int,
    app: Any,
    device: Optional[str] = None,
    seed: int = DEFAULT_ATTACK_SEED,
) -> np.ndarray:
    """相容舊呼叫：只回傳對抗圖。新程式請用 run_experiment。"""
    from advface.attacks.registry import AttackConfig, apply_attack

    apply_attack_seed(seed)
    out = apply_attack(
        img_bgr,
        attack=attack,
        app=app,
        config=AttackConfig(name=attack, eps=float(eps), steps=int(steps), seed=seed),
        device=device,
    )
    return out.adversarial_bgr


def attack_and_evaluate(
    original_bgr: np.ndarray,
    *,
    attack: str,
    eps: float,
    steps: int,
    app: Any,
    surrogate: EmbeddingModel,
    victim: EmbeddingModel,
    device: Optional[str] = None,
    seed: int = DEFAULT_ATTACK_SEED,
) -> tuple[np.ndarray, dict]:
    """攻擊 + surrogate/victim 評估。不寫檔。"""
    result = run_experiment(
        original_bgr,
        attack=attack,
        surrogate=surrogate,
        victims={"victim": victim},
        eps=eps,
        steps=steps,
        device=device,
        seed=seed,
        app=app,
    )
    assert result.adversarial_bgr is not None
    return result.adversarial_bgr, {
        "surrogate": result.surrogate,
        "victim": result.primary_victim,
    }
