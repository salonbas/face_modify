"""
Canonical experiment execution。

run_experiment(image, attack, surrogate, victims, config)
  → Attack → Adversarial → Surrogate / Victim evaluation → ExperimentResult

Single-image 與 batch benchmark 都走這條路徑。
"""
from __future__ import annotations

import time
from typing import Mapping, Optional, Union

import numpy as np

from advface.attacks.registry import AttackConfig, apply_attack
from advface.config import DEFAULT_ATTACK_SEED
from advface.evaluation.perturbation import perturbation_metrics
from advface.evaluation.transfer import evaluate_models
from advface.experiments.result import ExperimentResult
from advface.models.base import EmbeddingModel

VictimMap = Mapping[str, EmbeddingModel]


def apply_attack_seed(seed: int) -> None:
    try:
        import torch

        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
    except Exception:
        np.random.seed(int(seed))


def _normalize_victims(
    victims: Optional[Union[EmbeddingModel, VictimMap]],
) -> dict[str, EmbeddingModel]:
    if victims is None:
        return {}
    if isinstance(victims, Mapping):
        return dict(victims)
    return {"victim": victims}


def run_experiment(
    original_bgr: np.ndarray,
    *,
    attack: str,
    surrogate: EmbeddingModel,
    victims: Optional[Union[EmbeddingModel, VictimMap]] = None,
    eps: float,
    steps: Optional[int] = None,
    config: Optional[AttackConfig] = None,
    device: Optional[str] = None,
    seed: int = DEFAULT_ATTACK_SEED,
    app=None,
) -> ExperimentResult:
    """
    單一 (image × attack config) 實驗。

    victims：單一 EmbeddingModel，或 {role_or_name: model}。
    config.extra / target_* 可傳入，供未來 targeted 攻擊使用。
    """
    victim_map = _normalize_victims(victims)
    if config is None:
        cfg = AttackConfig(name=attack, eps=float(eps), steps=steps, seed=seed)
    else:
        cfg = AttackConfig(
            name=config.name or attack,
            eps=float(config.eps if config.eps is not None else eps),
            steps=config.steps if config.steps is not None else steps,
            seed=int(config.seed if config.seed is not None else seed),
            target_image=config.target_image,
            target_embedding=config.target_embedding,
            target_identity=config.target_identity,
            extra=dict(config.extra),
        )

    apply_attack_seed(int(cfg.seed))
    face_app = app if app is not None else getattr(surrogate, "app", None)
    t0 = time.perf_counter()
    output = apply_attack(
        original_bgr,
        attack=attack,
        app=face_app,
        config=cfg,
        device=device,
    )
    models: dict[str, EmbeddingModel] = {"surrogate": surrogate, **victim_map}
    evals = evaluate_models(original_bgr, output.adversarial_bgr, models)
    pert = perturbation_metrics(original_bgr, output.adversarial_bgr)
    runtime = time.perf_counter() - t0

    sur = evals["surrogate"]
    vic = {k: v for k, v in evals.items() if k != "surrogate"}
    err = sur.error
    if not err:
        for vr in vic.values():
            if vr.error:
                err = vr.error
                break

    return ExperimentResult(
        attack_name=output.name,
        attack_parameters=dict(output.parameters),
        surrogate=sur,
        victims=vic,
        perturbation=pert,
        runtime_sec=runtime,
        original_bgr=original_bgr,
        adversarial_bgr=output.adversarial_bgr,
        error=err,
    )
