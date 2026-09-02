"""
攻擊登錄：新增方法時在此註冊，不必改 runner / benchmark 的 if/elif。

AttackConfig.extra / target_* 預留給未來 targeted / transform 攻擊；
目前實作可忽略。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from advface.config import PIXEL_MAX


@dataclass
class AttackConfig:
    name: str
    eps: float
    steps: Optional[int] = None
    alpha: Optional[float] = None
    momentum: float = 1.0
    seed: int = 0
    target_image: Optional[np.ndarray] = None
    target_embedding: Optional[np.ndarray] = None
    target_identity: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackOutput:
    adversarial_bgr: np.ndarray
    name: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class AttackSpec:
    name: str
    apply: Callable[..., AttackOutput]
    requires_steps: bool = False
    default_steps: int = 1
    linf_constraint_domain: str = "full_image"
    aliases: tuple[str, ...] = ()


def _apply_fgsm(img_bgr, *, app, config: AttackConfig, device=None) -> AttackOutput:
    from advface.attacks.fgsm import run_fgsm

    results = run_fgsm(img_bgr, [float(config.eps)], app, device=device)
    r = results[0]
    return AttackOutput(
        adversarial_bgr=r.attacked_bgr,
        name="fgsm",
        parameters={"eps": float(config.eps), "steps": 1, "eps_255": float(r.eps_255)},
    )


def _apply_pgd(img_bgr, *, app, config: AttackConfig, device=None) -> AttackOutput:
    from advface.attacks.pgd import run_pgd

    steps = int(config.steps if config.steps is not None else 20)
    results = run_pgd(img_bgr, [float(config.eps)], app, steps=steps, device=device)
    r = results[0]
    return AttackOutput(
        adversarial_bgr=r.attacked_bgr,
        name="pgd",
        parameters={
            "eps": float(config.eps),
            "steps": steps,
            "eps_255": float(r.eps_255),
            "alpha_px": (float(config.eps) * PIXEL_MAX) / max(steps, 1),
        },
    )


def _apply_pgd_full(img_bgr, *, app, config: AttackConfig, device=None) -> AttackOutput:
    from advface.attacks.pgd import run_pgd_full

    steps = int(config.steps if config.steps is not None else 100)
    results = run_pgd_full(img_bgr, [float(config.eps)], app, steps=steps, device=device)
    r = results[0]
    return AttackOutput(
        adversarial_bgr=r.attacked_bgr,
        name="pgd_full",
        parameters={
            "eps": float(config.eps),
            "steps": steps,
            "eps_255": float(r.eps_255),
            "alpha_px": (float(config.eps) * PIXEL_MAX) / max(steps, 1),
            "random_start": False,
            "linf_tensor": r.linf_tensor,
        },
    )


def _apply_mi_fgsm(img_bgr, *, app, config: AttackConfig, device=None) -> AttackOutput:
    from advface.attacks.mi_fgsm import run_mi_fgsm

    steps = int(config.steps if config.steps is not None else 20)
    alpha = float(config.alpha) if config.alpha is not None else float(config.eps) / max(steps, 1)
    run_result = run_mi_fgsm(
        img_bgr, float(config.eps), app, steps=steps, alpha=alpha,
        momentum=float(config.momentum), device=device, return_metadata=True,
    )
    if isinstance(run_result, tuple):
        adversarial, metrics = run_result
    else:  # compatibility with callers/tests replacing the attack function
        adversarial, metrics = run_result, {}
    return AttackOutput(
        adversarial_bgr=adversarial,
        name="mi_fgsm",
        parameters={
            "method": "mi_fgsm", "eps": float(config.eps), "steps": steps,
            "alpha": alpha, "alpha_px": alpha * PIXEL_MAX,
            "eps_255": float(config.eps) * PIXEL_MAX,
            "momentum": float(config.momentum), "decay": float(config.momentum),
            "random_start": False, "gradient_normalization": "per-image L1 norm",
            "seed": int(config.seed),
            **metrics,
        },
    )


ATTACKS: dict[str, AttackSpec] = {
    "fgsm": AttackSpec(
        name="fgsm",
        apply=_apply_fgsm,
        requires_steps=False,
        default_steps=1,
        linf_constraint_domain="aligned_crop_paste",
        aliases=("fgsm",),
    ),
    "pgd": AttackSpec(
        name="pgd",
        apply=_apply_pgd,
        requires_steps=True,
        default_steps=20,
        linf_constraint_domain="aligned_crop_paste",
        aliases=("pgd", "pgd_crop"),
    ),
    "pgd_full": AttackSpec(
        name="pgd_full",
        apply=_apply_pgd_full,
        requires_steps=True,
        default_steps=100,
        linf_constraint_domain="full_image",
        aliases=("pgd_full", "pgd-full", "pgdfull"),
    ),
    "mi_fgsm": AttackSpec(
        name="mi_fgsm", apply=_apply_mi_fgsm, requires_steps=True,
        default_steps=20, linf_constraint_domain="full_image", aliases=("mi_fgsm",),
    ),
}

_ALIAS_TO_NAME: dict[str, str] = {}
for _spec in ATTACKS.values():
    for _a in _spec.aliases:
        _ALIAS_TO_NAME[_a.lower().replace("-", "_")] = _spec.name


def normalize_attack_name(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    if key in ATTACKS:
        return key
    if key in _ALIAS_TO_NAME:
        return _ALIAS_TO_NAME[key]
    known = ", ".join(sorted(ATTACKS))
    raise ValueError(f"未知 attack：{name}（已註冊：{known}）")


def get_attack_spec(name: str) -> AttackSpec:
    return ATTACKS[normalize_attack_name(name)]


def register_attack(spec: AttackSpec) -> None:
    """未來新增攻擊：實作 apply 後呼叫此函式（或直接寫入 ATTACKS）。"""
    ATTACKS[spec.name] = spec
    for a in spec.aliases or (spec.name,):
        _ALIAS_TO_NAME[a.lower().replace("-", "_")] = spec.name


def apply_attack(
    img_bgr: np.ndarray,
    *,
    attack: str,
    app: Any,
    config: AttackConfig,
    device: Optional[str] = None,
) -> AttackOutput:
    spec = get_attack_spec(attack)
    cfg = AttackConfig(
        name=spec.name,
        eps=float(config.eps),
        steps=config.steps if config.steps is not None else spec.default_steps,
        alpha=config.alpha,
        momentum=float(config.momentum),
        seed=config.seed,
        target_image=config.target_image,
        target_embedding=config.target_embedding,
        target_identity=config.target_identity,
        extra=dict(config.extra),
    )
    return spec.apply(img_bgr, app=app, config=cfg, device=device)
