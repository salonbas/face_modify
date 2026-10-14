"""Benchmark experiment grid。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from advface.attacks.registry import get_attack_spec, normalize_attack_name
from advface.benchmark.dataset import ManifestRow
from advface.config import DEFAULT_BENCHMARK_ATTACKS, DEFAULT_BENCHMARK_EPS, DEFAULT_BENCHMARK_PGD_STEPS


@dataclass(frozen=True)
class ExperimentUnit:
    image_id: str
    image_path: str
    identity_id: str
    source: str
    attack: str
    eps: float
    steps: int

    @property
    def key(self) -> str:
        return unit_key(self.image_id, self.attack, self.eps, self.steps)


def unit_key(image_id: str, attack: str, eps: float, steps: int) -> str:
    return f"{image_id}|{attack}|{eps:.3f}|{int(steps)}"


def _norm_attack(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    if key in ("pgd", "pgd_crop"):
        raise ValueError("benchmark 不包含 crop PGD；請使用 pgd_full")
    return normalize_attack_name(name)


def generate_experiment_grid(
    images: Sequence[ManifestRow],
    *,
    attacks: Sequence[str] | None = None,
    eps_list: Sequence[float] | None = None,
    pgd_steps: Sequence[int] | None = None,
) -> list[ExperimentUnit]:
    attacks_n = [_norm_attack(a) for a in (attacks or DEFAULT_BENCHMARK_ATTACKS)]
    eps_n = [float(e) for e in (eps_list or DEFAULT_BENCHMARK_EPS)]
    steps_n = [int(s) for s in (pgd_steps or DEFAULT_BENCHMARK_PGD_STEPS)]
    if not images:
        raise ValueError("grid 沒有圖片")
    if not attacks_n:
        raise ValueError("grid 沒有 attack")
    if not eps_n:
        raise ValueError("grid 沒有 epsilon")
    if any(get_attack_spec(a).requires_steps for a in attacks_n) and not steps_n:
        raise ValueError("需要 steps 的 attack 請提供 --pgd-steps")

    units: list[ExperimentUnit] = []
    seen: set[str] = set()
    for img in images:
        for attack in attacks_n:
            spec = get_attack_spec(attack)
            step_values = list(steps_n) if spec.requires_steps else [int(spec.default_steps)]
            for eps in eps_n:
                for steps in step_values:
                    u = ExperimentUnit(
                        image_id=img.image_id,
                        image_path=str(img.resolved_path()),
                        identity_id=img.identity_id,
                        source=img.source,
                        attack=attack,
                        eps=float(eps),
                        steps=int(steps),
                    )
                    if u.key in seen:
                        raise ValueError(f"重複 experiment unit：{u.key}")
                    seen.add(u.key)
                    units.append(u)
    return units


def find_duplicate_keys(units: Iterable[ExperimentUnit]) -> list[str]:
    seen: set[str] = set()
    dups: list[str] = []
    for u in units:
        if u.key in seen:
            dups.append(u.key)
        seen.add(u.key)
    return dups
