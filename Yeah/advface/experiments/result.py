"""統一 ExperimentResult：single-image 與 batch 共用。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from advface.evaluation.transfer import TransferEvalResult


@dataclass
class ExperimentResult:
    attack_name: str
    attack_parameters: dict[str, Any]
    surrogate: TransferEvalResult
    victims: dict[str, TransferEvalResult]
    perturbation: dict[str, Any]
    runtime_sec: float
    original_bgr: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    adversarial_bgr: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    error: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def primary_victim(self) -> Optional[TransferEvalResult]:
        if not self.victims:
            return None
        if "victim" in self.victims:
            return self.victims["victim"]
        return next(iter(self.victims.values()))

    def to_metrics_dict(self) -> dict[str, Any]:
        victims = {k: v.to_metrics_dict() for k, v in self.victims.items()}
        primary = self.primary_victim
        return {
            "attack": {"name": self.attack_name, **self.attack_parameters},
            "surrogate": self.surrogate.to_metrics_dict(),
            "victims": victims,
            "victim": primary.to_metrics_dict() if primary is not None else None,
            "perturbation": dict(self.perturbation),
            "runtime_sec": self.runtime_sec,
            "error": self.error,
        }
