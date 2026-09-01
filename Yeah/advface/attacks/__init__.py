"""攻擊模組：gaussian / fgsm / pgd / pgd_full。新增攻擊請登錄 registry。"""

from advface.attacks.registry import (
    ATTACKS,
    AttackConfig,
    AttackOutput,
    AttackSpec,
    apply_attack,
    get_attack_spec,
    normalize_attack_name,
    register_attack,
)

__all__ = [
    "ATTACKS",
    "AttackConfig",
    "AttackOutput",
    "AttackSpec",
    "apply_attack",
    "get_attack_spec",
    "normalize_attack_name",
    "register_attack",
]
