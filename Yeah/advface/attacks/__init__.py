"""攻擊模組：gaussian / fgsm / pgd / pgd_full / mi_fgsm。"""

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
