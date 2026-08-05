"""Deprecated：請改用 scripts/run_gaussian.py。"""
from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

warnings.warn(
    "scripts/attack_gaussian.py 已改名為 scripts/run_gaussian.py，請改用新入口。",
    DeprecationWarning,
    stacklevel=2,
)
print("[DEPRECATED] scripts/attack_gaussian.py → 請改用 scripts/run_gaussian.py", file=sys.stderr)

_target = Path(__file__).with_name("run_gaussian.py")
_spec = importlib.util.spec_from_file_location("advface_run_gaussian", _target)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

if __name__ == "__main__":
    raise SystemExit(_mod.main())
