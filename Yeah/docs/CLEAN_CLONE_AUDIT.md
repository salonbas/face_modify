# Clean clone validation log — 2026-10-29

Verdict: **FAIL / partially reproducible**. This was a real local Git clone into a new `/tmp/yeah-clean-clone-9nkSX9/repo` directory from commit `bdf50b10bf9e6c5166d26268c9b089e1002672f3`, with a separately created Python 3.12 venv. It was not a copy of the installed project tree. The clone was of the repository root (`face_modify_yeah`), with research code under `Yeah/`. The `/tmp` test directory was transient; this log records observed commands and exit states, not a preserved environment.

| Step | Actual command / observation | Result |
| --- | --- | --- |
| Clone | `git clone --local --no-hardlinks /home/ssilthorin/projects/face_modify_yeah /tmp/yeah-clean-clone-9nkSX9/repo` | PASS |
| New environment | `python3.12 -m venv /tmp/yeah-clean-clone-9nkSX9/venv`; pip 24.0 initially | PASS |
| Install from committed requirements | `venv/bin/python -m pip install -r repo/Yeah/requirements.txt` | FAIL: `facenet-pytorch==2.6.0` requires NumPy <2, while direct pin is NumPy 2.4.4 |
| Revised installer candidate | Ran working-tree `scripts/install_research_dependencies.sh` against the isolated venv with `requirements-constraints.txt` | PASS for installation and import (`numpy 2.4.4`, `torch 2.12.0+cu130`); this script is not in the tested clone commit, so it is **not** a clean-clone PASS |
| Committed environment checker | `venv/bin/python repo/Yeah/scripts/check_research_environment.py` before any asset setup | FAIL as expected: no LFW JPEGs, ArcFace, detector, landmark or FaceNet weight. V1 calibrations and v3 summary existed |
| Committed setup script | `venv/bin/python repo/Yeah/scripts/setup_research_assets.py --verify-only` | FAIL: option absent in committed version; current working-tree version adds it |
| Five-model assets and v2 split | Current `models/checkpoint_manifest.json`, author source/weights, v2 splits and v2 calibration are uncommitted or ignored | FAIL: absent from clean clone |
| End-to-end smoke in clone | Could not run without assets and v2 artifacts | NOT RUN |

Current working tree checks are separate evidence: `setup_research_assets.py --verify-only` passed all five checkpoint hashes and 13,233 per-image LFW SHA-256 values; `check_research_environment.py` passed; `smoke_lfw_five_model_pgd_mask.py --max-identities 1` resumed one existing identity and passed; full pytest passed 66 tests. These checks used locally preinstalled assets and must not be presented as clean-clone success.

Further blockers after committing source and small artifacts: AdaFace and MagFace author weights are hosted on Google Drive with no recorded independent publisher SHA-256, and acquisition may require manual interaction or terms acceptance. Perceptual-network weights and some serialized adversarial PNGs are only in ignored local caches/artifact directories. Exact formal result regeneration therefore remains unverified. No 100-person benchmark or OOS attack was run in this audit.
