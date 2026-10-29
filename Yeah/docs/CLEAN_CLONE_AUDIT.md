# GitHub clean-clone validation — 2026-10-29

**Verdict: NOT YET REPRODUCIBLE.** The final test cloned pushed `origin/main` commit `23abf6b039e4c5fd8fa505733f1518e78a8d3b51` from `git@github.com:salonbas/face_modify.git` into `/tmp/yeah-github-validated-20261029`. It created a new Python 3.12 venv at `/tmp/yeah-github-validated-venv`. The clone had no LFW image tree or AdaFace/MagFace checkpoints before setup. No original repository model, image, or private cache path was used for research assets. The temporary test directories are not preserved evidence; this file records the commands and outcomes.

| Step | Command / evidence | Result |
| --- | --- | --- |
| Clone | `git clone git@github.com:salonbas/face_modify.git /tmp/yeah-github-validated-20261029`; `git rev-parse HEAD` = `23abf6b...` | PASS |
| Install | `python3.12 -m venv /tmp/yeah-github-validated-venv`; `bash scripts/install_research_dependencies.sh /tmp/yeah-github-validated-venv/bin/python` from clone `Yeah/` | PASS; NumPy 1.26.4, PyTorch 2.2.2+cpu, `pip check`: no broken requirements; FaceNet and other required imports pass |
| Author source | `bash scripts/setup_five_model_sources.sh` | PASS; AdaFace/MagFace source commits fetched and pinned source file hashes verified by setup/checker |
| Automatic assets | `python scripts/setup_research_assets.py --accept-upstream-research-terms` | PARTIAL by design: LFW, buffalo_l, FaceNet, buffalo_sc downloaded; then exit 1 with clear missing AdaFace checkpoint. Figshare returned HTTP 403; checksum-identical Hugging Face mirror supplied LFW. |
| Asset verification | Direct calls to `verify_lfw`, buffalo_l and FaceNet `verify`, MobileFaceNet recognizer/detector hash checks, and InsightFace runtime-layout checks | PASS for 13,233 LFW JPEGs, all auto-downloaded model files and their recorded SHA-256 values |
| Model loading | `load_embedder` for `arcface`, `facenet`, `mobilefacenet` inside clone | PASS; AdaFace/MagFace loader cannot be exercised there without their author weights |
| Environment check | `python scripts/check_research_environment.py` | FAIL only for missing AdaFace and MagFace weights; v1/v2 calibration, Development 100, OOS 228, formal v3 summary, LFW, other weights and sources pass |
| Specs and Git artifacts | `python scripts/generate_research_specs.py --check`; compare SHA-256 of manifest, v2 splits/audit, five threshold JSONs and v3 config between original and clone | PASS; exact bytes match |
| Entrypoints | `--help` for setup, v2 calibration, v3 two-phase runner, five-model benchmark gate, one-person smoke and Momentum ablation | PASS (6) |
| Tests | `python -m pytest -q` in clone | 65 passed, 1 skipped because the required author weights are absent |
| Five-model 200-step smoke | Requires both author weights; no OOS attack or 100-person run was attempted | NOT RUN; cannot claim five-model end-to-end or output-tolerance PASS |

The original worktree's fresh dependency venv separately passed 66 tests, all five checkpoint hashes and model loaders, and an existing one-person smoke record was readable. That smoke resumed a local PNG and did not regenerate 200 attack steps; it is not clean-clone proof. Formal v3 results were not modified. The frozen v3 config, v2 splits and threshold JSONs are byte-identical across the worktree and GitHub clone, but numerical reproduction of a new five-model attack remains unverified.

## Remaining work

1. Open the AdaFace and MagFace author Google Drive links in `models/checkpoint_manifest.json`, review their terms, download the exact checkpoints to the documented paths, and run `python scripts/setup_research_assets.py --verify-only`. No author-published independent weight SHA-256 was found; the manifest records hashes of the frozen local copies.
2. Run `python scripts/check_research_environment.py` and `python scripts/smoke_lfw_five_model_pgd_mask.py --max-identities 1` in a clean clone with those manually acquired weights, with a fresh output directory. Verify 200 steps, five model scores, SSIM/LPIPS/DISTS/L∞, threshold crossing and numerical tolerances against the recorded smoke. Do not read or attack OOS.
3. Back up the ignored formal PNG bytes privately before any cleanup. `docs/FORMAL_PNG_SHA256.txt` proves integrity but cannot recover the images. Record perceptual-network cache hashes if bitwise metric reproduction is required.

The five-model 100-person benchmark remains behind its fail-closed timing/executor gate. Its two-person engineering smoke and Momentum ablation are separate from formal v3 evidence.
