# Five-model formal benchmark operations

Run on WSL2 Ubuntu with CPU PyTorch. `configs/lfw_dev_100_v2.json` freezes the
Development 100 split and 200-step epsilon 0.04 attack. The formal entry is
`scripts/run_lfw_five_model_benchmark.py`. Run `--preflight` first; `--execute`
refuses to start unless every preflight gate passes, including dependency
compatibility and offline perceptual-model cache hashes. On 2026-11-13, the
macOS CPU host passed a one-person four-method real attack, all five victim
evaluations, serialized image checksum, resume and saved-PNG reevaluation.
This is an executor acceptance test, not a 100-person attack result.
Compact checksums, quality metrics and environment evidence are in
`results/transfer/lfw_dev_100_five_model_v1/acceptance_2026_11_13.json`;
the sample PNGs and full records remain local under `results/smoke/`.
The 2026-11-13 clean eligibility recomputation, stored with per-identity
scores, checkpoint hashes and package versions in `clean_eligibility.json`,
found 83/100 jointly eligible. FaceNet alone is 96/100 in this run versus
95/100 in the 2026-10-29 report; the four current FaceNet ineligible identities
are Eric Hinske, Julianna Margulies, Phil Gramm and Richard Myers. The frozen
threshold was not changed. Keep the older report as historical evidence and
use the JSON rows for current-run denominators.

From `Yeah/`, a full run is `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv/bin/python -u scripts/run_lfw_five_model_benchmark.py --execute`.
Re-run the same command to resume. Use `--verify` to check saved checksums and
score provenance, or `--verify --reevaluate` to recompute all five victim
scores and SSIM/LPIPS/DISTS from saved PNGs without generating attacks.
The run prints completed/total and ETA, and writes `formal/run.log`.

A safe future run must use the artifact contract in
`advface/attack_artifact.py`: write B_adv and complete metadata, validate both,
then skip generation on restart. Evaluate the fixed PNGs separately with current
model checkpoints and calibration thresholds. Keep evaluation records keyed by
victim checkpoint and threshold hashes; never reuse an old ASR summary after
threshold changes. Transfer large PNGs through private backed-up storage and
verify hashes after copying. Never infer completion from file existence.

Historical four-method v3 results remain under
`results/transfer/lfw_attack_dev_20_four_method_v3/` and must not be overwritten.
The failed MPS investigation is preserved in `docs/MPS_ARCFACE_PROBE_2026-10-29.md`;
its one-off probe script was removed from the active scripts directory. Generic
CPU/CUDA device selection in attack implementations is retained.
