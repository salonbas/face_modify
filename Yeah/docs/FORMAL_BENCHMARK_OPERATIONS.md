# Five-model formal benchmark operations (pending executor validation)

Run on WSL2 Ubuntu with CPU PyTorch. `configs/lfw_dev_100_v2.json` freezes the
Development 100 split and 200-step epsilon 0.04 attack. The formal entry is
`scripts/run_lfw_five_model_benchmark.py`. Run `--preflight` first; `--execute`
currently refuses to start. Do not run the full benchmark until a validated
resume/evaluation executor, one-person WSL profile, bounded ETA, all five
checkpoint hashes, clean eligibility, and the explicit user go-ahead exist.

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
