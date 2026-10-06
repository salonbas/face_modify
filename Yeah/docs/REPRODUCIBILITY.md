# Reproducibility

Git preserves the source, scripts, tests, experiment protocol/provenance,
LFW manifest and frozen splits (`data/datasets/lfw/`), calibration authority
(`results/calibration/`), formal frozen-20 four-method results
(`results/transfer/lfw_attack_dev_20_four_method_v3/`), compact diagnostic
evidence, reports, and research log/status.

Git does not preserve raw LFW images, large ArcFace/FaceNet weights, caches,
virtual environments, generated embeddings, trajectory checkpoints, or
runtime/debug output.  Put LFW at
`data/datasets/lfw/images` (normally a symlink to the local sklearn LFW
cache); ArcFace `w600k_r50.onnx` is expected under `~/.insightface/models/
buffalo_l/`; FaceNet `20180402-114759-vggface2.pt` is expected at
`.cache/torch/checkpoints/`.

From `Yeah/`, run `python scripts/check_research_environment.py` to check the
versioned authority and optional local assets.  The frozen split,
calibration thresholds, and formal result authority are at the paths above.
