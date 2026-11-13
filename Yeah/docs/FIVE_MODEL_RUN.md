# Development 100 five-model run

Run from `Yeah/` after `python scripts/run_lfw_five_model_benchmark.py --preflight` reports `READY`:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 .venv/bin/python scripts/run_lfw_five_model_benchmark.py --execute
```

Ctrl+C stops after the current process exits; repeat the same command to resume. A saved PNG with absent or mismatched metadata causes a hard error. Restore that item from backup or move both files aside before deliberately regenerating it. No existing valid image is overwritten.

Recompute all five victim scores and summaries from saved PNGs, without running attacks:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 .venv/bin/python scripts/run_lfw_five_model_benchmark.py --verify --reevaluate
```

`--verify` alone checks saved checksums, provenance, scores and summaries without model inference. For a small engineering run, append `--identity NAME`; its output remains under `formal/`, so back it up or remove the engineering run before the 100-person run. The formal output is `results/transfer/lfw_dev_100_five_model_v1/formal/`. Back up the entire directory, the Development manifest and LFW input images, the five exact checkpoints, pinned author source files and Calibration v2 thresholds. Generated PNGs are the irreplaceable time-intensive output.

Automatic public assets: `python scripts/setup_research_assets.py --accept-upstream-research-terms --models-only` and `bash scripts/setup_five_model_sources.sh`. The AdaFace and MagFace author weights require manual acquisition from the links in `models/checkpoint_manifest.json`, at its documented model paths. Run `python scripts/setup_research_assets.py --verify-only --models-only` to check their recorded SHA-256 values. Generate missing clean eligibility with `python scripts/check_lfw_five_model_clean_eligibility.py` after all assets are present. The author-hosted weight hashes identify the previously frozen local files; they are not independent upstream attestations.

After each run, copy `formal/` with its relative paths to durable storage and
record the backup location and date. From
`results/transfer/lfw_dev_100_five_model_v1/`, run
`find formal -type f -print0 | sort -z | xargs -0 shasum -a 256 > formal_sha256.txt`
and commit that list with the small JSON/CSV results. Restore the whole directory
from backup and run `--verify` before resuming; hashes alone cannot recover PNGs.
Back up the ignored checkpoints and LFW images as well; their Git-tracked
manifests provide the expected hashes.
