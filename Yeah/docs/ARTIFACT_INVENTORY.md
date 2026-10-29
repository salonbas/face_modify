# Research artifact inventory

Audited 2026-10-29. Do not delete the ignored paths below: some are the only local copies of serialized attack images and diagnostic outputs. `git ls-files` and `git status --ignored --short` are the inventory commands. The current worktree also has uncommitted source, v2 splits/calibration and Momentum records; until committed and pushed, they are local-only.

| Class | Current Git status | Rebuild / backup contract |
| --- | --- | --- |
| LFW manifest, official pair CSVs, v1 splits | Tracked; image checksum manifest new/untracked | Frozen inputs; per-image SHA-256 in `data/datasets/lfw/images_sha256.txt` |
| V2 Development 100 / OOS 228 split and audit | Untracked, not ignored | Commit exact CSV/JSON after consistency review |
| V1 ArcFace/FaceNet calibration JSON/CSV | Tracked | Frozen authority for v3; no recalibration to rewrite results |
| V2 five-model calibration JSON/CSV | Untracked, not ignored | Commit thresholds, metadata, pair scores; exclude embedding arrays |
| Model provenance/checksums | `models/checkpoint_manifest.json` untracked, not ignored | Commit manifest; weights and author source copies remain ignored; re-download or reacquire with terms |
| V3 formal 20-person summary, per-identity CSV and JSON records | Tracked | Primary frozen numerical evidence; keep their source artifact hashes |
| Momentum ablation summary/records | Untracked, not ignored | Commit small JSON/CSV after review; do not merge into v3 |
| Five-model clean eligibility/preflight JSON | Mixed: preflight untracked, clean eligibility ignored | Keep preflight as gate evidence; back up clean eligibility JSON separately or regenerate from exact checkpoints and 100 clean pairs; it is not a formal attack result |
| LFW raw images, model weights, author code, embedding arrays, caches | Ignored | Reacquire from documented upstream; verify hashes; never commit third-party binaries without license review |
| V3/v2 formal adversarial PNGs | Ignored; 80 local copies | `docs/FORMAL_PNG_SHA256.txt` records all 80 current hashes; `sha256sum -c docs/FORMAL_PNG_SHA256.txt` passed locally. Back up the bytes privately before cleanup; a hash alone cannot restore them. Regeneration may depend on runtime versions |
| Historical v1 raw PNGs and other ignored images | Ignored; local copies only | Back up privately with checksums if exact visual/perceptual re-evaluation is required; formal summary values remain tracked |
| Perceptual and robustness result directories; diagnostic trajectories/checkpoints | Ignored | Historical evidence and diagnostics; private backup recommended before cleanup. Rebuild only with the original protocol, weights and images; do not overwrite the sole copy |
| Report figures and curated analysis JSON/CSV | Mostly tracked under `docs/reports` | Keep curated figures and their source commands; historical report PDF may have separate rights |

Before a new machine is considered reproducible, compare the clone's `git ls-files` to this inventory; install assets with the manifest; run the environment check, calibration consistency tests, specification `--check`, import/CLI tests and a small non-formal smoke. A formal 100-person benchmark or OOS attack is outside this recovery procedure.
