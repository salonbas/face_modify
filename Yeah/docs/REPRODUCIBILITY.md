# Reproducibility and asset contract

This repository versions the research definition and formal evidence. Third-party
dataset and pretrained-model binaries are not redistributed directly with this
Git repository.

Large third-party research assets are not versioned in this repository.
Run `python3 scripts/setup_research_assets.py --accept-upstream-research-terms`
after cloning to install required LFW and pretrained model assets.

## Repository does NOT include

- LFW funneled raw images.
- InsightFace `buffalo_l` model files, including:
  - ArcFace recognition model;
  - detector;
  - 68-point landmark model;
  - other runtime-required ONNX files.
- FaceNet VGGFace2 weight.

These third-party dataset / pretrained model binaries are not redistributed
directly with this Git repository.

## How to install required research assets

In a new clone, after creating the Python environment and installing project
dependencies, run from `Yeah/`:

```bash
python3 scripts/setup_research_assets.py --accept-upstream-research-terms
```

The script downloads required assets from their original upstream sources,
installs them at the repository's expected fixed paths, and verifies required
checksums / integrity. No manual file hunting or cache copying is required.

Then verify the environment:

```bash
python3 scripts/check_research_environment.py
```

Minimal new-machine flow:

```bash
git clone <repo>
cd face_modify/Yeah

# Create a Python environment and install dependencies first.
python3 scripts/setup_research_assets.py --accept-upstream-research-terms
python3 scripts/check_research_environment.py
```

## Versioned in ordinary Git

- Source, tests, configuration and experiment scripts.
- LFW manifest, official pair definitions, and frozen splits.
- Calibration configuration, score/ROC evidence and threshold metadata.
- Formal frozen-20 four-method results, compact mechanism evidence, reports,
  protocol, status, and research log.

`.venv`, global and project caches, downloaded model binaries, raw LFW images,
embeddings and disposable run output are not versioned.

## Fixed runtime paths and setup

After reviewing the upstream terms, run this once from `Yeah/`:

```bash
python3 scripts/setup_research_assets.py --accept-upstream-research-terms
python3 scripts/check_research_environment.py
```

The setup script uses official sources only, never copies a cache, and verifies
all model SHA-256 values. It installs these Git-ignored files:

| Asset | Canonical path | Status |
| --- | --- | --- |
| LFW funneled images (13,233 JPEGs) | `data/datasets/lfw/images` | restricted / no recorded redistribution grant |
| buffalo_l ArcFace recognition | `models/insightface/buffalo_l/w600k_r50.onnx` | restricted: non-commercial research only |
| buffalo_l detector and landmarks | `models/insightface/buffalo_l/{det_10g,1k3d68,2d106det,genderage}.onnx` | restricted: same package terms |
| FaceNet VGGFace2 | `models/facenet/20180402-114759-vggface2.pt` | unclear: code is MIT, but no separate weight redistribution grant was found |

The runtime prefers those repo-local locations. Existing workspaces retain a
fallback to `~/.insightface` and the former `.cache/torch` FaceNet path.

## Clean-clone status

A clone contains every versioned research artifact. An internet-connected user
must run the one setup command above and accept the upstream research terms.
No manual locating or copying of LFW, InsightFace, or FaceNet files is needed.
