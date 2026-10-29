# Reproducibility and asset contract

This repository versions the research definition and formal evidence. Third-party
dataset and pretrained-model binaries are not redistributed directly with this
Git repository.

Large third-party research assets are not versioned in this repository. Asset
setup is partly manual for author-hosted AdaFace and MagFace weights; a clean
clone is not presently a one-command five-model reproduction.

## Repository does NOT include

- LFW funneled raw images.
- InsightFace `buffalo_l` model files, including:
  - ArcFace recognition model;
  - detector;
  - 68-point landmark model;
  - other runtime-required ONNX files.
- FaceNet VGGFace2 weight.
- MobileFaceNet `buffalo_sc` weight.
- AdaFace R50 and MagFace iResNet100 author weights and source modules.

These third-party dataset / pretrained model binaries are not redistributed
directly with this Git repository.

## How to install required research assets

In a new clone, after creating the Python environment and installing project
dependencies, run from `Yeah/`:

```bash
python3 scripts/setup_research_assets.py --accept-upstream-research-terms
```

The script downloads LFW, InsightFace `buffalo_l` and `buffalo_sc`, and FaceNet
from the URLs in the script. For LFW it tries scikit-learn's Figshare URL first;
if Figshare returns 403, it uses the public Hugging Face LFW mirror. Both must
match scikit-learn's published archive SHA-256
`b47c8422c8cded889dc5a13418c4bc2abbda121092b3533a83306f90d900100a`,
then the repository verifies all 13,233 JPEGs individually. It checks recorded
model SHA-256 values. The AdaFace and
MagFace checkpoint URLs are in `models/checkpoint_manifest.json` and currently
lead to author Google Drive pages. Open each `source` URL in a browser, review
the author's terms, complete any required login or download confirmation, and
save the downloaded files under the exact paths below (create parent directories
if needed):

1. AdaFace → `models/adaface/adaface_ir50_ms1mv2.ckpt`.
2. MagFace → `models/magface/magface_iresnet100_ms1mv2.pth`.

Run `sha256sum` on both files and compare the output with the corresponding
`sha256` entries in `models/checkpoint_manifest.json`. A mismatch means the
checkpoint is a different version; stop rather than recalibrating silently.
Then run
`bash scripts/setup_five_model_sources.sh` to fetch pinned author source code.
`setup_research_assets.py --verify-only` checks all five local checkpoint hashes
without downloading. A matching hash proves equivalence to the recorded local
artifact; no independent author-published digest was located for AdaFace or
MagFace.

Then verify the environment:

```bash
python3 scripts/check_research_environment.py
```

Minimal new-machine flow:

```bash
git clone git@github.com:salonbas/face_modify.git
cd face_modify/Yeah
python3.12 -m venv .venv
bash scripts/install_research_dependencies.sh
# Review upstream dataset/model terms and obtain the two author-hosted files above.
bash scripts/setup_five_model_sources.sh
.venv/bin/python scripts/setup_research_assets.py --accept-upstream-research-terms
.venv/bin/python scripts/check_research_environment.py
.venv/bin/python scripts/generate_research_specs.py --check
.venv/bin/python -m pytest -q
```

`requirements.txt` pins direct dependencies and
`requirements-constraints.txt` fixes the transitive Python packages. The
installer first installs CPU-only PyTorch 2.2.2, then resolves FaceNet 2.6.0
normally with NumPy 1.26.4 and Pillow 10.2.0, and runs `pip check`. These pins
avoid overriding FaceNet's declared dependency bounds. Cross-platform wheel
availability and native builds remain to be verified outside this Linux host.

## Versioned in ordinary Git

- Source, tests, configuration and experiment scripts.
- LFW manifest, official pair definitions, and frozen splits.
- Calibration configuration, score/ROC evidence and threshold metadata.
- Formal frozen-20 four-method results, compact mechanism evidence, reports,
  protocol, status, and research log.

`.venv`, global and project caches, downloaded model binaries, raw LFW images,
embeddings and disposable run output are not versioned.
`data/datasets/lfw/images_sha256.txt` freezes the expected per-image LFW bytes;
asset setup and environment check hash every JPEG against it.

## Fixed runtime paths and setup

After reviewing the upstream terms, run this once from `Yeah/`:

```bash
python3 scripts/setup_research_assets.py --accept-upstream-research-terms
python3 scripts/check_research_environment.py
```

The setup script uses recorded upstream sources, never copies a user's cache,
and verifies model SHA-256 values. It installs or checks these Git-ignored files:

| Asset | Canonical path | Status |
| --- | --- | --- |
| LFW funneled images (13,233 JPEGs) | `data/datasets/lfw/images` | restricted / no recorded redistribution grant |
| buffalo_l ArcFace recognition | `models/insightface/buffalo_l/w600k_r50.onnx` | restricted: non-commercial research only |
| buffalo_l detector and landmarks | `models/insightface/buffalo_l/{det_10g,1k3d68,2d106det,genderage}.onnx` | restricted: same package terms |
| FaceNet VGGFace2 | `models/facenet/20180402-114759-vggface2.pt` | unclear: code is MIT, but no separate weight redistribution grant was found |
| MobileFaceNet | `models/insightface/buffalo_sc/w600k_mbf.onnx` | InsightFace noncommercial research terms; automatic download |
| AdaFace R50 | `models/adaface/adaface_ir50_ms1mv2.ckpt` | author Google Drive; manual acquisition and separate weight terms need review |
| MagFace iResNet100 | `models/magface/magface_iresnet100_ms1mv2.pth` | author Google Drive; manual acquisition and separate weight terms need review |

The runtime prefers those repo-local locations. Existing workspaces retain a
fallback to `~/.insightface` and the former `.cache/torch` FaceNet path.

## Clean-clone status

A clean clone of committed `origin/main` does not include the current uncommitted
v2 calibration/split and five-model code. Even after those are committed, the
manual author-weight acquisition, unrecorded perceptual-model cache hashes and
untracked attack PNGs prevent claiming full bitwise reproduction. See
`docs/ARTIFACT_INVENTORY.md` and the latest clean-clone test evidence before
claiming a pass.
