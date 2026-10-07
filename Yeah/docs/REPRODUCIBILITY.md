# Reproducibility and asset contract

This repository version-controls the research definition and formal evidence.
It intentionally does **not** redistribute raw LFW images or third-party model
weights until their upstream terms and this GitHub repository's visibility have
been reviewed. A fresh clone contains all code, splits, calibration and formal
results, but not the runtime dataset or model binaries listed below.

Run this read-only check from `Yeah/` after setup:

```bash
python scripts/check_research_environment.py
```

It reports a Git-LFS pointer as missing rather than treating its small pointer
file as a usable model.

## Versioned in ordinary Git

- Source, tests, configuration and experiment scripts.
- LFW manifest, official pair definitions, and frozen splits, including
  `data/datasets/lfw/evaluation_splits/attack_dev.csv`.
- ArcFace and FaceNet calibration configuration, scores, ROC data and
  threshold metadata under `results/calibration/`.
- Formal frozen-20 four-method summary, per-identity records, provenance,
  compact mechanism evidence and curated report figures.
- `RESEARCH_STATUS.md`, `RESEARCH_LOG.md`, this document, and the formal
  experiment protocol.

Formal CSV, JSON, HTML, and small report images are deliberately ordinary Git.
The repository does not blanket-track PNG/JPEG files.

## Git LFS policy

The root `.gitattributes` reserves Git LFS for reviewed binary assets:
`*.onnx`, `*.pth`, `*.pt`, and `*.npz`. There are currently **no LFS objects
committed**: no binary has been admitted without a licence and visibility
decision. If an asset is approved, place it in its documented canonical path,
verify its checksum, use `git lfs track`/`git add` for that specific file, and
never add a cache directory wholesale.

## Runtime assets intentionally outside this repository

| Asset | Expected path | Source / identifier | SHA-256 audited locally |
| --- | --- | --- | --- |
| LFW funneled raw images | `data/datasets/lfw/images` (symlink or directory) | [UMass LFW](http://vis-www.cs.umass.edu/lfw/), `lfw_funneled`, 13,233 JPEGs | Dataset tree; verify count rather than a single archive hash |
| ArcFace recognition | `~/.insightface/models/buffalo_l/w600k_r50.onnx` | InsightFace `buffalo_l` | `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` |
| InsightFace detector | `~/.insightface/models/buffalo_l/det_10g.onnx` | InsightFace `buffalo_l` | `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` |
| InsightFace 68-point landmark model | `~/.insightface/models/buffalo_l/1k3d68.onnx` | InsightFace `buffalo_l` | `df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc` |
| FaceNet VGGFace2 | `.cache/torch/checkpoints/20180402-114759-vggface2.pt` | `facenet-pytorch` pretrained `vggface2` | `281cebca8662831adb987a874bdcb36e73f5b1c6dc5ee5878f305e985625d99b` |

`FaceAnalysis(name="buffalo_l")` may load additional package components
(`2d106det.onnx`, `genderage.onnx`) depending on the installed InsightFace
version. Obtain the complete upstream `buffalo_l` package, not a model-cache
snapshot from another machine.

The local audit found LFW at
`~/scikit_learn_data/lfw_home/lfw_funneled` (13,233 JPEGs, 239.31 MiB) and a
complete `buffalo_l` package under `~/.insightface/models/buffalo_l` (about
326 MiB). They are excluded pending an explicit redistribution and repository
visibility decision; they must not be pushed merely because LFS is available.

## Intentionally ignored

- Virtual environments, Python bytecode and test/tool caches.
- Torch, Hugging Face, pip and perceptual-model caches.
- Raw LFW/cache links, copied benchmark images, generated embeddings,
  serialized adversarial images, trajectory checkpoints, disposable tensors,
  debug output, IDE/OS junk and local credentials.

## Fresh-clone status

After `git clone`, the frozen research definition and all formal results are
available. To rerun evaluation or attacks, a collaborator must independently
obtain the LFW funneled dataset and the upstream InsightFace/FaceNet weights,
place them at the canonical paths, and rerun the checker. Thus this is not yet
a self-contained `git clone` + `git lfs pull` runtime reproduction; the
remaining blocker is a documented licence/visibility approval for LFW and the
three upstream model families.
