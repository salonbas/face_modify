# Reproducibility and asset contract

This repository versions the research definition and formal evidence. It does
not redistribute raw LFW images or third-party model weights: the required
assets do not all have a clear public redistribution grant, and InsightFace
explicitly limits its pretrained buffalo_l package to non-commercial research.

## Versioned in ordinary Git

- Source, tests, configuration and experiment scripts.
- LFW manifest, official pair definitions, and frozen splits.
- Calibration configuration, score/ROC evidence and threshold metadata.
- Formal frozen-20 four-method results, compact mechanism evidence, reports,
  protocol, status, and research log.

`.venv`, global and project caches, downloaded model binaries, raw LFW images,
embeddings and disposable run output are not versioned.

## Git LFS

The root `.gitattributes` reserves LFS for `*.onnx`, `*.pth`, `*.pt`, and
`*.npz`; its LFW JPEG rule is scoped only to
`data/datasets/lfw/images/**/*.jpg`, never to report images. There are no LFS
objects in this revision. The audited machine's `git-lfs` executable is broken,
but that is not the deciding blocker: no required asset currently has recorded
redistribution approval for this repository.

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

A clone contains every ordinary-Git research artifact. It is not a complete
`git clone` + `git lfs pull` runtime reproduction: an internet-connected user
must run the one setup command above and accept the upstream research terms.
No manual locating or copying of LFW, InsightFace, or FaceNet files is needed.
