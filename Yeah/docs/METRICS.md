# Metrics specification

Applies to frozen LFW four-method v3 and the proposed five-model Development v2 protocol. Last verified: 2026-10-29 against `advface/evaluation/`, `advface/calibration.py`, `scripts/run_lfw_two_phase_v4.py`, `scripts/evaluate_lfw_perceptual_v2.py`, and `advface/evaluation/multi_victim.py`. Numeric thresholds come from the calibration artifacts, not this document. Do not pool results across protocol versions.

## Verification and attack rates

For an identity, A is the untouched reference image, B is the clean probe, and B_adv is the attacked probe. Model embedding `e(x)` is L2-normalized. Cosine score is `s(X,Y) = e(X)·e(Y) / (||e(X)||₂ ||e(Y)||₂)`. Higher means more similar. Clean score is `s(A,B)`; adversarial score is `s(A,B_adv)`. Cosine drop is `s(A,B) − s(A,B_adv)`; larger positive drop indicates more verification damage. The attack objective uses `s(B,B_adv)` as a self-reference surrogate; this is distinct from the reported A/B verification score.

The calibrated decision is same identity iff `s ≥ τ`. A threshold crossing requires `s(A,B) ≥ τ` **and** `s(A,B_adv) < τ`. Equality is accepted. A failed face detection or clean score below τ is not an attack success. The per-model clean eligible set `E_m` consists of identities with valid clean A/B embeddings and clean score ≥ τ_m. The common eligible set is `E_common = intersection_m E_m`.

| Quantity | Formula / denominator | Direction |
| --- | --- | --- |
| ASR for a specified evaluated set S | `# threshold crossings in S / # attacked identities in S`; report S and n | Higher means stronger dodging |
| Conditional ASR | `# crossings among attacked E_m / # attacked E_m`; implementation `conditional_asr` | Higher means stronger dodging among clean-eligible pairs |
| Common-eligible ASR | `# crossings among attacked E_common / # attacked E_common`; implementation key `paired_asr` | Higher means stronger dodging on a shared five-model denominator |
| Clean eligibility / clean TAR on attack cohort | `# E_m / # original cohort`; implementation `clean_tar` | Higher means more clean pairs accepted; this is not calibration TAR |
| Calibration FAR | `FP / (FP+TN)` on valid impostor pairs | Lower is better |
| Calibration TAR | `TP / (TP+FN)` on valid genuine pairs | Higher is better |

The frozen v3 aggregation reports boundary-crossing proportions over its fixed 20 identities for each method; inspect the source summaries and valid counts before interpreting them as conditional ASR. Five-model v2 must report both model-specific conditional and common-eligible rates; clean failures and missing attacks remain visible in the original denominator. `advface/evaluation/multi_victim.py` validates eligibility and success and reports both denominators. Empty eligible sets yield `null`, never zero.

## Pixel and perceptual metrics

The image comparison is **decoded original B versus decoded serialized B_adv PNG**, never A versus B_adv. Metrics are computed per identity before arithmetic mean aggregation; v1 perceptual evaluation also reports median and sample standard deviation. Do not average a model score with a perceptual score or mix different image preprocessing versions.

| Metric | Computation / dependency | Direction |
| --- | --- | --- |
| SSIM | `skimage.metrics.structural_similarity` on native-resolution RGB B/B_adv converted from uint8 to float [0,1], `channel_axis=2`, `data_range=1.0` | Higher means greater structural similarity; 1 is identical |
| LPIPS | `lpips.LPIPS(net="alex")`; paired RGB images jointly resized so maximum side ≤256, then normalized to [-1,1] | Lower means more perceptually similar |
| DISTS | `DISTS_pytorch.DISTS()` VGG16; same paired resize, RGB float [0,1] | Lower means more perceptually similar |
| Configured L∞ | `ε=0.04` in normalized [0,1] attack units, enforced as `ε×255=10.2` in float pixel tensor | Lower permits less perturbation; it is a budget, not an observation |
| Serialized actual L∞ | `max_{pixel,channel} |uint8(B_adv)−uint8(B)| / 255` after PNG reload | Lower is less pixel change; due to quantization 11/255 can occur at ε=0.04 |

The frozen five-person perceptual v2 evaluator uses PIL Lanczos to resize uint8 images before tensor conversion. The 20-person v3 runners resize float tensors with PyTorch bilinear interpolation (`align_corners=False`). Both keep maximum side ≤256, but their numerical values must not be pooled as one preprocessing version. The L∞ validator permits at most `ε+1/255` for serialized uint8 (`advface/evaluation/perturbation.py`). Pre-serialization tensor L∞ and serialized actual L∞ are separate fields; they must not be substituted for each other. Masked runs additionally check zero serialized delta outside the final mask. LPIPS and DISTS model weights are runtime dependencies fetched by their libraries; their cache bytes need explicit provenance for bitwise cross-machine reproducibility. SSIM, LPIPS and DISTS values depend on the pinned package versions in `requirements.txt`, image decoding, resize and their pretrained weights.

For exact implementation calls and aggregation fields, see `scripts/evaluate_lfw_perceptual_v2.py` (frozen five-person comparison), `scripts/run_lfw_dev_v2.py` and `scripts/run_lfw_two_phase_v4.py` (20-person runs). The proposed five-model smoke uses `advface/evaluation/multi_victim.py` for rates. Formal v3 summary artifacts are under `results/transfer/lfw_attack_dev_20_four_method_v3/`.
