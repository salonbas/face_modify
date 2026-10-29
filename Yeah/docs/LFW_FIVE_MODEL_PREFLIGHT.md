# LFW v2 Development 100: five-model benchmark readiness (2026-10-29)

## Final decision: NOT READY / HARD STOP

All five checkpoint files now load and all five Calibration v2 thresholds exist.
Development 100 clean eligibility and a two-person PGD+Mask end-to-end smoke
completed. The formal 100 × 4 × 5 run remains prohibited: the historical
four-method attack-only mean is 181.9 minutes, and this turn's two-method
attack-only measurements already provide a 77.2-minute lower bound. Both
exceed the absolute 60-minute wall-clock cap before inference, metrics, and
writing. A validated ≤45-minute full estimate and safe full executor are
absent. The fail-closed `--execute` gate exits 2 and creates no formal output.

## Completed model and calibration work

All model files are outside Git; exact local file hashes, backbone, training
data, preprocessing, source, and restrictions are in
`models/checkpoint_manifest.json`. The official InsightFace `buffalo_sc.zip`
matched its published archive SHA-256. The author Google Drive pages for
AdaFace and MagFace did not publish independent file hashes; their local hashes
are recorded, strict state-dict loading succeeded, and AdaFace's author sample
similarities were reproduced to within about 0.006. The author repositories
license their code, while pretrained-weight usage terms still need review
before a formal release or run.

| Model | Threshold | Train FAR / TAR | Held-out test FAR / TAR | Dev 100 clean valid / eligible |
| --- | ---: | ---: | ---: | ---: |
| ArcFace R50 | 0.140721 | 0.91% / 97.90% | 1.41% / 97.15% | 93 / 93 |
| FaceNet | 0.408859 | 0.91% / 94.70% | 1.60% / 94.13% | 100 / 95 |
| AdaFace R50 | 0.146976 | 0.92% / 93.92% | 1.01% / 93.46% | 99 / 93 |
| MobileFaceNet | 0.163314 | 0.91% / 99.73% | 2.41% / 99.19% | 93 / 93 |
| MagFace iResNet100 | 0.184499 | 0.91% / 99.82% | 2.21% / 99.19% | 93 / 93 |

Thresholds were selected on isolated calibration train pairs only. Held-out
test FAR above 1% for MobileFaceNet and MagFace is a calibration-quality
warning; no test-set tuning was performed. AdaFace's lower clean TAR remains
a separate quality limitation. All 100 Development identities retain their
original denominator and model-specific detection/eligibility flags; **83**
are eligible under all five thresholds. Seven ArcFace detections fail, seven
MobileFaceNet detections fail, seven MagFace detections fail, and one AdaFace
detection fails. Clean failures never count as attack successes. OOS images
were never scored or attacked.

## Small end-to-end validation

The first two Development identities, `Adrian_McPherson` and `Ahmad_Masood`,
ran one 200-step PGD+Mask attack each. The same decoded PNG was used for all
five model scores and perceptual metrics. Both serialized masks had zero
perturbation outside the final mask and a serialized L∞ of 11/255 (0.043137),
within the existing one-quantization-level allowance above ε=0.04. No metric
was NaN or Inf. The first PNG was regenerated independently from the same
settings and its file hash and pixels matched exactly. The second invocation
validated and resumed the first record instead of overwriting its artifact.

| Identity | Attack time | SSIM | LPIPS | DISTS | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Adrian_McPherson | 20.55 s | 0.98170 | 0.02724 | 0.01910 | 3.10 GiB |
| Ahmad_Masood | 20.30 s | 0.99259 | 0.01112 | 0.02202 | 3.08 GiB |

| Identity | Model | Clean cosine | B_adv cosine | Threshold crossing |
| --- | --- | ---: | ---: | --- |
| Adrian_McPherson | ArcFace / FaceNet / AdaFace / MobileFaceNet / MagFace | 0.6522 / 0.8062 / 0.6669 / 0.5351 / 0.6888 | 0.2648 / 0.8417 / 0.4136 / 0.3703 / 0.4036 | no / no / no / no / no |
| Ahmad_Masood | ArcFace / FaceNet / AdaFace / MobileFaceNet / MagFace | 0.5716 / 0.5992 / 0.4917 / 0.4704 / 0.5719 | 0.0872 / 0.5424 / 0.1824 / 0.3198 / 0.2469 | yes / no / no / no / no |

All ten clean scores passed their thresholds in this two-person subset, so
conditional and paired ASR have the same denominator of two here. The result
is an engineering smoke, not a 100-person efficacy estimate. The one-person
run took 31.66 s after Python startup; its initialization was 3.65 s,
perceptual metrics 5.07 s, and three-image inference was 0.74/0.17/0.20/
0.06/0.42 s for ArcFace/FaceNet/AdaFace/MobileFaceNet/MagFace. Peak process
RSS was 3.10 GiB. For 200 Development clean images, model-specific times
were 63.87/13.26/13.92/6.23/21.36 s, including one initialization each.

## Reproducible commands and remaining work

Use `.venv/bin/python scripts/run_lfw_five_model_benchmark.py --preflight` to
recheck all five files, hashes, thresholds, Development manifest, calibration
isolation, memory, output state, and timing gate. It currently exits 2.
`.venv/bin/python scripts/run_lfw_five_model_benchmark.py --execute` likewise
exits 2 without starting the full experiment. The smoke records and clean
eligibility JSON are under `results/smoke/lfw_five_model_pgd_mask_v1/` and
`results/transfer/lfw_dev_100_five_model_v1/`, respectively. The full
test suite passed 66 tests.

The next effective step is to establish a working differentiable GPU backend
for the unchanged ArcFace model and validate forward, gradient, mask,
projection, final PNG, and threshold decisions against the CPU baseline.
Then measure all four methods plus five-model inference/metrics with enough
identities to produce a conservative ≤45-minute estimate. A safe formal
resume executor should be implemented and tested only after that gate can
pass. This turn did not run the full 100-person benchmark.

A controlled same-image PGD+Mask CPU thread comparison after model warmup
measured 45.51 s with one Torch thread, 28.79 s with two, and 23.03 s with
four. All three reproduced the baseline PNG pixels exactly. Four threads
remain the fastest tested setting; reducing thread count was rejected.

## Earlier Stage 0 snapshot (superseded where it says checkpoints are absent)

## Current measured addendum (2026-10-29)

The fail-closed CLI is `python3 scripts/run_lfw_five_model_benchmark.py --preflight`
(use the repository `.venv/bin/python` here). It verifies all 100 Development
A/B paths and identity directories, reads the 228-name OOS manifest only for
isolation, checks calibration-pair identity isolation, hashes every present
checkpoint, checks five thresholds, available memory, and output collision.
It exits 2 with `NOT READY`. `--execute` reruns the same gate and also exits 2;
there is no formal five-model executor yet. Thus the CLI is a tested safety
control, not a claim that a one-click run is ready.

This WSL host reports Ryzen 5 7500F, eight visible logical CPUs (four visible
cores × two threads), Ubuntu 24.04.1, and kernel
`6.18.33.2-microsoft-standard-WSL2`; Windows reports WSL 2.7.10.0.
Windows video-controller query identifies AMD Radeon RX 6800 with driver
32.0.21045.16010 (plus an IddSampleDriver virtual display device).
Torch 2.12.0+cu130 reports no CUDA device
and no ROCm/HIP build; ONNX Runtime 1.26.0 exposes CPU and Azure providers,
and the actual InsightFace sessions use `CPUExecutionProvider`. The attack
ArcFace model parameters were observed on `cpu`. Neither `/dev/dxg` nor
`/dev/kfd` is available; `torch_directml` is not installed. Windows GPU presence therefore does not establish
WSL GPU acceleration for this attack.

One Development identity, `Adrian_McPherson`, was profiled with the existing
200-step PGD+Mask attack. FaceAnalysis creation took 1.09 s; attack.apply took
24.63 s. cProfile attributed about 9.40 s to 201 ArcFace forward passes,
9.98 s to 200 backward passes, and 2.36 s to one-time ONNX→Torch conversion
inside attack.apply. These cumulative timings overlap and must not be summed.
The same identity's unmodified attack took 24.96 s PGD+Mask and 21.34 s
MI-FGSM+Mask in a paired regression run. Their 46.29 s combined attack-only
time already extrapolates to 77.2 minutes for 100 identities, before PGD and
MI-FGSM unmasked, victims, perceptual metrics, I/O, and initialization.
Historical four-method paired mean remains 181.9 minutes attack-only. The
complete end-to-end time cannot be reliably estimated while three models and
their calibration are absent; either measured lower bound triggers hard stop.

A trial zero-TV graph skip gave bit-identical serialized arrays and L∞ for the
two masked methods but timings of 24.27 s PGD+Mask and 23.56 s MI-FGSM+Mask.
The mixed result was not accepted as a speed improvement, and the code was
reverted. No attack objective, step count, mask, projection, or output rule was
changed. The next viable performance path is an actually available accelerator
with validated forward and gradient equivalence, or a measured persistent
worker/batch implementation with independent per-identity gradients. Neither
is established on this host, so the 45-minute READY target is unmet.

The two original checkpoint SHA-256 values are ArcFace
`4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43`
and FaceNet
`281cebca8662831adb987a874bdcb36e73f5b1c6dc5ee5878f305e985625d99b`.
An elevated network fetch obtained the official InsightFace buffalo_sc ZIP
and author-linked AdaFace R50/MS1MV2 and MagFace iResNet100/MS1MV2 files.
Their local hashes and source URLs are recorded in
`models/checkpoint_manifest.json`; the author source commits are pinned in
`scripts/setup_five_model_sources.sh`. The source code is kept outside Git
under `models/third_party/`, as are the weights. All three produced finite
512-D embeddings for one Development clean pair. MobileFaceNet also passed a
second clean pair and completed Calibration v2: train threshold 0.16331394,
FAR 10/1093, TAR 1088/1091; held-out test FAR 12/497, TAR 488/492.
The elevated fetch resolved the earlier sandbox-only DNS failure. The held-out
MobileFaceNet FAR of 2.41% exceeds the 1% train target and is reported as a
quality warning; no threshold was selected on test. AdaFace and MagFace full
calibration and the five-model Development eligibility check were still in
progress when this addendum was first written; later sections should be used
for their final status.

AdaFace R50/MS1MV2 subsequently calibrated on 2,178 valid train and 986 valid
test pairs. Its threshold is 0.1469759941; train FAR/TAR are 0.92%/93.92%,
held-out test FAR/TAR are 1.01%/93.46%. Its author example-image cosine
matrix was reproduced within about 0.006 absolute cosine, supporting the
checkpoint and preprocessing wiring. The lower TAR relative to ArcFace and
MobileFaceNet merits attention but was not "fixed" by test-set tuning.

## Decision

**HARD STOP.** No 100-person attack was started. The proposed five-model benchmark
has not passed Stages 1–3. The historical four-method attack generation time
alone projects well beyond the 60-minute total budget; the three new checkpoint
files are also absent. `scripts/preflight_lfw_multi_victim_v1.py` records the
source checksums and exits with code 2 to signal the stop. Its output is
`results/transfer/lfw_dev_100_five_model_v1/preflight.json`.

## Stage 0 checks

- `scripts/check_research_environment.py` passed for the existing LFW images,
  ArcFace, FaceNet, and frozen v3 results. LFW v2 has 100 Development and 228
  OOS identities. The split audit records zero v2 calibration/attack identity
  or A/B image overlap. OOS was not scored or attacked in this preflight.
- Existing FAR=1e-2 v2 thresholds are ArcFace `0.14072093367576602`
  (train FAR 10/1093; test FAR 7/497) and FaceNet `0.4088586568832398`
  (train FAR 10/1099; test FAR 8/500). New-model thresholds do not exist.
- This host exposes 8 logical CPUs (AMD Ryzen 5 7500F), 4 Torch CPU threads,
  and Torch 2.12.0; `torch.cuda.is_available()` is false. New-model inference requirements and
  peak memory remain unmeasured. No new model initialization was performed.
- Existing v3 serialized attack images cover the 20-person pilot only. Reuse
  requires manifest and PNG checksum verification; those files cannot be
  silently relabeled as v2 calibration results.

## Historical attack timing and budget

The same 20 identities appear in all four valid, 200-step, ε=0.04 result
tables. Their `runtime_seconds` values measure `attack.apply` only. The
earlier runs used separate workers, so repeated initialization may contribute
to these numbers; that cost has not yet been isolated. These are historical
measurements, not a new Stage 3 end-to-end benchmark.

| Method | n | Mean attack seconds per identity |
| --- | ---: | ---: |
| PGD | 20 | 25.06 |
| MI-FGSM (μ=1) | 20 | 27.17 |
| PGD+Mask | 20 | 26.99 |
| MI-FGSM+Mask (μ=1) | 20 | 29.91 |

Paired four-method mean is **109.14 s/identity**. Multiplying by 100 yields
**181.9 minutes** of attack generation alone; even the fastest observed paired
identity projects to **165.0 minutes**. The observed paired p95 projects to
**196.6 minutes**. These figures omit the three new victims, calibration,
separate mask timing, all five-model evaluation, SSIM/LPIPS/DISTS, file I/O,
model initialization, and margin for runtime variance. Consequently a reliable
`T_total = one_time + 100 × marginal + overhead` is unavailable; any complete
estimate on the present pipeline would exceed the limit. Download and first
installation time are excluded from these attack figures, as required.

## Candidate model provenance and integration blockers

| Model | Candidate checkpoint and provenance | Contract to validate | Present? |
| --- | --- | --- | --- |
| ArcFace R50 | Existing InsightFace `buffalo_l/w600k_r50.onnx`, WebFace600K, 512-D | Existing FaceAnalysis 112×112 aligned recognition pipeline | yes |
| FaceNet | Existing `20180402-114759-vggface2.pt`, InceptionResnetV1 VGGFace2, 512-D | Existing MTCNN 160×160, post-process normalization | yes |
| AdaFace | Author's `adaface_ir50_ms1mv2.ckpt`, IR-50, MS1MV2, 512-D | Author's MTCNN alignment, 112×112 BGR, `(x/255-0.5)/0.5`; CPU inference possible but unmeasured | no |
| MobileFaceNet | InsightFace `buffalo_sc/w600k_mbf.onnx`, MBF@WebFace600K; output dimension must be verified from the graph | InsightFace 5-point aligned 112×112 recognition crop and model metadata; CPU ONNX Runtime possible but unmeasured | no |
| MagFace | Author's MagFace iResNet100/MS1MV2 checkpoint; expected 512-D, to verify after load | Author's 5-point aligned 112×112 preprocessing and checkpoint-specific normalization; CPU inference possible but unmeasured | no |

AdaFace's [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_AdaFace_Quality_Adaptive_Margin_for_Face_Recognition_CVPR_2022_paper.pdf),
[official model list and preprocessing](https://github.com/mk-minchul/AdaFace/blob/master/README.md),
and [inference loader](https://github.com/mk-minchul/AdaFace/blob/master/inference.py)
identify the selected checkpoint. The code repository carries an
[MIT license](https://github.com/mk-minchul/AdaFace/blob/master/LICENSE); a
separate pretrained-weight redistribution grant was not verified. Store the
checkpoint outside Git and record its SHA-256 before use.

InsightFace's [model zoo](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md)
identifies `buffalo_sc` as MBF@WebFace600K; its [release](https://github.com/deepinsight/insightface/releases)
publishes the package. InsightFace states that its open pretrained models are
for [noncommercial research](https://github.com/deepinsight/insightface).
This is a different backbone from the existing ArcFace R50 even though both
are in the InsightFace family. Checkpoint digest and output shape need local
verification before inclusion.

MagFace's [paper](https://arxiv.org/abs/2103.06627) and
[official model zoo](https://github.com/IrvingMeng/MagFace#model-zoo) list the
iResNet100/MS1MV2 checkpoint. Its repository has an
[Apache-2.0 code license](https://github.com/IrvingMeng/MagFace/blob/main/LICENSE).
The checkpoint's precise usage/redistribution terms and bytes have not been
verified locally. Do not substitute a third-party MagFace+UNPG checkpoint:
it is a different training method.

## Next gate

First obtain and hash the author checkpoints and implement AdaFace alone. Its
1–2 Development-identity smoke must confirm embedding shape/finite cosine,
official preprocessing, calibration-pair scoring, and evaluation of the same
serialized `B_adv`; it must preserve old ArcFace output checksums. Repeat for
MobileFaceNet and MagFace. Calibrate each on v2 train pairs at FAR=1e-2 and
evaluate test pairs without selecting on test. Then measure a 2–5 identity
four-method end-to-end run, separating initialization, mask, ms/step, each
victim, perceptual metrics, serialization, peak memory, and marginal time.
Only a conservative estimate of **≤45 minutes**, plus all correctness gates,
can authorize the full 100-person run. Avoid changing ε, α, 200 steps,
resolution, identity count, metric precision, or model weights to meet time.
