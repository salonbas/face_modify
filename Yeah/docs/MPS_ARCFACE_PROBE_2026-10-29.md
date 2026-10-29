# Apple M2 ArcFace MPS feasibility probe — 2026-10-29

## Decision

**NOT READY. Stop the present MPS attack route.** The model and inputs execute
forward on MPS, but the first backward pass fails because PyTorch 2.12.0 has no
MPS implementation of `aten::grid_sampler_2d_backward`. CPU fallback was
disabled. No MPS 200-step attack or formal 100-identity benchmark was run.

## Environment and assets

- macOS 15.6.1, MacBook Air M2, 16 GB unified memory, arm64.
- Project `.venv`: Python 3.12.15, PyTorch 2.12.0; MPS built and available
  outside the restricted command sandbox. The system Python has no PyTorch.
- Existing project LFW and `buffalo_l` assets were absent. After user
  authorization, the project-pinned upstream downloads were installed and
  verified against `scripts/setup_research_assets.py` and the LFW per-image
  SHA-256 manifest. The measured identity was the first frozen Development
  row, `Adrian_McPherson`. No OOS images were accessed.

## Measurements

All timings use `perf_counter`; MPS timings synchronize the device before
and after the measured operation. Same full-image affine crop, ArcFace ONNX
conversion, float32 tensors, 200 steps, epsilon 0.04, alpha 0.0002, and the
unchanged landmark superpixel mask. Outputs went only to `/tmp/face_mps_probe`.

| Observation | CPU | MPS |
| --- | ---: | ---: |
| One forward | 0.0495 s | 0.0175 s |
| One backward | 0.0759 s | unsupported operator |
| 200-step PGD+Mask | 23.9425 s | not run; cannot complete |
| Amortized time per step | 0.1197 s | not available |
| Peak process RSS | 2.06 GB | not measured after failed backward |

The model, input, and embedding tensor devices were `mps:0` during the MPS
forward. The forward embedding differed from CPU by maximum absolute
`4.17e-7` and L2 `3.29e-6`. A gradient comparison is impossible because the
MPS backward failed. No effective whole-attack acceleration factor exists.

The CPU attack's tensor L∞ was `0.03999999`; PNG L∞ was `11/255`, and maximum
change outside the mask was zero. The ArcFace A/B cosine changed from
`0.65218` to `0.31822`; the frozen v2 FAR=1e-2 threshold is `0.14072`, so this
one attack did **not** cross the verification threshold. This does not alter
the MPS feasibility finding.

## Full benchmark cost and next action

The frozen historical four-method CPU attack-only projection is 181.9 minutes
for 100 identities. The M2 CPU's one-method 23.94-second measurement does not
justify substituting a new four-method estimate. Five-model scoring, SSIM,
LPIPS, DISTS, initialization, and I/O would add time. A 45-minute complete
run would require more than 4.04-fold reduction in the historical attack-only
component; a 15-minute run would require more than 12.13-fold reduction.
This MPS route cannot achieve either target because backward cannot execute.
The 60-minute limit is likewise unsupported.

The most practical alternative is an accelerator with complete
`grid_sample` backward support, such as a CUDA host, followed by a one-person
four-method correctness and cost probe. Any MPS-specific rewrite of affine
sampling would require a separate mathematical and output validation effort;
it is not justified by this low-cost feasibility test. The formal executor
remains disabled, and `run_lfw_five_model_benchmark.py --preflight` also has a
macOS `SC_AVPHYS_PAGES` portability failure independent of the attack.
