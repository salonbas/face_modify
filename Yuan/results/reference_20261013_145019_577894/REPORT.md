# MI-FGSM self-reference vs A-reference

Fixed development identities: 20; eps=0.04; steps=200; alpha=0.0002; momentum=1.0.

FaceNet evaluation only; no victim gradients. Experimental results, not formal benchmark authority.

| Objective | ArcFace ASR | FaceNet crossing / all 20 | FaceNet ASR / clean-valid pairs | ArcFace mean cosine drop | FaceNet mean cosine drop (clean-valid) |
| --- | ---: | ---: | ---: | ---: | ---: |
| self | 20/20 (100.0%) | 8/20 (40.0%) | 8/19 (42.1%) | 0.734040 | 0.298428 |
| pair | 20/20 (100.0%) | 13/20 (65.0%) | 13/19 (68.4%) | 1.114724 | 0.398506 |

source fail→success: none
source success→fail: none
source clean-invalid: none

victim fail→success: Meryl_Streep, Pierre_Pettigrew, Maria_Shriver, Sally_Kirkland, Paul_Bremer
victim success→fail: none
victim clean-invalid: Richard_Myers (clean cosine=0.186393)

A-reference objective decreased on 20/20 identities; mean 0.665900 → -0.481600 (tensor objective before PNG serialization).

Verified 40 decoded PNGs: tensor epsilon and serialized epsilon+quantization checks passed.

No SSIM/LPIPS/DISTS evaluation; do not infer imperceptibility from this run.

Runtime differs from historical Yeah results; compare the two objectives within this run.
InsightFace 0.7.3 built from checksum-verified official sdist without the unused C++ mesh renderer; recognition/alignment Python code and ONNX weights are unchanged.
ONNX conversion passes an in-memory ModelProto to avoid a Windows temporary-file locking problem.
