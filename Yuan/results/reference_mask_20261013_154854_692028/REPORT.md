# MI-FGSM self-reference vs A-reference

Fixed development identities: 20; eps=0.04; steps=200; alpha=0.0002; momentum=1.0.
Mask: landmark_superpixel_mask; options: {'n_segments': 100, 'compactness': 10.0, 'sigma': 0.0}

FaceNet evaluation only; no victim gradients. Experimental results, not formal benchmark authority.

| Objective | ArcFace ASR | FaceNet crossing / all 20 | FaceNet ASR / clean-valid pairs | ArcFace mean cosine drop | FaceNet mean cosine drop (clean-valid) |
| --- | ---: | ---: | ---: | ---: | ---: |
| self | 8/20 (40.0%) | 6/20 (30.0%) | 6/19 (31.6%) | 0.529480 | 0.256241 |
| pair | 20/20 (100.0%) | 9/20 (45.0%) | 9/19 (47.4%) | 0.834529 | 0.327986 |

source fail→success: Michelle_Collins, Meryl_Streep, Jolanta_Kwasniewski, Richard_Myers, Michel_Temer, Justin_Timberlake, Pierre_Pettigrew, Maria_Shriver, Sally_Kirkland, Paul_Bremer, Junichiro_Koizumi, Heizo_Takenaka
source success→fail: none
source clean-invalid: none

victim fail→success: Justin_Timberlake, Maria_Shriver, Sally_Kirkland
victim success→fail: none
victim clean-invalid: Richard_Myers (clean cosine=0.186393)

A-reference objective decreased on 20/20 identities; mean 0.665900 → -0.195904 (tensor objective before PNG serialization).

Verified 40 decoded PNGs: tensor epsilon and serialized epsilon+quantization checks passed.

SSIM is evaluated separately by compare_mask_report.py; see MASK_COMPARISON.md when present. LPIPS/DISTS are not evaluated; do not infer imperceptibility from this run.

Runtime differs from historical Yeah results; compare the two objectives within this run.
InsightFace 0.7.3 built from checksum-verified official sdist without the unused C++ mesh renderer; recognition/alignment Python code and ONNX weights are unchanged.
ONNX conversion passes an in-memory ModelProto to avoid a Windows temporary-file locking problem.
