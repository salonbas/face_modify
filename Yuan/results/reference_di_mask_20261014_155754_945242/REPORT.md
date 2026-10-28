# MI-FGSM self-reference vs A-reference

Fixed development identities: 20; eps=0.04; steps=200; alpha=0.0002; momentum=1.0.
Mask: landmark_superpixel_mask; options: {'n_segments': 100, 'compactness': 10.0, 'sigma': 0.0}
Input diversity: {'enabled': True, 'probability': 0.7, 'min_scale': 0.9, 'seed': 0, 'location': 'normalized aligned B crop only; A/reference unchanged', 'padding': 'neutral RGB 127.5', 'variant': 'shrink+random pad to fixed 112x112'}

FaceNet evaluation only; no victim gradients. Experimental results, not formal benchmark authority.

| Objective | ArcFace ASR | FaceNet crossing / all 20 | FaceNet ASR / clean-valid pairs | ArcFace mean cosine drop | FaceNet mean cosine drop (clean-valid) |
| --- | ---: | ---: | ---: | ---: | ---: |
| self | 9/20 (45.0%) | 8/20 (40.0%) | 8/19 (42.1%) | 0.528764 | 0.285397 |
| pair | 20/20 (100.0%) | 12/20 (60.0%) | 12/19 (63.2%) | 0.823948 | 0.389512 |

source fail→success: Michelle_Collins, Meryl_Streep, Jolanta_Kwasniewski, Michel_Temer, Justin_Timberlake, Pierre_Pettigrew, Jorge_Rodolfo_Canicoba_Corral, Sally_Kirkland, Paul_Bremer, Junichiro_Koizumi, Heizo_Takenaka
source success→fail: none
source clean-invalid: none

victim fail→success: Meryl_Streep, Jolanta_Kwasniewski, Angela_Lansbury, Maria_Shriver, Jennifer_Rodriguez
victim success→fail: Michel_Temer
victim clean-invalid: Richard_Myers (clean cosine=0.186393)

A-reference objective decreased on 20/20 identities; mean 0.665900 → -0.171571 (tensor objective before PNG serialization).

Verified 40 decoded PNGs: tensor epsilon and serialized epsilon+quantization checks passed.

SSIM is evaluated separately from persisted PNGs; see DI_COMPARISON.md or MASK_COMPARISON.md when present. LPIPS/DISTS are not evaluated; do not infer imperceptibility from this run.

Runtime differs from historical Yeah results; compare the two objectives within this run.
InsightFace 0.7.3 built from checksum-verified official sdist without the unused C++ mesh renderer; recognition/alignment Python code and ONNX weights are unchanged.
ONNX conversion passes an in-memory ModelProto to avoid a Windows temporary-file locking problem.
