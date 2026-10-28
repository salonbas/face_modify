# MI-FGSM self-reference vs A-reference

Fixed development identities: 20; eps=0.04; steps=200; alpha=0.0002; momentum=1.0.
Mask: None; options: None
Input diversity: {'enabled': True, 'probability': 0.7, 'min_scale': 0.9, 'seed': 0, 'location': 'normalized aligned B crop only; A/reference unchanged', 'padding': 'neutral RGB 127.5', 'variant': 'shrink+random pad to fixed 112x112'}

FaceNet evaluation only; no victim gradients. Experimental results, not formal benchmark authority.

| Objective | ArcFace ASR | FaceNet crossing / all 20 | FaceNet ASR / clean-valid pairs | ArcFace mean cosine drop | FaceNet mean cosine drop (clean-valid) |
| --- | ---: | ---: | ---: | ---: | ---: |
| self | 20/20 (100.0%) | 10/20 (50.0%) | 10/19 (52.6%) | 0.723860 | 0.366002 |
| pair | 20/20 (100.0%) | 14/20 (70.0%) | 14/19 (73.7%) | 1.086586 | 0.448421 |

source fail→success: none
source success→fail: none
source clean-invalid: none

victim fail→success: Jolanta_Kwasniewski, Angela_Lansbury, Pierre_Pettigrew, Maria_Shriver, Jennifer_Rodriguez
victim success→fail: Michelle_Collins
victim clean-invalid: Richard_Myers (clean cosine=0.186393)

A-reference objective decreased on 20/20 identities; mean 0.665900 → -0.437328 (tensor objective before PNG serialization).

Verified 40 decoded PNGs: tensor epsilon and serialized epsilon+quantization checks passed.

SSIM is evaluated separately by compare_mask_report.py; see MASK_COMPARISON.md when present. LPIPS/DISTS are not evaluated; do not infer imperceptibility from this run.

Runtime differs from historical Yeah results; compare the two objectives within this run.
InsightFace 0.7.3 built from checksum-verified official sdist without the unused C++ mesh renderer; recognition/alignment Python code and ONNX weights are unchanged.
ONNX conversion passes an in-memory ModelProto to avoid a Windows temporary-file locking problem.
