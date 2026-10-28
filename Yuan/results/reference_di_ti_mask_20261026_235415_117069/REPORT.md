# MI-FGSM self-reference vs A-reference

Fixed development identities: 20; eps=0.04; steps=200; alpha=0.0002; momentum=1.0.
Mask: landmark_superpixel_mask; options: {'n_segments': 100, 'compactness': 10.0, 'sigma': 0.0}
Input diversity: {'enabled': True, 'probability': 0.7, 'min_scale': 0.9, 'seed': 0, 'location': 'normalized aligned B crop only; A/reference unchanged', 'padding': 'neutral RGB 127.5', 'variant': 'shrink+random pad to fixed 112x112'}
Translation-invariant gradient: {'enabled': True, 'kernel_size': 5, 'sigma_pixels': 1.0, 'padding': 'replicate', 'domain': 'full-image gradient', 'channels': 'independent depthwise', 'order': 'DI backward -> TI -> mask -> L1 normalization -> momentum -> sign -> projection'}

FaceNet evaluation only; no victim gradients. Experimental results, not formal benchmark authority.

| Objective | ArcFace ASR | FaceNet crossing / all 20 | FaceNet ASR / clean-valid pairs | ArcFace mean cosine drop | FaceNet mean cosine drop (clean-valid) |
| --- | ---: | ---: | ---: | ---: | ---: |
| pair | 20/20 (100.0%) | 12/20 (60.0%) | 12/19 (63.2%) | 0.802178 | 0.388417 |
source clean-invalid: none
victim clean-invalid: Richard_Myers (clean cosine=0.186393)

A-reference objective decreased on 20/20 identities; mean 0.665900 → -0.141909 (tensor objective before PNG serialization).

Verified 20 decoded PNGs: tensor epsilon and serialized epsilon+quantization checks passed.

SSIM is evaluated separately from persisted PNGs; see DI_COMPARISON.md or MASK_COMPARISON.md when present. LPIPS/DISTS are not evaluated; do not infer imperceptibility from this run.

Runtime differs from historical Yeah results; compare the two objectives within this run.
InsightFace 0.7.3 built from checksum-verified official sdist without the unused C++ mesh renderer; recognition/alignment Python code and ONNX weights are unchanged.
ONNX conversion passes an in-memory ModelProto to avoid a Windows temporary-file locking problem.
