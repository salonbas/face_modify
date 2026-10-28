# A-reference MI-FGSM + DI + TI

Same frozen 20 identities; epsilon=0.04, steps=200, alpha=0.0002, momentum=1.0, seed=0. FaceNet is evaluation-only.
DI: p=0.7, shrink normalized aligned B crop to 90-100%, random neutral padding to 112x112; reference A is fixed.
TI settings: {'enabled': True, 'kernel_size': 5, 'sigma_pixels': 1.0, 'padding': 'replicate', 'domain': 'full-image gradient', 'channels': 'independent depthwise', 'order': 'DI backward -> TI -> mask -> L1 normalization -> momentum -> sign -> projection'}
Gaussian convolution is channel-independent on the full-image raw gradient, before mask, L1 normalization, momentum and sign. Replicate boundary padding is an implementation choice.

| Method (all A-reference) | ArcFace /20 | FaceNet /20 | FaceNet /clean-valid | SSIM RGB | RGB MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| DI | 20/20 (100.0%) | 14/20 (70.0%) | 14/19 (73.7%) | 0.933692 | 2.267205 |
| DI+Mask | 20/20 (100.0%) | 12/20 (60.0%) | 12/19 (63.2%) | 0.974080 | 0.984219 |
| DI+TI | 20/20 (100.0%) | 16/20 (80.0%) | 16/19 (84.2%) | 0.933951 | 2.434873 |
| DI+TI+Mask | 20/20 (100.0%) | 12/20 (60.0%) | 12/19 (63.2%) | 0.974968 | 1.013600 |

DI -> DI+TI: FaceNet gained: ['Michelle_Collins', 'Jorge_Rodolfo_Canicoba_Corral']; lost: []

DI+Mask -> DI+TI+Mask: FaceNet gained: []; lost: []

All 80 decoded PNGs passed tensor/serialized epsilon checks; every masked output has zero outside-mask delta.
Richard Myers is clean-invalid on FaceNet; all-20 and eligible-19 proportions are reported separately.
SSIM: native decoded RGB, data_range=255, channel_axis=-1, scikit-image defaults. LPIPS/DISTS not measured.
One fixed TI setting and seed on a development sample; no parameter tuning on victim scores and no general transfer-rate claim.
TI reference: https://arxiv.org/abs/1904.02884 (CVPR 2019); kernel size/sigma and boundary rule above are explicit implementation choices.

DI: C:\senior_project\face_modify\Yuan\results\reference_di_20261014_153851_517219
DI+Mask: C:\senior_project\face_modify\Yuan\results\reference_di_mask_20261014_155754_945242
DI+TI: C:\senior_project\face_modify\Yuan\results\reference_di_ti_20261026_235225_545727
DI+TI+Mask: C:\senior_project\face_modify\Yuan\results\reference_di_ti_mask_20261026_235415_117069
