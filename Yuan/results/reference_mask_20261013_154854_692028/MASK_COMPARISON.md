# MI-FGSM: reference objective x landmark-superpixel mask

Same frozen 20 development identities, epsilon=0.04, 200 steps, alpha=0.0002, momentum=1.0, seed=0, tv_weight=0.
FaceNet receives no gradients. Mask defaults: SLIC n_segments=100, compactness=10, sigma=0.

| Method | ArcFace /20 | FaceNet /20 | FaceNet /clean-valid | SSIM RGB (higher is better) | RGB MAE (0-255, lower is better) |
| --- | ---: | ---: | ---: | ---: | ---: |
| self_unmasked | 20/20 (100.0%) | 8/20 (40.0%) | 8/19 (42.1%) | 0.934743 | 2.170874 |
| pair_unmasked | 20/20 (100.0%) | 13/20 (65.0%) | 13/19 (68.4%) | 0.934300 | 2.178232 |
| self_mask | 8/20 (40.0%) | 6/20 (30.0%) | 6/19 (31.6%) | 0.974616 | 0.934571 |
| pair_mask | 20/20 (100.0%) | 9/20 (45.0%) | 9/19 (47.4%) | 0.974454 | 0.945041 |

Richard Myers is clean-invalid on FaceNet; all-20 and eligible-19 proportions are reported separately.
All 80 saved PNGs passed serialized epsilon checks; all 40 masked PNGs have zero change outside the saved binary mask.
SSIM is computed on native-resolution decoded RGB images with scikit-image defaults, data_range=255, channel_axis=-1.
LPIPS and DISTS are not evaluated; higher SSIM does not establish human imperceptibility.
Landmark semantic index mapping follows the existing implementation assumption; overlays are saved for inspection.
This single fixed development comparison does not estimate general transferability.

Unmasked source: C:\senior_project\face_modify\Yuan\results\reference_20261013_145019_577894
Masked source: C:\senior_project\face_modify\Yuan\results\reference_mask_20261013_154854_692028
