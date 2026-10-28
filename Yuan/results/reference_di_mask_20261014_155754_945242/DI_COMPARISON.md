# MI-FGSM + DI development comparison

Same fixed 20 identities; epsilon=0.04, steps=200, alpha=0.0002, momentum=1.0, seed=0.
DI fixed-size ArcFace adaptation: p=0.7; shrink aligned normalized B crop to 90-100%, random padding to 112x112 with neutral RGB=127.5; reference unchanged.
FaceNet is evaluation-only. This is one preselected DI configuration, not a tuned optimum.

| Reference | Method | ArcFace /20 | FaceNet /20 | FaceNet /clean-valid | SSIM RGB | RGB MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| self | MI | 20/20 (100.0%) | 8/20 (40.0%) | 8/19 (42.1%) | 0.934743 | 2.170874 |
| pair | MI | 20/20 (100.0%) | 13/20 (65.0%) | 13/19 (68.4%) | 0.934300 | 2.178232 |
| self | MI+Mask | 8/20 (40.0%) | 6/20 (30.0%) | 6/19 (31.6%) | 0.974616 | 0.934571 |
| pair | MI+Mask | 20/20 (100.0%) | 9/20 (45.0%) | 9/19 (47.4%) | 0.974454 | 0.945041 |
| self | MI+DI | 20/20 (100.0%) | 10/20 (50.0%) | 10/19 (52.6%) | 0.933405 | 2.258430 |
| pair | MI+DI | 20/20 (100.0%) | 14/20 (70.0%) | 14/19 (73.7%) | 0.933692 | 2.267205 |
| self | MI+DI+Mask | 9/20 (45.0%) | 8/20 (40.0%) | 8/19 (42.1%) | 0.974182 | 0.972841 |
| pair | MI+DI+Mask | 20/20 (100.0%) | 12/20 (60.0%) | 12/19 (63.2%) | 0.974080 | 0.984219 |

self, MI -> MI+DI: FaceNet gained: ['Michelle_Collins', 'Meryl_Streep', 'Sally_Kirkland', 'Paul_Bremer']; lost: ['Jolanta_Kwasniewski', 'Angela_Lansbury']

pair, MI -> MI+DI: FaceNet gained: ['Jennifer_Rodriguez']; lost: []

self, MI+Mask -> MI+DI+Mask: FaceNet gained: ['Michel_Temer', 'Justin_Timberlake', 'Sally_Kirkland', 'Paul_Bremer']; lost: ['Jolanta_Kwasniewski', 'Angela_Lansbury']

pair, MI+Mask -> MI+DI+Mask: FaceNet gained: ['Meryl_Streep', 'Paul_Bremer', 'Jennifer_Rodriguez']; lost: []

Richard Myers is clean-invalid on FaceNet; denominator 19 is reported alongside all-20 counts.
All 160 decoded adversarial PNGs passed tensor/serialized epsilon checks; all masked outputs have zero outside-mask delta.
SSIM: native decoded RGB, data_range=255, channel_axis=-1, scikit-image defaults. LPIPS/DISTS not measured.
Only this fixed development sample is evaluated; do not infer general transfer rates.
DI inspiration: https://arxiv.org/abs/1803.06978; fixed-size adaptation and parameters above are implementation choices.

MI: C:\senior_project\face_modify\Yuan\results\reference_20261013_145019_577894
MI+Mask: C:\senior_project\face_modify\Yuan\results\reference_mask_20261013_154854_692028
MI+DI: C:\senior_project\face_modify\Yuan\results\reference_di_20261014_153851_517219
MI+DI+Mask: C:\senior_project\face_modify\Yuan\results\reference_di_mask_20261014_155754_945242
