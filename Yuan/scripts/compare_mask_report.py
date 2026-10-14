"""Read-only evaluation of saved masked/unmasked PNGs; no attack regeneration."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', required=True, type=Path)
parser.add_argument('--masked', required=True, type=Path)
args = parser.parse_args()
runs = {'unmasked': args.baseline.resolve(), 'mask': args.masked.resolve()}
configs = {name: json.loads((path / 'config.json').read_text(encoding='utf-8')) for name, path in runs.items()}
for key in ('identities', 'eps', 'steps', 'alpha', 'momentum', 'seed', 'split_sha256', 'runtime'):
    assert configs['unmasked'][key] == configs['mask'][key], f'comparison mismatch: {key}'
assert configs['unmasked']['mask'] is None
assert configs['mask']['mask'] == 'landmark_superpixel_mask'
root = Path(configs['mask']['split']).parents[4]
with Path(configs['mask']['split']).open(encoding='utf-8') as handle:
    pairs = {r['identity_id']: r for r in csv.DictReader(handle)}
metrics, aggregates = [], {}
for condition, path in runs.items():
    summary = json.loads((path / 'summary.json').read_text(encoding='utf-8'))
    assert summary['completed_pairs'] == 20 and not summary['failures']
    with (path / 'results.csv').open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    for mode in ('self', 'pair'):
        selected = [row for row in rows if row['mode'] == mode]
        assert len(selected) == 20
        quality = []
        for row in selected:
            b = cv2.imread(str(root / pairs[row['identity_id']]['probe_image']))
            adv = cv2.imread(row['adversarial_image'])
            assert b is not None and adv is not None and b.shape == adv.shape
            difference = np.abs(adv.astype(np.int16) - b.astype(np.int16))
            assert difference.max() / 255 <= configs[condition]['eps'] + 1 / 255 + 1e-6
            if condition == 'mask':
                mask = cv2.imread(row['mask_image'], cv2.IMREAD_GRAYSCALE)
                outside = difference[mask == 0]
                assert outside.size == 0 or outside.max() == 0
            metric = {'condition': condition, 'mode': mode, 'identity_id': row['identity_id'],
                      'ssim_rgb': float(structural_similarity(cv2.cvtColor(b, cv2.COLOR_BGR2RGB),
                           cv2.cvtColor(adv, cv2.COLOR_BGR2RGB), channel_axis=-1, data_range=255)),
                      'rgb_mae': float(difference.mean()),
                      'changed_pixel_ratio': float(np.any(difference != 0, axis=2).mean())}
            quality.append(metric)
            metrics.append(metric)
        label = f'{mode}_{condition}'
        aggregates[label] = {'n': 20, 'source': summary[mode]['source'], 'victim': summary[mode]['victim'],
                            **{key: float(np.mean([row[key] for row in quality]))
                               for key in ('ssim_rgb', 'rgb_mae', 'changed_pixel_ratio')}}
lines = ['# MI-FGSM: reference objective x landmark-superpixel mask', '',
         'Same frozen 20 development identities, epsilon=0.04, 200 steps, alpha=0.0002, momentum=1.0, seed=0, tv_weight=0.',
         'FaceNet receives no gradients. Mask defaults: SLIC n_segments=100, compactness=10, sigma=0.', '',
         '| Method | ArcFace /20 | FaceNet /20 | FaceNet /clean-valid | SSIM RGB (higher is better) | RGB MAE (0-255, lower is better) |',
         '| --- | ---: | ---: | ---: | ---: | ---: |']
for label, item in aggregates.items():
    source, victim = item['source'], item['victim']
    lines.append(f"| {label} | {source['successes']}/20 ({source['successes']/20:.1%}) | {victim['successes']}/20 ({victim['successes']/20:.1%}) | {victim['successes']}/{victim['eligible']} ({victim['asr']:.1%}) | {item['ssim_rgb']:.6f} | {item['rgb_mae']:.6f} |")
lines += ['', 'Richard Myers is clean-invalid on FaceNet; all-20 and eligible-19 proportions are reported separately.',
          'All 80 saved PNGs passed serialized epsilon checks; all 40 masked PNGs have zero change outside the saved binary mask.',
          'SSIM is computed on native-resolution decoded RGB images with scikit-image defaults, data_range=255, channel_axis=-1.',
          'LPIPS and DISTS are not evaluated; higher SSIM does not establish human imperceptibility.',
          'Landmark semantic index mapping follows the existing implementation assumption; overlays are saved for inspection.',
          'This single fixed development comparison does not estimate general transferability.', '',
          f"Unmasked source: {runs['unmasked']}", f"Masked source: {runs['mask']}"]
out = runs['mask']
(out / 'MASK_COMPARISON.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
(out / 'mask_comparison.json').write_text(json.dumps({'runs': {k: str(v) for k,v in runs.items()}, 'aggregate': aggregates}, indent=2), encoding='utf-8')
with (out / 'ssim_metrics.csv').open('w', encoding='utf-8', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
    writer.writeheader()
    writer.writerows(metrics)
print('\n'.join(lines))
