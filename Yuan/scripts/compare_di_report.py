"""Validate persisted DI/baseline artifacts and compute native RGB SSIM."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

parser = argparse.ArgumentParser()
for name in ('baseline', 'baseline-mask', 'di', 'di-mask'):
    parser.add_argument('--' + name, required=True, type=Path)
args = parser.parse_args()
runs = {'MI': args.baseline.resolve(), 'MI+Mask': args.baseline_mask.resolve(),
        'MI+DI': args.di.resolve(), 'MI+DI+Mask': args.di_mask.resolve()}
configs = {name: json.loads((path / 'config.json').read_text(encoding='utf-8')) for name, path in runs.items()}
for config in configs.values():
    for key in ('identities', 'eps', 'steps', 'alpha', 'momentum', 'seed', 'split_sha256', 'runtime'):
        assert config[key] == configs['MI'][key], f'comparison mismatch: {key}'
assert all(configs[name]['di']['enabled'] for name in ('MI+DI', 'MI+DI+Mask'))
assert all(not configs[name].get('di', {}).get('enabled', False) for name in ('MI', 'MI+Mask'))
assert configs['MI+DI']['di'] == configs['MI+DI+Mask']['di']
assert all(configs[name]['mask'] is None for name in ('MI', 'MI+DI'))
assert all(configs[name]['mask'] == 'landmark_superpixel_mask' for name in ('MI+Mask', 'MI+DI+Mask'))
assert configs['MI+Mask']['mask_options'] == configs['MI+DI+Mask']['mask_options']
root = Path(configs['MI']['split']).parents[4]
with Path(configs['MI']['split']).open(encoding='utf-8') as handle:
    pairs = {r['identity_id']: r for r in csv.DictReader(handle)}
metrics, aggregates, indexed = [], {}, {}
for condition, path in runs.items():
    summary = json.loads((path / 'summary.json').read_text(encoding='utf-8'))
    assert summary['completed_pairs'] == 20 and not summary['failures']
    with (path / 'results.csv').open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    indexed[condition] = {(r['mode'], r['identity_id']): r for r in rows}
    assert len(rows) == len(indexed[condition]) == 40
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
            assert float(row['linf_tensor']) <= configs[condition]['eps'] + 1e-6
            if configs[condition]['mask']:
                mask = cv2.imread(row['mask_image'], cv2.IMREAD_GRAYSCALE)
                outside = difference[mask == 0]
                assert outside.size == 0 or outside.max() == 0
            if configs[condition].get('di', {}).get('enabled'):
                assert 0 < int(row['di_applied_steps']) <= configs[condition]['steps']
            metric = {'condition': condition, 'mode': mode, 'identity_id': row['identity_id'],
                      'ssim_rgb': float(structural_similarity(cv2.cvtColor(b, cv2.COLOR_BGR2RGB),
                          cv2.cvtColor(adv, cv2.COLOR_BGR2RGB), channel_axis=-1, data_range=255)),
                      'rgb_mae': float(difference.mean())}
            quality.append(metric)
            metrics.append(metric)
        aggregates[f'{mode}:{condition}'] = {'n': 20, 'source': summary[mode]['source'],
            'victim': summary[mode]['victim'], **{key: float(np.mean([r[key] for r in quality]))
                                                for key in ('ssim_rgb', 'rgb_mae')}}
lines = ['# MI-FGSM + DI development comparison', '',
         'Same fixed 20 identities; epsilon=0.04, steps=200, alpha=0.0002, momentum=1.0, seed=0.',
         'DI fixed-size ArcFace adaptation: p=0.7; shrink aligned normalized B crop to 90-100%, random padding to 112x112 with neutral RGB=127.5; reference unchanged.',
         'FaceNet is evaluation-only. This is one preselected DI configuration, not a tuned optimum.', '',
         '| Reference | Method | ArcFace /20 | FaceNet /20 | FaceNet /clean-valid | SSIM RGB | RGB MAE |',
         '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
for label, item in aggregates.items():
    mode, method = label.split(':')
    src, vic = item['source'], item['victim']
    lines.append(f"| {mode} | {method} | {src['successes']}/20 ({src['successes']/20:.1%}) | {vic['successes']}/20 ({vic['successes']/20:.1%}) | {vic['successes']}/{vic['eligible']} ({vic['asr']:.1%}) | {item['ssim_rgb']:.6f} | {item['rgb_mae']:.6f} |")
for old, new in (('MI', 'MI+DI'), ('MI+Mask', 'MI+DI+Mask')):
    for mode in ('self', 'pair'):
        gained, lost = [], []
        for identity in configs['MI']['identities']:
            a, b = indexed[old][mode, identity], indexed[new][mode, identity]
            for key in ('source_clean_cosine', 'victim_clean_cosine'):
                assert abs(float(a[key]) - float(b[key])) < 1e-5, f'clean preprocessing mismatch: {identity}'
            success_old = a['victim_boundary_crossed'] == 'True'
            success_new = b['victim_boundary_crossed'] == 'True'
            if success_new and not success_old:
                gained.append(identity)
            if success_old and not success_new:
                lost.append(identity)
        lines += ['', f'{mode}, {old} -> {new}: FaceNet gained: {gained}; lost: {lost}']
lines += ['', 'Richard Myers is clean-invalid on FaceNet; denominator 19 is reported alongside all-20 counts.',
          'All 160 decoded adversarial PNGs passed tensor/serialized epsilon checks; all masked outputs have zero outside-mask delta.',
          'SSIM: native decoded RGB, data_range=255, channel_axis=-1, scikit-image defaults. LPIPS/DISTS not measured.',
          'Only this fixed development sample is evaluated; do not infer general transfer rates.',
          'DI inspiration: https://arxiv.org/abs/1803.06978; fixed-size adaptation and parameters above are implementation choices.', '',
          *[f'{name}: {path}' for name, path in runs.items()]]
out = runs['MI+DI+Mask']
(out / 'DI_COMPARISON.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
(out / 'di_comparison.json').write_text(json.dumps({'runs': {k: str(v) for k,v in runs.items()}, 'aggregate': aggregates}, indent=2), encoding='utf-8')
with (out / 'di_ssim_metrics.csv').open('w', encoding='utf-8', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
    writer.writeheader()
    writer.writerows(metrics)
print('\n'.join(lines))
