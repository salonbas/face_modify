"""Verify A-reference DI vs DI+TI from saved outputs; no attack regeneration."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

parser = argparse.ArgumentParser()
for name in ('di', 'di-mask', 'ti', 'ti-mask'):
    parser.add_argument('--' + name, required=True, type=Path)
args = parser.parse_args()
runs = {'DI': args.di.resolve(), 'DI+Mask': args.di_mask.resolve(),
        'DI+TI': args.ti.resolve(), 'DI+TI+Mask': args.ti_mask.resolve()}
configs = {name: json.loads((path / 'config.json').read_text(encoding='utf-8')) for name, path in runs.items()}
for config in configs.values():
    for key in ('identities', 'eps', 'steps', 'alpha', 'momentum', 'seed', 'split_sha256', 'runtime', 'di'):
        assert config[key] == configs['DI'][key], f'comparison mismatch: {key}'
assert all(not configs[name].get('ti', {}).get('enabled', False) for name in ('DI', 'DI+Mask'))
assert all(configs[name]['ti']['enabled'] for name in ('DI+TI', 'DI+TI+Mask'))
assert configs['DI+TI']['ti'] == configs['DI+TI+Mask']['ti']
assert all(configs[name]['mask'] is None for name in ('DI', 'DI+TI'))
assert all(configs[name]['mask'] == 'landmark_superpixel_mask' for name in ('DI+Mask', 'DI+TI+Mask'))
assert configs['DI+Mask']['mask_options'] == configs['DI+TI+Mask']['mask_options']
root = Path(configs['DI']['split']).parents[4]
with Path(configs['DI']['split']).open(encoding='utf-8') as handle:
    pairs = {r['identity_id']: r for r in csv.DictReader(handle)}
metrics, aggregate, indexed = [], {}, {}
for condition, path in runs.items():
    summary = json.loads((path / 'summary.json').read_text(encoding='utf-8'))
    assert summary['completed_pairs'] == 20 and not summary['failures']
    with (path / 'results.csv').open(encoding='utf-8') as handle:
        rows = [r for r in csv.DictReader(handle) if r['mode'] == 'pair']
    assert len(rows) == 20 and {r['identity_id'] for r in rows} == set(configs['DI']['identities'])
    indexed[condition] = {r['identity_id']: r for r in rows}
    quality = []
    for row in rows:
        b = cv2.imread(str(root / pairs[row['identity_id']]['probe_image']))
        adv = cv2.imread(row['adversarial_image'])
        assert b is not None and adv is not None and b.shape == adv.shape
        difference = np.abs(adv.astype(np.int16) - b.astype(np.int16))
        linf = float(difference.max() / 255)
        assert abs(linf - float(row['linf_serialized'])) < 1e-8
        assert linf <= configs[condition]['eps'] + 1 / 255 + 1e-6
        assert float(row['linf_tensor']) <= configs[condition]['eps'] + 1e-6
        assert 0 < int(row['di_applied_steps']) <= configs[condition]['steps']
        if configs[condition]['mask']:
            mask = cv2.imread(row['mask_image'], cv2.IMREAD_GRAYSCALE)
            assert mask is not None and mask.shape == b.shape[:2]
            outside = difference[mask == 0]
            assert outside.size == 0 or outside.max() == 0
        if configs[condition].get('ti', {}).get('enabled'):
            assert int(row['ti_kernel_size']) == configs[condition]['ti']['kernel_size']
            assert float(row['ti_sigma']) == configs[condition]['ti']['sigma_pixels']
        metric = {'method': condition, 'identity_id': row['identity_id'],
                  'ssim_rgb': float(structural_similarity(cv2.cvtColor(b, cv2.COLOR_BGR2RGB),
                      cv2.cvtColor(adv, cv2.COLOR_BGR2RGB), channel_axis=-1, data_range=255)),
                  'rgb_mae': float(difference.mean())}
        quality.append(metric)
        metrics.append(metric)
    # Recompute ASR from clean-valid boundary crossings rather than trusting summaries.
    scores = {}
    for model in ('source', 'victim'):
        eligible = [r for r in rows if r[f'{model}_clean_valid'] == 'True']
        successes = sum(r[f'{model}_boundary_crossed'] == 'True' for r in eligible)
        assert successes == summary['pair'][model]['successes']
        assert len(eligible) == summary['pair'][model]['eligible']
        scores[model] = {'eligible': len(eligible), 'successes': successes,
                         'asr': successes / len(eligible) if eligible else None}
    aggregate[condition] = {'n': 20, **scores,
        **{key: float(np.mean([r[key] for r in quality])) for key in ('ssim_rgb', 'rgb_mae')}}
lines = ['# A-reference MI-FGSM + DI + TI', '',
         'Same frozen 20 identities; epsilon=0.04, steps=200, alpha=0.0002, momentum=1.0, seed=0. FaceNet is evaluation-only.',
         'DI: p=0.7, shrink normalized aligned B crop to 90-100%, random neutral padding to 112x112; reference A is fixed.',
         f"TI settings: {configs['DI+TI']['ti']}",
         'Gaussian convolution is channel-independent on the full-image raw gradient, before mask, L1 normalization, momentum and sign. Replicate boundary padding is an implementation choice.', '',
         '| Method (all A-reference) | ArcFace /20 | FaceNet /20 | FaceNet /clean-valid | SSIM RGB | RGB MAE |',
         '| --- | ---: | ---: | ---: | ---: | ---: |']
for label, item in aggregate.items():
    src, vic = item['source'], item['victim']
    lines.append(f"| {label} | {src['successes']}/20 ({src['successes']/20:.1%}) | {vic['successes']}/20 ({vic['successes']/20:.1%}) | {vic['successes']}/{vic['eligible']} ({vic['asr']:.1%}) | {item['ssim_rgb']:.6f} | {item['rgb_mae']:.6f} |")
for old, new in (('DI', 'DI+TI'), ('DI+Mask', 'DI+TI+Mask')):
    gained, lost = [], []
    for identity in configs['DI']['identities']:
        a, b = indexed[old][identity], indexed[new][identity]
        for key in ('source_clean_cosine', 'victim_clean_cosine'):
            assert abs(float(a[key]) - float(b[key])) < 1e-5, f'clean preprocessing mismatch: {identity}'
        assert int(a['di_applied_steps']) == int(b['di_applied_steps'])
        if configs[old]['mask']:
            assert np.array_equal(cv2.imread(a['mask_image'], 0), cv2.imread(b['mask_image'], 0)), f'mask mismatch: {identity}'
        before = a['victim_boundary_crossed'] == 'True'
        after = b['victim_boundary_crossed'] == 'True'
        if after and not before:
            gained.append(identity)
        if before and not after:
            lost.append(identity)
    lines += ['', f'{old} -> {new}: FaceNet gained: {gained}; lost: {lost}']
lines += ['', 'All 80 decoded PNGs passed tensor/serialized epsilon checks; every masked output has zero outside-mask delta.',
          'Richard Myers is clean-invalid on FaceNet; all-20 and eligible-19 proportions are reported separately.',
          'SSIM: native decoded RGB, data_range=255, channel_axis=-1, scikit-image defaults. LPIPS/DISTS not measured.',
          'One fixed TI setting and seed on a development sample; no parameter tuning on victim scores and no general transfer-rate claim.',
          'TI reference: https://arxiv.org/abs/1904.02884 (CVPR 2019); kernel size/sigma and boundary rule above are explicit implementation choices.', '',
          *[f'{name}: {path}' for name, path in runs.items()]]
out = runs['DI+TI+Mask']
(out / 'TI_COMPARISON.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
(out / 'ti_comparison.json').write_text(json.dumps({'runs': {k: str(v) for k,v in runs.items()}, 'aggregate': aggregate}, indent=2), encoding='utf-8')
with (out / 'ti_ssim_metrics.csv').open('w', encoding='utf-8', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
    writer.writeheader()
    writer.writerows(metrics)
print('\n'.join(lines))
