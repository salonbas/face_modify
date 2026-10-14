"""Verify saved paired results and write a concise experiment report."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('run', type=Path)
args = parser.parse_args()
run = args.run.resolve()
config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
with (run / 'results.csv').open(encoding='utf-8', newline='') as handle:
    rows = list(csv.DictReader(handle))
assert not summary['failures'], summary['failures']
assert summary['completed_pairs'] == summary['requested']
assert len(rows) == 2 * summary['requested']
assert len({(r['identity_id'], r['mode']) for r in rows}) == len(rows)
assert {r['identity_id'] for r in rows} == set(config['identities'])

import cv2
import numpy as np
root = Path(config['split']).parents[4]
with Path(config['split']).open(encoding='utf-8', newline='') as handle:
    pairs = {r['identity_id']: r for r in csv.DictReader(handle)}
for row in rows:
    b = cv2.imread(str(root / pairs[row['identity_id']]['probe_image']))
    adv = cv2.imread(row['adversarial_image'])
    assert b is not None and adv is not None and b.shape == adv.shape
    linf = np.abs(adv.astype(np.int16) - b.astype(np.int16)).max() / 255
    assert abs(linf - float(row['linf_serialized'])) < 1e-8
    assert linf <= config['eps'] + 1 / 255 + 1e-6
    assert float(row['linf_tensor']) <= config['eps'] + 1e-6
    if config.get('mask'):
        mask = cv2.imread(row['mask_image'], cv2.IMREAD_GRAYSCALE)
        assert mask is not None and mask.shape == b.shape[:2]
        outside = np.abs(adv.astype(np.int16) - b.astype(np.int16))[mask == 0]
        assert outside.size == 0 or outside.max() == 0
        assert int(row['mask_outside_delta_max']) == 0

lines = ['# MI-FGSM self-reference vs A-reference', '',
         f"Fixed development identities: {summary['requested']}; eps={config['eps']}; steps={config['steps']}; alpha={config['alpha']}; momentum={config['momentum']}.",
         f"Mask: {config.get('mask')}; options: {config.get('mask_options')}",
         '', 'FaceNet evaluation only; no victim gradients. Experimental results, not formal benchmark authority.', '',
         '| Objective | ArcFace ASR | FaceNet crossing / all 20 | FaceNet ASR / clean-valid pairs | ArcFace mean cosine drop | FaceNet mean cosine drop (clean-valid) |',
         '| --- | ---: | ---: | ---: | ---: | ---: |']
for mode in ('self', 'pair'):
    src, vic = summary[mode]['source'], summary[mode]['victim']
    lines.append(f"| {mode} | {src['successes']}/{src['eligible']} ({src['asr']:.1%}) | {vic['successes']}/{summary['requested']} ({vic['successes']/summary['requested']:.1%}) | {vic['successes']}/{vic['eligible']} ({vic['asr']:.1%}) | {src['mean_cosine_drop']:.6f} | {vic['mean_cosine_drop']:.6f} |")
by_mode = {mode: {r['identity_id']: r for r in rows if r['mode'] == mode} for mode in ('self', 'pair')}
for model in ('source', 'victim'):
    gained, lost = [], []
    for identity in config['identities']:
        old = by_mode['self'][identity][f'{model}_boundary_crossed'] == 'True'
        new = by_mode['pair'][identity][f'{model}_boundary_crossed'] == 'True'
        if new and not old:
            gained.append(identity)
        if old and not new:
            lost.append(identity)
    lines += ['', f"{model} fail→success: {', '.join(gained) or 'none'}",
              f"{model} success→fail: {', '.join(lost) or 'none'}"]
    excluded = [f"{r['identity_id']} (clean cosine={float(r[f'{model}_clean_cosine']):.6f})"
                for r in rows if r['mode'] == 'self' and r[f'{model}_clean_valid'] != 'True']
    lines.append(f"{model} clean-invalid: {', '.join(excluded) or 'none'}")
pair_rows = [r for r in rows if r['mode'] == 'pair']
decreased = sum(float(r['final_objective_cosine']) < float(r['initial_objective_cosine']) for r in pair_rows)
mean_initial = sum(float(r['initial_objective_cosine']) for r in pair_rows) / len(pair_rows)
mean_final = sum(float(r['final_objective_cosine']) for r in pair_rows) / len(pair_rows)
lines += ['', f"A-reference objective decreased on {decreased}/{len(pair_rows)} identities; mean {mean_initial:.6f} → {mean_final:.6f} (tensor objective before PNG serialization)."]
lines += ['', f"Verified {len(rows)} decoded PNGs: tensor epsilon and serialized epsilon+quantization checks passed.",
          '', 'SSIM is evaluated separately by compare_mask_report.py; see MASK_COMPARISON.md when present. LPIPS/DISTS are not evaluated; do not infer imperceptibility from this run.',
          '', 'Runtime differs from historical Yeah results; compare the two objectives within this run.',
          'InsightFace 0.7.3 built from checksum-verified official sdist without the unused C++ mesh renderer; recognition/alignment Python code and ONNX weights are unchanged.',
          'ONNX conversion passes an in-memory ModelProto to avoid a Windows temporary-file locking problem.']
(run / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
(run / 'requirements-tested.txt').write_text(freeze, encoding='utf-8')
source_root = run.parents[1]
files = ['scripts/compare_reference.py', 'advface/attacks/mi_fgsm.py',
         'advface/models/arcface_torch.py', 'advface/models/insightface_app.py', 'advface/models/facenet.py']
if config.get('mask'):
    files.append('advface/constraints/landmark_superpixel.py')
(run / 'source_sha256.json').write_text(json.dumps({name: hashlib.sha256((source_root / name).read_bytes()).hexdigest() for name in files}, indent=2), encoding='utf-8')
print('\n'.join(lines))
