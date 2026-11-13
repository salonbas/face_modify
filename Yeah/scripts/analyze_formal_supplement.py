"""Read-only pixel and paired-outcome audit of the frozen Development 100 run."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from advface.constraints import LandmarkSuperpixelMask

BASE = ROOT / 'results/transfer/lfw_dev_100_five_model_v1'
OUT = BASE / 'analysis_2026_11_14'
METHODS = {'PGD': 'pgd', 'MI-FGSM': 'mi_fgsm', 'PGD+Mask': 'pgd_mask', 'MI-FGSM+Mask': 'mi_fgsm_mask'}
MODELS = ('arcface', 'facenet', 'adaface', 'mobilefacenet', 'magface')


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    with (ROOT / 'data/datasets/lfw/evaluation_splits/v2/development.csv').open(newline='') as f:
        cohort = list(csv.DictReader(f))
    with (OUT / 'per_identity_scores.csv').open(newline='') as f:
        scores = list(csv.DictReader(f))
    by_score = {(r['method'], r['identity_id'], r['model']): r for r in scores}
    joint = [r['identity_id'] for r in cohort if all(by_score[('PGD', r['identity_id'], m)]['eligible'] == 'True' for m in MODELS)]
    assert len(joint) == 83
    mask_builder = LandmarkSuperpixelMask()
    pixels = []
    for row in cohort:
        ident = row['identity_id']
        if by_score[('PGD', ident, 'arcface')]['status'] != 'attacked':
            continue
        original = cv2.imread(str(ROOT / row['probe_image']), cv2.IMREAD_COLOR)
        assert original is not None
        mask, _, _ = mask_builder.build(original)
        allowed = mask.astype(bool)
        for method, folder in METHODS.items():
            base = BASE / 'formal' / folder
            image = cv2.imread(str(base / 'images' / (ident + '.png')), cv2.IMREAD_COLOR)
            meta = json.loads((base / 'artifacts' / (ident + '.json')).read_text())
            assert image is not None and image.shape == original.shape
            delta = np.abs(image.astype(np.int16) - original.astype(np.int16))
            changed = np.any(delta != 0, axis=2)
            mx = int(delta.max())
            assert mx / 255 == meta['serialized_linf']
            assert meta['tensor_linf'] <= .04 + 1e-6
            assert mx <= 11
            outside = int(np.count_nonzero(changed & ~allowed))
            if '+Mask' in method:
                assert outside == 0, (method, ident, outside)
            pixels.append(dict(method=method, identity_id=ident, pixels=changed.size,
                               mask_pixels=int(allowed.sum()), mask_coverage=float(allowed.mean()),
                               changed_pixels=int(changed.sum()), changed_ratio=float(changed.mean()),
                               changed_inside_pixels=int(np.count_nonzero(changed & allowed)),
                               changed_outside_pixels=outside, max_channel_delta=mx,
                               tensor_linf=meta['tensor_linf']))
    assert len(pixels) == 372
    write_csv('pixel_validation.csv', pixels)
    summaries = []
    for method in METHODS:
        group = [r for r in pixels if r['method'] == method]
        summaries.append(dict(method=method, n=len(group), mean_mask_coverage=np.mean([r['mask_coverage'] for r in group]),
                              mean_changed_ratio=np.mean([r['changed_ratio'] for r in group]),
                              pooled_changed_ratio=sum(r['changed_pixels'] for r in group)/sum(r['pixels'] for r in group),
                              changed_outside_pixels=sum(r['changed_outside_pixels'] for r in group),
                              max_channel_delta=max(r['max_channel_delta'] for r in group),
                              max_tensor_linf=max(r['tensor_linf'] for r in group)))
    write_csv('pixel_summary.csv', summaries)
    comparisons = []
    pairs = [('MI-FGSM', 'PGD'), ('PGD+Mask', 'PGD'), ('MI-FGSM+Mask', 'MI-FGSM'), ('MI-FGSM+Mask', 'PGD+Mask')]
    rng = np.random.default_rng(20261114)
    for model in MODELS:
        for a, b in pairs:
            x = np.array([int(by_score[(a, ident, model)]['success'] == 'True') for ident in joint])
            y = np.array([int(by_score[(b, ident, model)]['success'] == 'True') for ident in joint])
            d = x-y
            gain, loss = int(np.sum(d == 1)), int(np.sum(d == -1))
            boot = rng.choice(d, (20000, len(d)), replace=True).mean(axis=1)
            lo, hi = np.quantile(boot, [.025, .975])
            p = binomtest(min(gain, loss), gain+loss, .5).pvalue if gain+loss else 1.0
            comparisons.append(dict(model=model, comparison=f'{a} - {b}', n=len(joint), a_success=int(x.sum()),
                                    b_success=int(y.sum()), difference_pp=100*d.mean(), ci_low_pp=100*lo,
                                    ci_high_pp=100*hi, gain=gain, loss=loss, mcnemar_exact_p=p))
    write_csv('paired_comparisons.csv', comparisons)
    print(json.dumps({'pixels': summaries, 'comparisons': comparisons}, indent=2))


if __name__ == '__main__':
    main()
