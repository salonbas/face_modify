"""Stored-image generation and independent five-victim evaluation for LFW v2."""
from __future__ import annotations

import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from advface.attack_artifact import config_hash, save, sha256, validate, _atomic_bytes
from advface.attacks.registry import ATTACKS, AttackConfig
from advface.constraints import LandmarkSuperpixelMask
from advface.evaluation.multi_victim import MODELS, load_v2_thresholds, summarize
from advface.evaluation.similarity import cosine_similarity
from advface.image_io import load_bgr
from advface.models import load_embedder

OUT = ROOT / 'results/transfer/lfw_dev_100_five_model_v1/formal'
SPLIT = ROOT / 'data/datasets/lfw/evaluation_splits/v2/development.csv'
CLEAN_ELIGIBILITY = ROOT / 'results/transfer/lfw_dev_100_five_model_v1/clean_eligibility.json'
METHODS = ('PGD', 'MI-FGSM', 'PGD+Mask', 'MI-FGSM+Mask')
CONFIG = {'epsilon': .04, 'alpha': .0002, 'steps': 200, 'momentum': 1.0, 'tv_weight': 0.0,
          'mask': 'landmark_superpixel_mask', 'split': 'development_v2'}


def environment():
    packages = ('torch', 'torchvision', 'numpy', 'opencv-python', 'onnxruntime',
                'insightface', 'facenet-pytorch', 'scikit-image', 'lpips', 'DISTS-pytorch')
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': {name: importlib.metadata.version(name) for name in packages}}


def write_json(path, value):
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


def pairs(identity):
    with SPLIT.open(newline='', encoding='utf-8') as stream:
        values = list(csv.DictReader(stream))
    if len(values) != 100 or len({r['identity_id'] for r in values}) != 100 or any(r['split'] != 'development' for r in values):
        raise ValueError('Development manifest contract failed')
    if identity:
        values = [r for r in values if r['identity_id'] == identity]
        if len(values) != 1:
            raise ValueError('identity is absent from Development manifest')
    return values


def paths(method, ident):
    base = OUT / method.lower().replace('-', '_').replace('+', '_')
    return base / 'images' / f'{ident}.png', base / 'artifacts' / f'{ident}.json', base / 'evaluations' / f'{ident}.json'


def record_clean_skip(row, method, clean, thresholds, model_hashes, *, verify=False):
    ident = row['identity_id']
    png, meta, target = paths(method, ident)
    if png.exists() or meta.exists():
        raise ValueError(f'clean-ineligible identity has an attack artifact: {ident}')
    source = clean['models']['arcface']
    if source['clean_cosine'] is not None or not source.get('error'):
        raise ValueError(f'clean skip requires a recorded ArcFace detection error: {ident}')
    key = {'clean_eligibility_sha256': sha256(CLEAN_ELIGIBILITY),
           'reference_sha256': sha256(ROOT / row['reference_image']),
           'probe_sha256': sha256(ROOT / row['probe_image']),
           'thresholds': thresholds, 'model_hashes': model_hashes}
    expected = {'identity_id': ident, 'method': method, 'status': 'skipped',
                'reason': 'arcface_clean_detection_failed', 'key': key,
                'models': clean['models'], 'quality': None}
    if target.is_file():
        existing = json.loads(target.read_text())
        if any(existing.get(k) != v for k, v in expected.items()):
            raise ValueError(f'stale or invalid clean-skip record: {target}')
        return existing
    if verify:
        raise FileNotFoundError(f'missing clean-skip record: {target}')
    record = dict(expected, environment=environment(),
                  evaluated_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(target, record)
    return record


def provenance(row, method):
    import torch, cv2
    manifest = json.loads((ROOT / 'models/checkpoint_manifest.json').read_text())
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    return {'identity_id': row['identity_id'], 'reference_sha256': sha256(ROOT / row['reference_image']),
            'probe_sha256': sha256(ROOT / row['probe_image']), 'method': method,
            'mask': 'landmark_superpixel_mask' if '+Mask' in method else 'none',
            'arcface_checkpoint_sha256': manifest['arcface']['sha256'],
            'attack_config_sha256': config_hash(dict(CONFIG, method=method)),
            'preprocessing_version': 'lfw_v2_arcface_full_image', 'python_version': platform.python_version(),
            'torch_version': torch.__version__, 'dependency_versions': {'cv2': cv2.__version__, 'numpy': np.__version__},
            'git_commit': commit}


def check_image(png, meta, expected, probe, method):
    # The creation commit is retained in metadata, but a later analysis commit
    # must not invalidate otherwise identical saved attack inputs and bytes.
    validation = {key: value for key, value in expected.items() if key != 'git_commit'}
    record = validate(png, meta, validation, probe)
    adv = load_bgr(png)
    linf = float(np.abs(adv.astype(np.int16) - probe.astype(np.int16)).max() / 255)
    if linf > CONFIG['epsilon'] + 1/255 + 1e-12:
        raise ValueError(f'serialized L∞ exceeds quantization tolerance: {linf}')
    if '+Mask' in method:
        mask, _, _ = LandmarkSuperpixelMask().build(probe)
        if np.any(adv[mask == 0] != probe[mask == 0]):
            raise ValueError('perturbation outside final mask')
    return record, adv


def generate(row, method):
    from advface.models.insightface_app import create_face_app
    png, meta, _ = paths(method, row['identity_id'])
    probe = load_bgr(ROOT / row['probe_image'])
    expected = provenance(row, method)
    if png.exists() or meta.exists():
        check_image(png, meta, expected, probe, method)
        return False
    app = create_face_app()
    name = 'pgd_full' if method.startswith('PGD') else 'mi_fgsm'
    extra = {'tv_weight': 0.0}
    if '+Mask' in method:
        extra['constraint'] = 'landmark_superpixel_mask'
    config = AttackConfig(name=name, eps=.04, steps=200, alpha=.0002, momentum=1.0, extra=extra)
    started = time.monotonic()
    attack = ATTACKS[name].apply(probe, app=app, config=config)
    del app
    if float(attack.parameters.get('linf_tensor', 0)) > .04 + 1e-6:
        raise ValueError('tensor L∞ exceeds epsilon')
    png.parent.mkdir(parents=True, exist_ok=True)
    meta.parent.mkdir(parents=True, exist_ok=True)
    save(png, meta, attack.adversarial_bgr, probe, dict(expected, runtime_seconds=time.monotonic()-started,
                                                       tensor_linf=attack.parameters.get('linf_tensor')))
    check_image(png, meta, expected, probe, method)
    return True


def evaluate(row, method, thresholds, model_hashes, *, force=False):
    png, meta, target = paths(method, row['identity_id'])
    probe = load_bgr(ROOT / row['probe_image'])
    expected = provenance(row, method)
    artifact, adv = check_image(png, meta, expected, probe, method)
    key = {'artifact_sha256': sha256(meta), 'image_sha256': artifact['output_png_sha256'],
           'thresholds': thresholds, 'model_hashes': model_hashes,
           'reference_sha256': expected['reference_sha256'], 'probe_sha256': expected['probe_sha256']}
    if target.is_file() and not force:
        old = json.loads(target.read_text())
        if old.get('key') != key:
            raise ValueError(f'stale evaluation; use --reevaluate: {target}')
        return old, False
    if target.exists() and not force:
        raise ValueError(f'evaluation file exists but is invalid: {target}')
    ref = load_bgr(ROOT / row['reference_image'])
    scores = {}
    for name in MODELS:
        embedder = load_embedder(name)
        try:
            try:
                a = embedder.get_embedding(ref, f"{row['identity_id']}:A")
                b = embedder.get_embedding(probe, f"{row['identity_id']}:B")
                clean = float(cosine_similarity(a, b))
            except (OSError, ValueError, RuntimeError) as exc:
                scores[name] = {'clean_cosine': None, 'eligible': False, 'error': f'{type(exc).__name__}: {exc}'}
                continue
            try:
                x = embedder.get_embedding(adv, f"{row['identity_id']}:B_adv")
                attacked = float(cosine_similarity(a, x))
                scores[name] = {'clean_cosine': clean, 'eligible': clean >= thresholds[name],
                                'adv_cosine': attacked, 'success': clean >= thresholds[name] and attacked < thresholds[name]}
            except (OSError, ValueError, RuntimeError) as exc:
                scores[name] = {'clean_cosine': clean, 'eligible': clean >= thresholds[name],
                                'error': f'adversarial detection: {type(exc).__name__}: {exc}'}
        finally:
            del embedder
            gc.collect()
    from scripts.run_lfw_dev_v2 import metrics_from_png
    linf, ssim, lpips, dists = metrics_from_png(ROOT / row['probe_image'], png)
    quality = {'serialized_linf': linf, 'ssim': ssim, 'lpips': lpips, 'dists': dists}
    if abs(linf - artifact['serialized_linf']) > 1e-12:
        raise ValueError('perceptual input differs from validated PNG')
    result = {'identity_id': row['identity_id'], 'method': method, 'key': key, 'models': scores,
              'quality': quality, 'environment': environment(),
              'evaluated_at_utc': datetime.now(timezone.utc).isoformat()}
    write_json(target, result)
    return result, True


def summary(method, selected, thresholds):
    records = []
    for row in selected:
        target = paths(method, row['identity_id'])[2]
        if not target.is_file():
            continue
        record = json.loads(target.read_text())
        records.append(record)
    attacked = [r for r in records if r.get('status') != 'skipped']
    result = {'method': method, 'complete': len(records), 'expected': len(selected),
              'attacked': len(attacked), 'skipped': len(records) - len(attacked),
              'metrics': summarize(records, thresholds) if records else None,
              'quality_means': {k: float(np.mean([r['quality'][k] for r in attacked])) for k in ('serialized_linf', 'ssim', 'lpips', 'dists')} if attacked else None}
    write_json(OUT / method.lower().replace('-', '_').replace('+', '_') / 'summary.json', result)
    return result


def run(args, preflight):
    report = preflight()
    if report['decision'] != 'READY':
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print('Formal execute refused: preflight failed.')
        return 2
    selected = pairs(args.identity)
    methods = [args.method] if args.method else METHODS
    thresholds = load_v2_thresholds(ROOT)
    model_hashes = {name: sha256(path) for name, path in __import__('run_lfw_five_model_benchmark').MODELS.items()}
    eligibility = {r['identity_id']: r for r in json.loads(CLEAN_ELIGIBILITY.read_text())['rows']}
    if args.worker:
        if not args.identity or not args.method:
            raise ValueError('worker requires identity and method')
        generate(selected[0], args.method)
        return 0
    if args.verify and args.reevaluate is False:
        # Verify saved scores and checksums without loading victims; summary is recomputed.
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    run_metadata = OUT / 'run.json'
    state = {'config': CONFIG, 'split_sha256': sha256(SPLIT), 'thresholds': thresholds,
             'model_hashes': model_hashes, 'python': platform.python_version(),
             'platform': platform.platform(), 'environment': environment(),
             'git_commit': provenance(selected[0], methods[0])['git_commit']}
    if run_metadata.exists():
        previous = json.loads(run_metadata.read_text())
        descriptive = ('git_commit', 'environment', 'python', 'platform')
        if {k: v for k, v in previous.items() if k not in descriptive} != {k: v for k, v in state.items() if k not in descriptive}:
            raise ValueError('run metadata changed; refusing to reuse formal output')
    elif args.execute:
        write_json(run_metadata, state)
    elif not run_metadata.exists():
        raise ValueError('no saved run to verify')
    total = len(selected) * len(methods)
    done, started = 0, time.monotonic()
    log = OUT / 'run.log'
    for method in methods:
        for row in selected:
            ident = row['identity_id']
            current = f'{method}/{ident}'
            try:
                clean = eligibility[ident]
                if clean['models']['arcface']['clean_cosine'] is None:
                    record_clean_skip(row, method, clean, thresholds, model_hashes, verify=args.verify)
                elif args.execute:
                    command = [sys.executable, str(ROOT / 'scripts/run_lfw_five_model_benchmark.py'), '--worker', '--identity', ident, '--method', method]
                    png, meta, _ = paths(method, ident)
                    if not png.exists() and not meta.exists():
                        subprocess.run(command, cwd=ROOT, check=True, env=dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'))
                    else:
                        check_image(png, meta, provenance(row, method), load_bgr(ROOT / row['probe_image']), method)
                if clean['models']['arcface']['clean_cosine'] is None:
                    pass
                elif args.verify and not args.reevaluate:
                    png, meta, target = paths(method, ident)
                    check_image(png, meta, provenance(row, method), load_bgr(ROOT / row['probe_image']), method)
                    saved = json.loads(target.read_text())
                    if saved['key'] != {'artifact_sha256': sha256(meta), 'image_sha256': sha256(png), 'thresholds': thresholds,
                                        'model_hashes': model_hashes, 'reference_sha256': sha256(ROOT / row['reference_image']),
                                        'probe_sha256': sha256(ROOT / row['probe_image'])}:
                        raise ValueError('evaluation provenance mismatch')
                    summarize([saved], thresholds)
                else:
                    evaluate(row, method, thresholds, model_hashes, force=args.reevaluate)
                done += 1
                elapsed = time.monotonic() - started
                message = f'[{done}/{total}] {current} elapsed={elapsed:.0f}s ETA={elapsed/done*(total-done):.0f}s'
                print(message, flush=True)
                with log.open('a') as stream: stream.write(message + '\n')
            except (Exception, KeyboardInterrupt) as exc:
                message = f'ERROR {current}: {type(exc).__name__}: {exc}'
                print(message, file=sys.stderr, flush=True)
                with log.open('a') as stream: stream.write(message + '\n')
                raise
        result = summary(method, selected, thresholds)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0
