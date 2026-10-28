"""Build an offline professor-meeting HTML from persisted Yuan experiments."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import re
from html import escape
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def image_data(path):
    mime = 'image/jpeg' if path.suffix.lower() in ('.jpg', '.jpeg') else 'image/png'
    return f'data:{mime};base64,' + base64.b64encode(path.read_bytes()).decode('ascii')


def transcript_paragraphs(body):
    paragraphs = []
    for block in body.strip().split('\n\n'):
        block = block.strip()
        if not block:
            continue
        cls = ' class="speaker-cue"' if block.startswith('[') else ''
        paragraphs.append(f'<p{cls}>' + escape(block).replace('\n', ' ') + '</p>')
    return ''.join(paragraphs)


def attach_full_transcript(document, path):
    """Use the Markdown as the single source for all fourteen full notes."""
    if not path.is_file():
        return document, 0
    text = path.read_text(encoding='utf-8')
    sections = re.findall(r'^## 第 (\d+) 頁｜([^\n]+)\n([\s\S]*?)(?=^## |\Z)', text, re.M)
    assert [int(number) for number, _, _ in sections] == list(range(1, 15))
    assert len(re.findall(r'<aside class="notes">', document)) == 14
    notes = iter(sections)
    def replace_note(_):
        number, title, body = next(notes)
        return (f'<aside class="notes" data-full-transcript="true"><b>完整逐字稿｜第{number}頁・'
                + escape(title) + '</b>' + transcript_paragraphs(body) + '</aside>')
    document = re.sub(r'<aside class="notes">[\s\S]*?</aside>', replace_note, document)
    toc, blocks = [], []
    for block in text.strip().split('\n\n'):
        block = block.strip()
        heading = re.match(r'^(#{1,3}) (.+)$', block)
        if heading:
            level, title = len(heading[1]), heading[2]
            match = re.match(r'第 (\d+) 頁｜', title)
            anchor = 'slide-' + match[1] if match else 'section-' + str(len(blocks))
            blocks.append(f'<h{level} id="{anchor}">{escape(title)}</h{level}>')
            if level == 2:
                toc.append(f'<a href="#{anchor}">{escape(title)}</a>')
        else:
            blocks.append(transcript_paragraphs(block))
    reader = '''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>教授會議完整逐字稿</title>
<style>*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#edf2f2;color:#203341;font:19px/1.95 "Microsoft JhengHei","Segoe UI",sans-serif}header{background:#123e48;color:white;padding:30px max(24px,calc((100vw - 1180px)/2));}header a{color:#b9eee2}header p{margin:0;font-size:16px}button{font:inherit;padding:5px 13px;border:1px solid #9ac1c6;border-radius:8px;background:transparent;color:#fff;cursor:pointer}.layout{max-width:1220px;margin:26px auto;display:grid;grid-template-columns:260px 1fr;gap:24px;padding:0 20px}nav{position:sticky;top:20px;align-self:start;max-height:92vh;overflow:auto;background:#fff;padding:18px;border-radius:12px;border:1px solid #d9e3e6}nav a{display:block;padding:5px 0;color:#396874;text-decoration:none;font-size:14px;line-height:1.6}main{min-width:0;padding:34px 42px;background:#fff;border-radius:12px;border:1px solid #d9e3e6}h1{font-size:33px;line-height:1.4;margin:0 0 18px}h2{font-size:25px;line-height:1.5;margin:40px 0 18px;padding-top:18px;border-top:2px solid #dce9e9;scroll-margin-top:18px;color:#086e74}h3{font-size:20px;margin-top:28px;line-height:1.6}p{margin:0 0 17px}.speaker-cue{font-size:15px;line-height:1.7;color:#697b85;background:#f1f6f7;padding:9px 13px;border-left:3px solid #a8cbd0;border-radius:0 6px 6px 0}a{color:#087e82}@media(max-width:850px){.layout{display:block}nav{position:static;max-height:230px;margin-bottom:18px}main{padding:22px}body{font-size:18px}}@media print{@page{size:A4;margin:18mm}body{background:#fff;font-size:12pt;line-height:1.8}header,nav{display:none}.layout{display:block;margin:0;padding:0}main{border:0;padding:0}h2{break-before:page;font-size:18pt}h1{font-size:24pt}h3{break-after:avoid;font-size:14pt}p{orphans:3;widows:3}.speaker-cue{font-size:10pt}}</style></head><body>
<header><p>2026.10.28 教授會議 · 14頁完整主講＋10題備答</p><p>建議主講約15–20分鐘，依語速與圖表停留調整；方括號提示不念出來。</p><p><a href="report.html">開啟投影片</a> · <a href="transcript.md" download>下載 Markdown</a> · <button onclick="window.print()">列印逐字稿</button></p></header>
<div class="layout"><nav aria-label="逐頁目錄">__TOC__</nav><main>__BODY__</main></div></body></html>'''
    reader = reader.replace('__TOC__', ''.join(toc)).replace('__BODY__', ''.join(blocks))
    reader = reader.replace('</style>', 'main{overflow-wrap:anywhere}</style>')
    path.with_suffix('.html').write_text(reader, encoding='utf-8')
    return document, len(sections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yuan-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--layout', choices=('page', 'slides'), default='page')
    args = parser.parse_args()
    yuan = args.yuan_root.resolve()
    out = args.output or yuan / 'docs/reports/meeting_2026_10_28/report.html'
    specs = [
        ('self', 'MI-FGSM・self-reference', 'reference_20261013_145019_577894', 'self', False),
        ('pair', 'MI-FGSM・A-reference', 'reference_20261013_145019_577894', 'pair', False),
        ('mask', 'MI-FGSM＋Mask', 'reference_mask_20261013_154854_692028', 'pair', True),
        ('di', 'MI-FGSM＋DI', 'reference_di_20261014_153851_517219', 'pair', False),
        ('di_mask', 'MI-FGSM＋DI＋Mask', 'reference_di_mask_20261014_155754_945242', 'pair', True),
        ('ti', 'MI-FGSM＋DI＋TI', 'reference_di_ti_20261026_235225_545727', 'pair', False),
        ('ti_mask', 'MI-FGSM＋DI＋TI＋Mask', 'reference_di_ti_mask_20261026_235415_117069', 'pair', True),
    ]
    if args.layout == 'page':
        specs = [item for item in specs if item[0] in ('self', 'pair', 'di', 'ti')]
    first = load_json(yuan / 'results' / specs[0][2] / 'config.json')
    split = Path(first['split'])
    assert hashlib.sha256(split.read_bytes()).hexdigest() == first['split_sha256']
    yeah = split.parents[4]
    with split.open(encoding='utf-8', newline='') as handle:
        pairs = {r['identity_id']: r for r in csv.DictReader(handle)}
    methods, indexed = [], {}
    audited = 0
    for key, label, dirname, mode, masked in specs:
        run = yuan / 'results' / dirname
        config, summary = load_json(run / 'config.json'), load_json(run / 'summary.json')
        assert summary['completed_pairs'] == summary['requested'] == 20
        assert not summary['failures']
        for common in ('identities', 'eps', 'steps', 'alpha', 'momentum', 'seed', 'split_sha256', 'runtime'):
            assert config[common] == first[common], f'configuration differs: {common}'
        with (run / 'results.csv').open(encoding='utf-8', newline='') as handle:
            rows = [r for r in csv.DictReader(handle) if r['mode'] == mode]
        assert len(rows) == 20
        indexed[key] = {}
        values = []
        for row in rows:
            identity = row['identity_id']
            clean_path = yeah / pairs[identity]['probe_image']
            adv_path = Path(row['adversarial_image'])
            b, adv = cv2.imread(str(clean_path)), cv2.imread(str(adv_path))
            assert b is not None and adv is not None and b.shape == adv.shape
            difference = np.abs(adv.astype(np.int16) - b.astype(np.int16))
            actual_linf = float(difference.max() / 255)
            assert actual_linf <= config['eps'] + 1 / 255 + 1e-6
            assert abs(actual_linf - float(row['linf_serialized'])) < 1e-8
            assert float(row['linf_tensor']) <= config['eps'] + 1e-6
            if masked:
                mask = cv2.imread(row['mask_image'], cv2.IMREAD_GRAYSCALE)
                assert mask is not None and mask.shape == b.shape[:2]
                outside = difference[mask == 0]
                assert outside.size == 0 or outside.max() == 0
            metric = {
                'identity': identity, 'sourceClean': float(row['source_clean_cosine']),
                'victimClean': float(row['victim_clean_cosine']),
                'sourceAdv': float(row['source_adv_cosine']), 'victimAdv': float(row['victim_adv_cosine']),
                'sourceValid': row['source_clean_valid'] == 'True',
                'victimValid': row['victim_clean_valid'] == 'True',
                'sourceSuccess': row['source_boundary_crossed'] == 'True',
                'victimSuccess': row['victim_boundary_crossed'] == 'True',
                'ssim': float(structural_similarity(cv2.cvtColor(b, cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(adv, cv2.COLOR_BGR2RGB), channel_axis=-1, data_range=255)),
                'mae': float(difference.mean()), 'linf': actual_linf,
            }
            indexed[key][identity] = {'row': row, 'metric': metric, 'path': adv_path}
            values.append(metric)
            audited += 1
        src_success = sum(r['sourceSuccess'] for r in values)
        vic_success = sum(r['victimSuccess'] for r in values)
        src_valid = sum(r['sourceValid'] for r in values)
        vic_valid = sum(r['victimValid'] for r in values)
        assert src_success == summary[mode]['source']['successes']
        assert vic_success == summary[mode]['victim']['successes']
        methods.append({'key': key, 'label': label, 'run': dirname, 'mode': mode, 'masked': masked,
            'n': 20, 'sourceSuccess': src_success, 'sourceEligible': src_valid,
            'victimSuccess': vic_success, 'victimEligible': vic_valid,
            'ssim': float(np.mean([r['ssim'] for r in values])),
            'mae': float(np.mean([r['mae'] for r in values])),
            'maxLinf': max(r['linf'] for r in values)})
    # A fair comparison requires identical clean scores across every condition.
    for identity in first['identities']:
        for key in indexed:
            for field in ('sourceClean', 'victimClean'):
                assert abs(indexed[key][identity]['metric'][field] - indexed['pair'][identity]['metric'][field]) < 1e-5
    changes = {}
    for old, new in (('self', 'pair'), ('pair', 'di'), ('di', 'ti'),
                     ('mask', 'di_mask'), ('di_mask', 'ti_mask')):
        if old not in indexed or new not in indexed:
            continue
        changes[f'{old}:{new}'] = {
            'gained': [i for i in first['identities'] if indexed[new][i]['metric']['victimSuccess']
                       and not indexed[old][i]['metric']['victimSuccess']],
            'lost': [i for i in first['identities'] if indexed[old][i]['metric']['victimSuccess']
                     and not indexed[new][i]['metric']['victimSuccess']],
        }
    cases = {}
    for identity in ('Meryl_Streep', 'Michelle_Collins'):
        pair = pairs[identity]
        cases[identity] = {
            'name': identity.replace('_', ' '),
            'a': image_data(yeah / pair['reference_image']),
            'b': image_data(yeah / pair['probe_image']),
            'methods': {key: {'image': image_data(indexed[key][identity]['path']),
                              **indexed[key][identity]['metric']} for key in indexed},
        }
    masked_run = yuan / 'results/reference_di_ti_mask_20261026_235415_117069'
    mask_images = {key: image_data(masked_run / filename) for key, filename in {
        'indices': '000_landmark_indices.png', 'initial': '000_initial_mask.png',
        'final': '000_mask.png', 'overlay': '000_mask_overlay.png'}.items()} if args.layout == 'slides' else {}
    source_threshold = load_json(Path(first['source_calibration']))['far_1e-2']['threshold']
    victim_threshold = load_json(Path(first['victim_calibration']))['far_1e-2']['threshold']
    data = {
        'date': '2026-10-28', 'title': '對準驗證使用的 A、DI 與 TI',
        'n': 20, 'eps': first['eps'], 'alpha': first['alpha'], 'steps': first['steps'],
        'momentum': first['momentum'], 'seed': first['seed'],
        'sourceThreshold': source_threshold, 'victimThreshold': victim_threshold,
        'runtime': first['runtime'], 'splitSha256': first['split_sha256'],
        'identities': first['identities'], 'methods': methods, 'changes': changes,
        'matrix': [{'identity': i, 'name': i.replace('_', ' '),
                    **{key: indexed[key][i]['metric'] for key in indexed}} for i in first['identities']],
        'cases': cases, 'maskImages': mask_images,
        'auditedPngCount': audited,
        'sourceFiles': [f'results/{dirname}/config.json' for _, _, dirname, _, _ in specs],
    }
    template_name = 'professor_web_template.html' if args.layout == 'page' else 'professor_meeting_template.html'
    template = Path(__file__).with_name(template_name).read_text(encoding='utf-8')
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    html = template.replace('__MEETING_DATA__', payload)
    assert '__MEETING_DATA__' not in html
    out.parent.mkdir(parents=True, exist_ok=True)
    full_notes = 0
    if args.layout == 'slides':
        html, full_notes = attach_full_transcript(html, out.parent / 'transcript.md')
    out.write_text(html, encoding='utf-8')
    # A small image-free snapshot makes the numbers easy to review separately.
    snapshot = {k: v for k, v in data.items() if k not in ('cases', 'maskImages')}
    (out.parent / 'meeting_data.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'report': str(out), 'bytes': out.stat().st_size,
                      'conditions': len(methods), 'audited_pngs': audited,
                      'full_transcript_slides': full_notes,
                      'layout': args.layout,
                      'all_results_complete': True}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
