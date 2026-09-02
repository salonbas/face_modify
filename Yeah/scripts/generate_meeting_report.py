#!/usr/bin/env python3
"""Build the 2026-09-02 meeting report from existing canonical artifacts only."""

from __future__ import annotations

import csv
import html
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/reports/meeting_2026_09_02"
ASSETS = OUT / "assets"


def read_json(relative: str) -> dict:
    with (ROOT / relative).open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(value: float | str, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def pct(value: float | str, digits: int = 1) -> str:
    return f"{float(value) * 100:.{digits}f}%"


def name(value: str) -> str:
    return value.replace("_", " ")


def histogram_svg(rows: list[dict[str, str]]) -> str:
    """Pre-render a simple score distribution from the canonical pair-score CSV."""
    bins = 24
    low, high = -0.20, 1.00
    genuine = [0] * bins
    impostor = [0] * bins
    for row in rows:
        if row["split"] != "dev_train" or not row["cosine_similarity"]:
            continue
        score = float(row["cosine_similarity"])
        index = max(0, min(bins - 1, int((score - low) / (high - low) * bins)))
        (genuine if row["same_identity"].lower() == "true" else impostor)[index] += 1
    maximum = max(*genuine, *impostor)
    parts = ['<svg viewBox="0 0 760 245" role="img" aria-label="ArcFace genuine and impostor cosine distributions">']
    parts.append('<rect width="760" height="245" fill="#fbfcfe"/>')
    for tick in range(5):
        y = 190 - tick * 35
        parts.append(f'<line x1="52" x2="740" y1="{y}" y2="{y}" stroke="#dbe3eb"/>')
    width = 688 / bins
    for i, (g, im) in enumerate(zip(genuine, impostor)):
        x = 52 + i * width
        gh = 170 * g / maximum
        ih = 170 * im / maximum
        parts.append(f'<rect x="{x + 1:.1f}" y="{190 - gh:.1f}" width="{width / 2 - 2:.1f}" height="{gh:.1f}" fill="#2979a9" opacity=".85"/>')
        parts.append(f'<rect x="{x + width / 2 + 1:.1f}" y="{190 - ih:.1f}" width="{width / 2 - 2:.1f}" height="{ih:.1f}" fill="#d1654f" opacity=".82"/>')
    for value in (-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0):
        x = 52 + (value - low) / (high - low) * 688
        parts.append(f'<text x="{x:.1f}" y="211" text-anchor="middle" font-size="11" fill="#627080">{value:.1f}</text>')
    parts.extend(['<text x="52" y="228" font-size="12" fill="#627080">Cosine similarity</text>', '<rect x="520" y="20" width="12" height="12" fill="#2979a9"/><text x="538" y="31" font-size="12" fill="#334155">Genuine</text>', '<rect x="630" y="20" width="12" height="12" fill="#d1654f"/><text x="648" y="31" font-size="12" fill="#334155">Impostor</text>', '</svg>'])
    return "".join(parts)


def row(cells: list[str]) -> str:
    return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


def main() -> None:
    arc = read_json("results/calibration/arcface_lfw_v1/summary.json")
    face = read_json("results/calibration/facenet_lfw_v1/summary.json")
    white = read_json("results/robustness/lfw_verification_pgd_v1/summary.json")
    pgd = read_json("results/transfer/pgd_arcface_to_facenet_verification_v1/summary.json")
    mi = read_json("results/transfer/mi_fgsm_arcface_to_facenet_verification_v1/summary.json")
    split = read_json("data/datasets/lfw/evaluation_splits/split_config.json")
    white_rows = read_csv("results/robustness/lfw_verification_pgd_v1/results.csv")
    pgd_rows = read_csv("results/transfer/pgd_arcface_to_facenet_verification_v1/results.csv")
    mi_rows = read_csv("results/transfer/mi_fgsm_arcface_to_facenet_verification_v1/results.csv")
    compare_rows = read_csv("results/transfer/mi_fgsm_arcface_to_facenet_verification_v1/comparison_pgd_vs_mi_fgsm.csv")
    pair_scores = read_csv("results/calibration/arcface_lfw_v1/pair_scores.csv")
    historical = read_csv("results/pgd/pgdfull_musk_v2/pgd_full_cosine_metrics.csv")
    historical_004 = next(item for item in historical if item["eps"] == "0.040000")

    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    copied_assets = []
    for source, target in [
        ("results/pgd/pgdfull_musk_v2/base_musk1.png", "base_musk1.png"),
        ("results/pgd/pgdfull_musk_v2/pgdfull_musk1_eps_0.040.png", "pgdfull_musk1_eps_0.040.png"),
    ]:
        src = ROOT / source
        if src.is_file():
            shutil.copy2(src, ASSETS / target)
            copied_assets.append(target)

    snapshot = {
        "report_title": "Research Progress — Threshold Calibration → Transfer Verification",
        "date": "2026-09-02",
        "evidence_policy": "Read-only snapshot of existing canonical artifacts; no attacks, embeddings, or calibrations were run for this report.",
        "sources": [
            "results/calibration/arcface_lfw_v1/{summary.json,thresholds.json,pair_scores.csv,roc.csv}",
            "results/calibration/facenet_lfw_v1/{summary.json,thresholds.json,pair_scores.csv,roc.csv}",
            "results/robustness/lfw_verification_pgd_v1/{summary.json,results.csv}",
            "results/transfer/pgd_arcface_to_facenet_verification_v1/{summary.json,results.csv}",
            "results/transfer/mi_fgsm_arcface_to_facenet_verification_v1/{summary.json,results.csv,comparison_pgd_vs_mi_fgsm.csv}",
            "results/pgd/pgdfull_musk_v2/pgd_full_cosine_metrics.csv",
            "data/datasets/lfw/evaluation_splits/split_config.json",
        ],
        "arcface_calibration": arc,
        "facenet_calibration": face,
        "white_box_pgd": white,
        "pgd_transfer": pgd,
        "mi_fgsm_transfer": mi,
        "attack_split": split,
        "historical_pgd_full_eps_004": historical_004,
        "white_box_rows": white_rows,
        "pgd_transfer_rows": pgd_rows,
        "mi_fgsm_transfer_rows": mi_rows,
        "comparison_rows": compare_rows,
        "copied_assets": copied_assets,
    }
    (OUT / "report_data.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    aop = arc["operating_points"]
    fop = face["operating_points"]
    hist = histogram_svg(pair_scores)
    white_table = "".join(row([name(x["identity_id"]), f(x["clean_cosine_A_B"]), f(x["self_cosine_B_Bmodified"]), f(x["modified_cosine_A_Bmodified"]), f(x["primary_threshold"]), '<span class="badge red">Crossed</span>']) for x in white_rows)
    transfer_table = "".join(row([name(x["identity_id"]), f(x["source_adv_A_Badv"]), '<span class="badge red">Yes</span>', f(x["victim_clean_A_B"]), f(x["victim_self_B_Badv"]), f(x["victim_adv_A_Badv"]), '<span class="badge gray">No</span>']) for x in pgd_rows)
    comparison_table = "".join(row([name(x["identity_id"]), f(x["pgd_victim_A_B_adv"]), f(x["mi_fgsm_victim_A_B_adv"]), '<span class="badge red">Yes</span>' if x["mi_fgsm_victim_boundary_crossed"] == "True" else '<span class="badge gray">No</span>']) for x in compare_rows)

    html_document = f'''<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Research Progress — 2026-09-02</title>
<style>
:root{{--ink:#18212f;--muted:#64748b;--line:#dbe3eb;--paper:#fff;--bg:#f5f7fa;--blue:#176b9b;--blue-soft:#eaf4fa;--green:#246b4d;--green-soft:#e9f6ef;--red:#a74336;--red-soft:#fcebea;--gray-soft:#edf1f5}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}.shell{{max-width:1180px;margin:auto;display:grid;grid-template-columns:230px minmax(0,1fr);gap:28px;padding:26px}}aside{{position:sticky;top:18px;height:max-content;padding:18px;background:#fff;border:1px solid var(--line);border-radius:12px}}aside strong{{display:block;font-size:14px;margin-bottom:10px}}aside a{{display:block;color:#526273;text-decoration:none;padding:5px 0;font-size:13px}}aside a:hover{{color:var(--blue)}}main{{min-width:0}}section{{background:var(--paper);border:1px solid var(--line);border-radius:12px;margin-bottom:18px;padding:30px;box-shadow:0 1px 2px #0f172a08}}h1{{font-size:34px;line-height:1.17;margin:0 0 10px}}h2{{font-size:23px;line-height:1.25;margin:0 0 16px}}h3{{font-size:17px;margin:18px 0 8px}}p{{margin:9px 0}}.eyebrow{{font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--blue);font-weight:750}}.lede{{font-size:18px;color:#465568;max-width:850px}}.flow{{display:flex;align-items:stretch;gap:8px;flex-wrap:wrap;margin:20px 0}}.flow span{{background:var(--blue-soft);border:1px solid #c6e3f1;color:#145477;border-radius:8px;padding:8px 10px;font-size:13px;font-weight:650}}.flow b{{color:#8da0af;align-self:center}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}.grid3{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}.card,.callout{{border:1px solid var(--line);border-radius:9px;padding:16px}}.card b{{font-size:22px;display:block;color:var(--blue)}}.callout{{background:var(--blue-soft);border-color:#c6e3f1}}.callout.red{{background:var(--red-soft);border-color:#efc7c1}}.note{{font-size:13px;color:var(--muted)}}.warning{{color:#8c362c;font-weight:650}}table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:12px}}th,td{{padding:9px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{color:#516173;background:#f8fafc;font-weight:700}}.badge{{display:inline-block;border-radius:99px;padding:2px 8px;font-size:12px;font-weight:700}}.badge.red{{background:var(--red-soft);color:var(--red)}}.badge.green{{background:var(--green-soft);color:var(--green)}}.badge.gray{{background:var(--gray-soft);color:#526273}}.numberline{{position:relative;height:86px;margin:24px 12px 4px;border-bottom:3px solid #9db0be}}.mark{{position:absolute;bottom:-4px;width:2px;height:24px;background:var(--blue)}}.mark span{{position:absolute;bottom:29px;transform:translateX(-40%);white-space:nowrap;font-size:12px;color:#385167}}.mark small{{position:absolute;top:29px;transform:translateX(-38%);white-space:nowrap;color:var(--muted)}}.protocol{{display:grid;grid-template-columns:1fr 42px 1fr 42px 1fr;align-items:center;text-align:center;margin:20px 0}}.protocol .node{{padding:18px 8px;background:#f8fafc;border:1px solid var(--line);border-radius:10px;font-weight:650}}.protocol .arrow{{font-size:26px;color:var(--blue)}}.two-col{{columns:2;column-gap:30px;padding-left:20px}}.two-col li{{margin:7px 0}}.images{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}figure{{margin:0;border:1px solid var(--line);border-radius:10px;padding:10px;background:#fafcff}}figure img{{width:100%;height:270px;object-fit:contain;background:#e9edf1;border-radius:5px}}figcaption{{font-size:13px;color:var(--muted);padding:7px 3px 0}}.quote{{border-left:4px solid var(--blue);padding:10px 15px;background:#f8fafc;margin:14px 0;color:#31465b}}.check{{list-style:none;padding:0;margin:0}}.check li{{padding:5px 0}}.done{{color:var(--green)}}.todo{{color:#667789}}footer{{padding:10px 2px 28px;color:var(--muted);font-size:12px}}@media(max-width:850px){{.shell{{display:block;padding:14px}}aside{{position:static;margin-bottom:16px}}.grid,.grid3{{grid-template-columns:1fr}}.protocol{{grid-template-columns:1fr;gap:8px}}.protocol .arrow{{transform:rotate(90deg)}}.two-col{{columns:1}}section{{padding:21px}}}}
</style></head><body><div class="shell"><aside><strong>Meeting report<br><span class="note">02 Sep 2026</span></strong><a href="#summary">1. Executive Summary</a><a href="#why">2. Why 0.4?</a><a href="#arcface">3. ArcFace Calibration</a><a href="#historical">4. Historical 0.347</a><a href="#protocol">5. Verification Protocol</a><a href="#whitebox">6. PGD White-box</a><a href="#facenet">7. FaceNet Calibration</a><a href="#transfer">8. PGD Transfer</a><a href="#mi">9. MI-FGSM Comparison</a><a href="#conclusions">10. Current Conclusions</a><a href="#talking">11. Meeting Talking Points</a><a href="#appendix">Appendix</a></aside><main>
<section id="summary"><div class="eyebrow">Professor meeting · research progress</div><h1>Threshold Calibration → Transfer Verification</h1><p class="lede">A read-only synthesis of the evidence produced since the question: “Why is the cosine threshold 0.4?”</p><div class="flow"><span>0.4 question</span><b>↓</b><span>authority audit</span><b>↓</b><span>LFW calibration</span><b>↓</b><span>A / B / B<sub>adv</sub> protocol</span><b>↓</b><span>PGD white-box</span><b>↓</b><span>FaceNet calibration</span><b>↓</b><span>PGD baseline</span><b>↓</b><span>MI-FGSM comparison</span></div><div class="grid3"><div class="card"><b>0.140721</b>ArcFace calibrated FAR = 10<sup>−2</sup> threshold</div><div class="card"><b>0.404789</b>FaceNet calibrated FAR = 10<sup>−2</sup> threshold</div><div class="card"><b>5 / 5 → 0 / 5</b>PGD: ArcFace crossings → FaceNet crossings</div><div class="card"><b>5 / 5 → 1 / 5</b>MI-FGSM: ArcFace crossings → FaceNet crossings</div><div class="card"><b>0.4 ≠ universal</b>Implementation reference, not ArcFace’s universal official threshold</div><div class="card"><b>Development only</b>Observed on five development identities; not success-rate estimates</div></div></section>
<section id="why"><div class="eyebrow">Threshold authority audit</div><h2>Why was 0.4 questioned?</h2><p>The previous value is now described precisely: it was a <b>provisional / implementation-reference threshold</b>, not the primary empirical decision boundary for this study.</p><table><thead><tr><th>Source</th><th>0.4 status</th></tr></thead><tbody>{row(['ArcFace original paper','<b>NOT defined</b> as a universal cosine threshold'])}{row(['InsightFace service / guidance','Operational example or default exists'])}{row(['Our old project','Provisional threshold'])}{row(['Current research','Replaced by model-specific empirical calibration'])}</tbody></table><p class="note">The ArcFace paper, InsightFace implementation guidance, and empirical LFW calibration answer different questions; they should not be conflated.</p></section>
<section id="arcface"><div class="eyebrow">LFW calibration</div><h2>ArcFace calibration setup and score distribution</h2><p>LFW contains 13,233 funneled images from 5,749 identities, evaluated with official verification pairs. Thresholds were selected using Development Train and checked on held-out Development Test.</p><div class="grid"><div class="card"><b>Train</b>1,093 genuine + 1,093 impostor valid pairs<br><span class="note">14 failures</span></div><div class="card"><b>Test</b>492 genuine + 497 impostor valid pairs<br><span class="note">11 failures</span></div></div><h3>Genuine vs impostor cosine distribution</h3>{hist}<div class="grid"><div class="callout"><b>Cosine</b><br>Genuine mean {f(arc['distributions']['train_genuine_cosine']['mean'],4)} · median {f(arc['distributions']['train_genuine_cosine']['median'],4)}<br>Impostor mean {f(arc['distributions']['train_impostor_cosine']['mean'],4)} · median {f(arc['distributions']['train_impostor_cosine']['median'],4)}</div><div class="callout"><b>Euclidean</b><br>Genuine mean {f(arc['distributions']['train_genuine_euclidean']['mean'],4)}<br>Impostor mean {f(arc['distributions']['train_impostor_euclidean']['mean'],4)}</div></div><h3>Calibrated ArcFace operating points</h3><table><thead><tr><th>Operating point</th><th>Threshold</th><th>Dev FAR</th><th>Dev TAR</th><th>Test FAR</th><th>Test TAR</th></tr></thead><tbody>{row(['EER',f(aop['eer']['threshold']),'2.10%','97.90%','2.62%','97.15%'])}{row(['FAR 10⁻² — primary',f(aop['far_1e-2']['threshold']),pct(aop['far_1e-2']['development']['far']),pct(aop['far_1e-2']['development']['tar']),pct(aop['far_1e-2']['held_out_test']['far']),pct(aop['far_1e-2']['held_out_test']['tar'])])}{row(['FAR 10⁻³ — exploratory',f(aop['far_1e-3']['threshold']),pct(aop['far_1e-3']['development']['far']),pct(aop['far_1e-3']['development']['tar']),pct(aop['far_1e-3']['held_out_test']['far']),pct(aop['far_1e-3']['held_out_test']['tar'])])}</tbody></table><p class="note">FAR 10⁻³ is exploratory: 1,093 train impostors give a resolution of ≈ 9.15×10⁻⁴.</p><h3>What does legacy 0.4 mean now?</h3><p>At 0.4, ArcFace has Dev FAR 0%, TAR 97.2553%; Test FAR 0%, TAR 96.1382%. It is a much stricter threshold than the empirical LFW operating points—not “wrong,” but not this study’s primary boundary.</p><div class="numberline"><div class="mark" style="left:16%"><span>0.1266</span><small>EER</small></div><div class="mark" style="left:19%"><span>0.1407</span><small>FAR 1%</small></div><div class="mark" style="left:27%"><span>0.1884</span><small>FAR 0.1%</small></div><div class="mark" style="left:79%;background:#8c362c"><span>0.400</span><small>Legacy</small></div></div></section>
<section id="historical"><div class="eyebrow">Historical evidence · easy reference</div><h2>Historical PGD Result: Why was it 0.347?</h2><div class="callout red"><b>Historical artifact: <code>results/pgd/pgdfull_musk_v2</code></b><br>PGD Full · ε = 0.04 · 200 steps · historical self cosine = <b>{historical_004['cosine_similarity']}</b></div><div class="images"><figure><img src="assets/base_musk1.png" alt="Historical base musk image"><figcaption>Historical single-image PGD Full experiment: base image.</figcaption></figure><figure><img src="assets/pgdfull_musk1_eps_0.040.png" alt="Historical PGD adversarial musk image"><figcaption>PGD Full adversarial image at ε = 0.04.</figcaption></figure></div><p><b>Semantics:</b> 0.34716704 is <b>self similarity</b> (B ↔ B<sub>modified</sub>), not the new A ↔ B<sub>adv</sub> verification metric. The parity audit found the PGD function, objective, ε, steps, α, random start, preprocessing, evaluator, and embedding semantics unchanged.</p><p>The historical input was raw musk1 (667×500); the current inputs are 250×250 LFW funneled samples. The difference is consistent with sample/dataset-dependent attackability, <b>not</b> evidence of PGD implementation drift. It does not identify funneling, alignment, pose, scale, face occupancy, or any one cause. Although 0.347 is above 0.1407 numerically, it must not be retroactively labelled a formal verification failure: no independent A/B pair existed.</p></section>
<section id="protocol"><div class="eyebrow">Methodology</div><h2>New A / B / B<sub>adv</sub> verification protocol</h2><div class="protocol"><div class="node">Identity X<br><b>A</b> = reference</div><div class="arrow">→</div><div class="node"><b>B</b> = independent clean probe<br><span class="note">measure A ↔ B</span></div><div class="arrow">→</div><div class="node"><b>B<sub>adv</sub></b> = attack B only<br><span class="note">measure B ↔ B<sub>adv</sub> and A ↔ B<sub>adv</sub></span></div></div><div class="quote"><b>Primary success rule:</b> clean A–B ≥ that model’s calibrated threshold <i>and</i> A–B<sub>adv</sub> &lt; that model’s calibrated threshold.</div><h3>Dataset isolation</h3><div class="grid3"><div class="card"><b>3,077</b>identities used by calibration</div><div class="card"><b>2,672</b>completely calibration-unused identities</div><div class="card"><b>328</b>unused identities with ≥2 images</div></div><p>Frozen attack split: <b>20 development</b> · <b>100 held-out test</b> · <b>208 reserve</b>. Attack identities are identity-disjoint from threshold calibration. The held-out 100 have not been used.</p></section>
<section id="whitebox"><div class="eyebrow">Capability check · five development identities</div><h2>PGD white-box verification on ArcFace</h2><p>Primary calibrated threshold: <b>{f(white['calibration']['primary_threshold'])}</b>. All five clean pairs were valid, and all five crossed the ArcFace boundary. This is a <b>capability check</b>, not “PGD ASR = 100%.”</p><table><thead><tr><th>Identity</th><th>Clean A–B</th><th>B–B<sub>adv</sub></th><th>A–B<sub>adv</sub></th><th>ArcFace threshold</th><th>Boundary</th></tr></thead><tbody>{white_table}</tbody></table><p class="note">Observed on five development identities only — not a formal success-rate estimate.</p></section>
<section id="facenet"><div class="eyebrow">Victim-specific calibration</div><h2>FaceNet calibration</h2><p>ArcFace’s threshold cannot be applied to FaceNet. FaceNet was calibrated on the same LFW Development Train → Development Test protocol.</p><div class="grid"><div class="callout"><b>FAR 10⁻² primary threshold: {f(fop['far_1e-2']['threshold'])}</b><br>Dev: FAR 1.0%, TAR 94.7%<br>Test: FAR 1.8%, TAR 94.2%</div><div class="callout"><b>EER: {f(fop['eer']['threshold'])}</b><br>EER = 4.27%<br>FAR 10⁻³: {f(fop['far_1e-3']['threshold'])} <span class="badge gray">EXPLORATORY</span></div></div><p class="note">FAR 10⁻³ is exploratory because it is supported by 1,100 impostor pairs.</p></section>
<section id="transfer"><div class="eyebrow">ArcFace → FaceNet · PGD baseline</div><h2>PGD transfer verification</h2><p>Using the same five development identities, PGD crossed the source ArcFace boundary in <b>5 / 5</b>, while FaceNet crossed in <b>0 / 5</b>.</p><table><thead><tr><th>Identity</th><th>ArcFace A–B<sub>adv</sub></th><th>ArcFace crossed?</th><th>FaceNet A–B</th><th>FaceNet B–B<sub>adv</sub></th><th>FaceNet A–B<sub>adv</sub></th><th>FaceNet crossed?</th></tr></thead><tbody>{transfer_table}</tbody></table><div class="grid3"><div class="card"><b>{f(pgd['aggregate']['mean_victim_self_B_Badv_cosine'])}</b>mean FaceNet B–B<sub>adv</sub></div><div class="card"><b>{f(pgd['aggregate']['mean_victim_self_cosine_drop'])}</b>mean FaceNet self drop</div><div class="card"><b>{f(pgd['aggregate']['mean_victim_verification_drop'])}</b>mean FaceNet verification drop</div></div><div class="quote">PGD shows representation transfer, but no observed FaceNet verification-boundary transfer on these five development identities.</div></section>
<section id="mi"><div class="eyebrow">Development comparison</div><h2>Why add MI-FGSM? PGD vs MI-FGSM</h2><p>MI-FGSM (Momentum Iterative Fast Gradient Sign Method; Dong et al., CVPR 2018) accumulates normalized gradients through momentum, a design intended in part to improve transferability. Conditions were held constant: ε = 0.04, 200 steps, α = 0.0002, random start = false, same five identities, ArcFace surrogate, FaceNet victim, and calibrated thresholds. MI-FGSM additionally uses momentum = 1.0 and per-image L1 gradient normalization.</p><div class="protocol"><div class="node">PGD<br><span class="note">g₁ → g₂ → g₃</span></div><div class="arrow">vs</div><div class="node">MI-FGSM<br><span class="note">g₁ → g₁+g₂ → g₁+g₂+g₃</span></div></div><table><thead><tr><th>Metric</th><th>PGD</th><th>MI-FGSM</th></tr></thead><tbody>{row(['ArcFace source crossing','5 / 5','5 / 5'])}{row(['Mean ArcFace A–Badv','−0.300323','−0.050581'])}{row(['Mean FaceNet B–Badv','0.763469','0.652908'])}{row(['Mean FaceNet self drop','0.236531','0.347092'])}{row(['Mean FaceNet A–Badv','0.615620','0.564636'])}{row(['Mean FaceNet verification drop','0.165169','0.216153'])}{row(['FaceNet boundary crossing','0 / 5','1 / 5'])}</tbody></table><h3>Per-sample comparison</h3><table><thead><tr><th>Identity</th><th>PGD FaceNet A–B<sub>adv</sub></th><th>MI-FGSM FaceNet A–B<sub>adv</sub></th><th>MI-FGSM crossed FaceNet?</th></tr></thead><tbody>{comparison_table}</tbody></table><div class="callout red"><b>Cyndi Thompson — first observed cross-model verification-boundary transfer</b><br>PGD: 0.504656 &gt; FaceNet threshold 0.404789 (no crossing). MI-FGSM: 0.365727 &lt; 0.404789 (crossed). <span class="warning">One development sample, not a formal success rate.</span></div></section>
<section id="conclusions"><div class="eyebrow">Interpretation and limits</div><h2>Observed source / transfer tradeoff</h2><div class="callout"><b>PGD</b> had stronger ArcFace source degradation. <b>MI-FGSM</b> had weaker source degradation, but stronger observed FaceNet representation movement and verification degradation. On these five development identities: PGD 0 / 5 victim crossings; MI-FGSM 1 / 5.</div><p class="quote">Momentum showed a stronger observed cross-model verification effect on these five development identities.</p><div class="grid"><div><h3>Supported</h3><ul class="check"><li class="done">✓ 0.4 is not a universal ArcFace threshold.</li><li class="done">✓ Model-specific calibration gives materially different thresholds.</li><li class="done">✓ PGD crosses ArcFace’s boundary in this five-sample capability check.</li><li class="done">✓ PGD affects FaceNet representation but did not cross its boundary here.</li><li class="done">✓ MI-FGSM showed stronger observed FaceNet degradation; one sample crossed.</li></ul></div><div><h3>Not yet supported</h3><ul class="check"><li class="todo">○ PGD has a 100% white-box ASR.</li><li class="todo">○ MI-FGSM has a 20% transfer ASR.</li><li class="todo">○ MI-FGSM is statistically better than PGD.</li><li class="todo">○ LFW funneling causes stronger attacks.</li><li class="todo">○ Results generalize to other models or real-world systems.</li></ul></div></div><h3>Research status now</h3><ul class="two-col"><li class="done">✓ 0.4 authority audit</li><li class="done">✓ LFW canonical dataset and empirical calibration</li><li class="done">✓ Identity-disjoint A/B/Badv protocol</li><li class="done">✓ PGD white-box check and historical parity audit</li><li class="done">✓ FaceNet calibration and PGD transfer baseline</li><li class="done">✓ MI-FGSM calibrated comparison</li><li class="todo">○ Held-out 100-identity evaluation</li><li class="todo">○ Additional transfer attacks and formal statistical comparison</li><li class="todo">○ Targeted attack, mapper integration, final evaluation design</li></ul></section>
<section id="talking"><div class="eyebrow">Meeting aid</div><h2>60-second progress summary</h2><p class="quote">老師上次問為什麼 threshold 用 0.4。我回去查證後確認，0.4 不是 ArcFace 的 universal official threshold，比較像 InsightFace 的 implementation/reference 值。所以我用 LFW 的 development split 分別校準 ArcFace 和 FaceNet，得到 FAR 1% 的門檻是 0.140721 和 0.404789。接著我把攻擊評估改成 A/B/B_adv 的 verification protocol。PGD 在五個 development identity 上都能讓 ArcFace 跨過門檻，但轉到 FaceNet 是 0 個。加入 MI-FGSM 後，source model 的效果沒有 PGD 強，但 FaceNet 的變化更大，第一次看到 1 個跨過 FaceNet 門檻。這些都還是 development observation；held-out 的 100 identities 尚未使用，會等方法確定後才正式評估。</p><h3>If professor asks about 0.347</h3><p class="quote">舊的 0.347 和新的 B–B_adv 都是 self-similarity，所以可以直接比較。audit 顯示 PGD 的數學和 evaluator 沒有因 refactor 改變；主要差別是 input condition：舊的是 raw single image，現在是不同的 LFW funneled samples。現在只能說 attackability 有 sample/dataset dependency，不能指定是 funneling 造成。最重要的是，0.347 是 historical self-similarity，不是新的 A–B_adv verification metric，所以不能拿它當正式 verification success 或 failure。</p></section>
<section id="appendix"><div class="eyebrow">Appendix</div><h2>Reproducibility and scope</h2><p>Python 3.12.3 · repo-local <code>.venv</code> · <code>requirements.txt</code> canonicalized · latest recorded tests: 58 passed. L∞ tensor constraint is recorded separately from serialized quantization.</p><p><b>LFW:</b> 13,233 images, 5,749 identities. <b>Attack split:</b> 20 development, 100 held-out, 208 reserve. FAR 10⁻³ thresholds are exploratory because of limited impostor support.</p><p class="note">Evidence snapshot: <code>report_data.json</code>. This static report contains no browser-side metric computation and requires no network connection.</p></section>
<footer>Generated from existing canonical artifacts only · {html.escape(snapshot['evidence_policy'])}</footer></main></div></body></html>'''
    (OUT / "report.html").write_text(html_document, encoding="utf-8")


if __name__ == "__main__":
    main()
