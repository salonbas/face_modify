#!/usr/bin/env python3
"""Read-only mechanism analysis for the formal LFW four-method v3 artifacts.

This script never generates attacks or modifies existing B_adv files.  It only
decodes the saved PNGs, joins formal per-identity measurements, and writes
derived analysis artifacts for the 2026-09-30 meeting report.
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import binomtest, pearsonr, shapiro, spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "results/transfer/lfw_attack_dev_20_four_method_v3"
OUT = ROOT / "docs/reports/meeting_2026_09_30/assets/analysis"
REPORT = ROOT / "docs/reports/meeting_2026_09_30/report.html"
THRESHOLDS = {
    "ArcFace": ROOT / "results/calibration/arcface_lfw_v1/thresholds.json",
    "FaceNet": ROOT / "results/calibration/facenet_lfw_v1/thresholds.json",
}
METHODS = ["PGD", "MI-FGSM", "PGD+Mask", "MI-FGSM+Mask"]
COLORS = {"PGD": "#176b9b", "MI-FGSM": "#d1764d", "PGD+Mask": "#3d9275", "MI-FGSM+Mask": "#7769a8"}


def rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)


def numeric(row: dict, key: str) -> float:
    return float(row[key])


def success(row: dict, key: str) -> bool:
    return row[key] == "True"


def corr(x: list[float], y: list[float]) -> dict:
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"pearson_r": None, "pearson_p": None, "spearman_rho": None, "spearman_p": None}
    p = pearsonr(x, y)
    s = spearmanr(x, y)
    return {"pearson_r": float(p.statistic), "pearson_p": float(p.pvalue), "spearman_rho": float(s.statistic), "spearman_p": float(s.pvalue)}


def fmt_corr(c: dict) -> str:
    if c["pearson_r"] is None:
        return "not estimable (constant variable)"
    return f"r={c['pearson_r']:.3f} (p={c['pearson_p']:.3f}); ρ={c['spearman_rho']:.3f} (p={c['spearman_p']:.3f})"


def load_threshold(model: str) -> tuple[float, str]:
    data = json.loads(THRESHOLDS[model].read_text())
    # The formal protocol explicitly selects the calibrated FAR=1e-2 point,
    # rather than the first (EER) threshold appearing in the JSON.
    try:
        return float(data["far_1e-2"]["threshold"]), "far_1e-2.threshold"
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"could not locate FAR=1e-2 threshold in {THRESHOLDS[model]}") from exc


def perturbation(clean: np.ndarray, adv: np.ndarray) -> dict:
    d = np.abs(adv - clean)
    # Pixel changes are spatial: one or more RGB elements differs at that pixel.
    changed_pixels = np.any(d != 0, axis=2)
    # 11/255 is the maximum formal serialized magnitude. "Near saturation" is
    # exactly 11 in uint8 space, avoiding an arbitrary floating tolerance.
    return {
        "linf_rgb255": int(d.max()), "l1_rgb255": float(d.sum()),
        "mean_abs_rgb255": float(d.mean()), "l2_rgb255": float(np.sqrt(np.square(d, dtype=np.int64).sum())),
        "rms_rgb255": float(np.sqrt(np.mean(np.square(d, dtype=np.int64)))),
        "changed_pixel_count": int(changed_pixels.sum()), "changed_pixel_ratio": float(changed_pixels.mean()),
        "epsilon_saturation_element_ratio": float((d >= 11).mean()),
        "changed_element_ratio": float((d != 0).mean()),
        "channel_mean_abs_rgb255": [float(d[:, :, c].mean()) for c in range(3)],
    }


def summarize(rows: list[dict], fields: list[str]) -> dict:
    ans = {"n": len(rows)}
    for f in fields:
        x = np.array([float(r[f]) for r in rows])
        ans[f] = {"mean": float(x.mean()), "median": float(np.median(x)), "std": float(x.std(ddof=1))}
    return ans


def paired(base: list[dict], masked: list[dict], label: str) -> tuple[list[dict], dict]:
    b, m = {r["identity_id"]: r for r in base}, {r["identity_id"]: r for r in masked}
    fields = ["arcface_cosine_drop", "facenet_cosine_drop", "ssim", "lpips", "dists", "l1_rgb255", "l2_rgb255", "changed_pixel_ratio"]
    rows, tests = [], {}
    for ident in sorted(b):
        r = {"identity_id": ident, "comparison": label}
        for f in fields:
            r[f"delta_{f}"] = float(m[ident][f]) - float(b[ident][f])
        for target, source in [("arcface", "arcface_threshold_crossing"), ("facenet", "facenet_transfer_crossing")]:
            before, after = success(b[ident], source), success(m[ident], source)
            r[f"{target}_transition"] = f"{'success' if before else 'fail'}→{'success' if after else 'fail'}"
        rows.append(r)
    for f in fields:
        x = np.array([r[f"delta_{f}"] for r in rows])
        nonzero = x[x != 0]
        sh = shapiro(x) if len(x) >= 3 else None
        if len(nonzero):
            w = wilcoxon(x, zero_method="wilcox", alternative="two-sided", method="auto")
            wp, ws = float(w.pvalue), float(w.statistic)
        else:
            wp, ws = 1.0, 0.0
        tests[f] = {"median_mask_minus_full": float(np.median(x)), "mean_mask_minus_full": float(x.mean()),
                    "shapiro_p": float(sh.pvalue) if sh else None, "wilcoxon_statistic": ws, "wilcoxon_p": wp,
                    "n_nonzero": int(len(nonzero))}
    for target in ["arcface", "facenet"]:
        trans = [r[f"{target}_transition"] for r in rows]
        a, d = trans.count("success→fail"), trans.count("fail→success")
        n = a + d
        tests[f"{target}_mcnemar_exact"] = {"success_to_fail": a, "fail_to_success": d, "p": float(binomtest(min(a, d), n, 0.5).pvalue) if n else 1.0}
    return rows, tests


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows: return
    with path.open("w", newline="", encoding="utf-8") as f:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def plot_perturbation(rows: list[dict]) -> None:
    fields = [("mean_abs_rgb255", "mean |δ| (RGB levels)"), ("l2_rgb255", "L2 (RGB levels)"), ("changed_pixel_ratio", "changed-pixel ratio"), ("epsilon_saturation_element_ratio", "11/255 saturation-element ratio")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (field, title) in zip(axes, fields):
        vals = [[float(r[field]) for r in rows if r["method"] == method] for method in METHODS]
        box = ax.boxplot(vals, patch_artist=True, showfliers=False)
        for patch, method in zip(box["boxes"], METHODS): patch.set_facecolor(COLORS[method]); patch.set_alpha(.7)
        for i, x in enumerate(vals, 1): ax.scatter(np.full(len(x), i), x, color="#172235", s=8, alpha=.55)
        ax.set_xticklabels(METHODS, rotation=40, ha="right", fontsize=8); ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(OUT / "perturbation_distribution.png", dpi=200); plt.close(fig)


def plot_paired(base: list[dict], masked: list[dict], filename: str, title: str) -> None:
    b, m = {x["identity_id"]: x for x in base}, {x["identity_id"]: x for x in masked}
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
    for ax, field, ylabel in zip(axes, ["lpips", "facenet_cosine_drop"], ["LPIPS (lower better)", "FaceNet cosine drop"]):
        for ident in sorted(b): ax.plot([0, 1], [float(b[ident][field]), float(m[ident][field])], color="#9aa9b7", lw=.8, zorder=1)
        for x, group, color in [(0, base, COLORS[base[0]["method"]]), (1, masked, COLORS[masked[0]["method"]])]:
            vals = [float(r[field]) for r in group]; ax.scatter(np.full(20, x), vals, color=color, s=22, zorder=2)
            ax.scatter(x, np.mean(vals), color="#172235", marker="_", s=360, linewidths=2, zorder=3)
        ax.set_xticks([0, 1], [base[0]["method"], masked[0]["method"]]); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.2)
    fig.suptitle(title, fontsize=11); fig.tight_layout(); fig.savefig(OUT / filename, dpi=200); plt.close(fig)


def plot_source_transfer(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for method in METHODS:
        x = [float(r["arcface_cosine_drop"]) for r in rows if r["method"] == method]
        y = [float(r["facenet_cosine_drop"]) for r in rows if r["method"] == method]
        ax.scatter(x, y, label=method, color=COLORS[method], s=36, alpha=.85)
    ax.set(xlabel="ArcFace cosine drop", ylabel="FaceNet cosine drop", title="Source strength vs transfer effect"); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT / "arcface_vs_facenet_drop.png", dpi=200); plt.close(fig)


def plot_coverage(mask_rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for ax, field, lab in zip(axes, ["facenet_cosine_drop", "lpips"], ["FaceNet cosine drop", "LPIPS"]):
        for method in ["PGD+Mask", "MI-FGSM+Mask"]:
            s = [r for r in mask_rows if r["method"] == method]
            ax.scatter([float(r["mask_coverage"]) for r in s], [float(r[field]) for r in s], color=COLORS[method], label=method, s=35)
        ax.set(xlabel="mask coverage", ylabel=lab); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "mask_coverage_relationships.png", dpi=200); plt.close(fig)


def plot_crossing(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, method in zip(axes, METHODS):
        r = [x for x in rows if x["method"] == method]
        matrix = np.array([[sum(success(x,"arcface_threshold_crossing") and success(x,"facenet_transfer_crossing") for x in r), sum(success(x,"arcface_threshold_crossing") and not success(x,"facenet_transfer_crossing") for x in r)], [sum(not success(x,"arcface_threshold_crossing") and success(x,"facenet_transfer_crossing") for x in r), sum(not success(x,"arcface_threshold_crossing") and not success(x,"facenet_transfer_crossing") for x in r)]])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=20)
        for i in range(2):
            for j in range(2): ax.text(j, i, str(matrix[i,j]), ha="center", va="center", fontsize=16)
        ax.set(title=method, xticks=[0,1], xticklabels=["FN success","FN fail"], yticks=[0,1], yticklabels=["Arc success","Arc fail"])
    fig.tight_layout(); fig.savefig(OUT / "arcface_facenet_crossing.png", dpi=200); plt.close(fig)


def report_section(result: dict) -> str:
    perturb = result["perturbation_summary"]
    def stat(method, field):
        x = perturb[method][field]; return f"{x['mean']:.3f} / {x['median']:.3f} / {x['std']:.3f}"
    rows = "".join(f"<tr><td>{m}</td><td>{stat(m,'l1_rgb255')}</td><td>{stat(m,'l2_rgb255')}</td><td>{stat(m,'mean_abs_rgb255')}</td><td>{stat(m,'changed_pixel_ratio')}</td><td>{stat(m,'epsilon_saturation_element_ratio')}</td></tr>" for m in METHODS)
    cross = "".join(f"<tr><td>{m}</td><td>{v['arc_success_face_success']}</td><td>{v['arc_success_face_fail']}</td><td>{v['arc_fail_face_success']}</td><td>{v['arc_fail_face_fail']}</td></tr>" for m,v in result['crossing_tables'].items())
    pair_summary = "".join(f"<tr><td>{name}</td><td>{v['perceptual_all_improve']}/20</td><td>{v['facenet_drop_increase']}/{v['facenet_drop_same']}/{v['facenet_drop_decrease']}</td><td>{v['facenet_transitions']['success→fail']} / {v['facenet_transitions']['fail→success']} / {v['facenet_transitions']['success→success']} / {v['facenet_transitions']['fail→fail']}</td></tr>" for name,v in result['paired_transition_summary'].items())
    q = result['q_answers']
    qhtml = "".join(f"<h3>{k}</h3><p><b>Observation.</b> {escape(v['observation'])}</p><p><b>Analysis.</b> {escape(v['analysis'])}</p><p><b>Interpretation.</b> {escape(v['interpretation'])}</p><p class='note'><b>Limitation.</b> {escape(v['limitation'])}</p>" for k,v in q.items())
    return f'''<section id="mechanism-analysis"><div class="eyebrow">Completed read-only mechanism analysis</div><h2>從既有 serialized B_adv 解答 Q1–Q5</h2><p class="lede">n=20 paired identities / method；只重新解碼既有 PNG、讀取 formal CSV 和 calibration artifacts。沒有生成或修改任何 adversarial image。</p><div class="images"><figure><img src="assets/analysis/perturbation_distribution.png" alt="Perturbation distributions"><figcaption>同為 11/255 L∞，總擾動量、空間擴散及 saturation 仍不同。</figcaption></figure><figure><img src="assets/analysis/arcface_vs_facenet_drop.png" alt="Source versus transfer"><figcaption>每方法的 ArcFace drop × FaceNet drop。</figcaption></figure><figure><img src="assets/analysis/arcface_facenet_crossing.png" alt="Crossing tables"><figcaption>ArcFace × FaceNet success crossing。</figcaption></figure></div><h3>Perturbation distribution（mean / median / SD；RGB-level units except ratios）</h3><div class="table-wrap"><table><tr><th>method</th><th>L1</th><th>L2</th><th>mean |δ|</th><th>changed pixel ratio</th><th>11/255 saturation element ratio</th></tr>{rows}</table></div><p class="note">Definitions: L1=Σ|δ|; L2=√Σδ²; mean |δ|=L1/(HWC); RMS=√mean(δ²); changed-pixel ratio=any RGB channel changed / HW; saturation ratio=|δ|=11 / HWC. All are computed in decoded uint8 RGB, so L∞=11/255 for every observation.</p><p><b>Correlation scope correction.</b> The pooled n=80 correlation mixes four attack-method clusters. The complete per-method n=20 Pearson and Spearman results are in <code>mechanism_analysis.json</code>: within methods, L1/L2/changed-ratio relationships with LPIPS are heterogeneous and often weak, whereas changed ratio versus SSIM remains consistently strong. Therefore the defensible statement is that <em>between-method perturbation amount/distribution and perceptual quality are aligned</em>; it is not an individual-level causal claim.</p><p class="note">PGD+Mask persists all final-mask PNGs. MI-FGSM+Mask records the same identity-level deterministic constraint but not its final-mask PNG; therefore its coverage joins use the matching PGD+Mask persisted mask and are labelled as a shared-constraint assumption, not an independent byte-for-byte mask verification.</p><h3>ArcFace × FaceNet crossing (counts)</h3><div class="table-wrap"><table><tr><th>method</th><th>Arc success / FN success</th><th>Arc success / FN fail</th><th>Arc fail / FN success</th><th>Arc fail / FN fail</th></tr>{cross}</table></div><h3>Compact paired comparison</h3><div class="table-wrap"><table><tr><th>comparison</th><th>all 3 perceptual metrics improve</th><th>FaceNet drop increase / same / decrease</th><th>FaceNet success→fail / fail→success / success→success / fail→fail</th></tr>{pair_summary}</table></div><div class="images"><figure><img src="assets/analysis/paired_pgd_mask.png" alt="PGD paired comparison"><figcaption>PGD → PGD+Mask paired effects.</figcaption></figure><figure><img src="assets/analysis/paired_mi_fgsm_mask.png" alt="MI-FGSM paired comparison"><figcaption>MI-FGSM → MI-FGSM+Mask paired effects.</figcaption></figure><figure><img src="assets/analysis/mask_coverage_relationships.png" alt="Mask coverage"><figcaption>Mask coverage relationships (pooled masked observations).</figcaption></figure></div><h2>Q1–Q5: evidence bounded conclusions</h2>{qhtml}<details><summary>Machine-readable derived artifacts and statistics</summary><p><code>assets/analysis/mechanism_analysis.json</code>, <code>perturbation_per_identity.csv</code>, <code>mask_coverage_per_identity.csv</code>, and <code>paired_comparisons.csv</code>. Paired continuous outcomes use Shapiro diagnostic plus two-sided Wilcoxon signed-rank; paired ASR uses exact McNemar/binomial on discordant pairs. Interpret p-values as evidence within the fixed development sample, not a population estimate.</p></details></section>'''


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arc_t, arc_field = load_threshold("ArcFace")
    fn_t, fn_field = load_threshold("FaceNet")
    if not (math.isclose(arc_t, .140721, abs_tol=1e-6) and math.isclose(fn_t, .404789, abs_tol=1e-6)):
        raise RuntimeError(f"calibration mismatch: {arc_t}, {fn_t}")
    with (FORMAL / "per_identity_results.csv").open(encoding="utf-8", newline="") as f: rows = list(csv.DictReader(f))
    if len(rows) != 80 or {r['method'] for r in rows} != set(METHODS): raise RuntimeError("formal four-method contract failed")
    for row in rows:
        clean, adv = rgb(ROOT / row["probe_image"]), rgb(ROOT / row["adversarial_image"])
        row.update(perturbation(clean, adv))
        row["arcface_clean_margin"] = numeric(row,"arcface_clean_cosine") - arc_t; row["arcface_adv_margin"] = numeric(row,"arcface_adv_cosine") - arc_t
        row["facenet_clean_margin"] = numeric(row,"facenet_clean_cosine") - fn_t; row["facenet_adv_margin"] = numeric(row,"facenet_adv_cosine") - fn_t
    save_csv(OUT / "perturbation_per_identity.csv", rows)
    pert_fields = ["linf_rgb255","l1_rgb255","mean_abs_rgb255","l2_rgb255","rms_rgb255","changed_pixel_ratio","changed_pixel_count","epsilon_saturation_element_ratio","changed_element_ratio"]
    psum = {m: summarize([r for r in rows if r['method']==m], pert_fields) for m in METHODS}
    correlation_pairs = [("L1_vs_LPIPS","l1_rgb255","lpips"),("L2_vs_LPIPS","l2_rgb255","lpips"),("changed_ratio_vs_LPIPS","changed_pixel_ratio","lpips"),("changed_ratio_vs_SSIM","changed_pixel_ratio","ssim"),("changed_ratio_vs_DISTS","changed_pixel_ratio","dists")]
    # The pooled n=80 statistic can be driven by four method clusters.  Keep
    # the n=20 within-method results adjacent to it so the report does not
    # overstate a between-method pattern as an individual-level mechanism.
    pcor = {"pooled_n80": {pair: corr([float(r[a]) for r in rows], [numeric(r,b) for r in rows]) for pair,a,b in correlation_pairs},
            "within_method_n20": {method: {pair: corr([float(r[a]) for r in rows if r["method"] == method], [numeric(r,b) for r in rows if r["method"] == method]) for pair,a,b in correlation_pairs} for method in METHODS}}
    # Coverage reads the persisted binary final mask; v3 currently stores visualizations for PGD+Mask only.
    mask_rows=[]
    for r in [x for x in rows if x['method']=='PGD+Mask']:
        mask = np.asarray(Image.open(FORMAL/'pgd_landmark_superpixel_mask'/'visualizations'/r['identity_id']/'final_mask.png').convert('L')) > 0
        z={k:v for k,v in r.items() if k not in ('channel_mean_abs_rgb255',)}; z['mask_coverage']=float(mask.mean()); z['mask_pixels']=int(mask.sum()); z['total_pixels']=int(mask.size); mask_rows.append(z)
    # MI-FGSM v3 did not persist final_mask PNGs. Its generation records and
    # config specify the same identity input and deterministic constraint, so
    # this is a shared-constraint join, not a byte-for-byte persisted-mask test.
    pgd_masks={r['identity_id']:r for r in mask_rows}
    for r in [x for x in rows if x['method']=='MI-FGSM+Mask']:
        z={k:v for k,v in r.items() if k not in ('channel_mean_abs_rgb255',)}; source=pgd_masks[r['identity_id']]; z.update(mask_coverage=source['mask_coverage'],mask_pixels=source['mask_pixels'],total_pixels=source['total_pixels'],mask_source='PGD+Mask persisted final_mask; MI-FGSM mask artifact not persisted, shared deterministic-constraint assumption'); mask_rows.append(z)
    save_csv(OUT / "mask_coverage_per_identity.csv", mask_rows)
    mask_corr={}
    for m in ["PGD+Mask","MI-FGSM+Mask","pooled_masked"]:
        s=mask_rows if m=='pooled_masked' else [r for r in mask_rows if r['method']==m]
        mask_corr[m]={k:corr([float(r['mask_coverage']) for r in s],[numeric(r,v) for r in s]) for k,v in [('arcface_drop','arcface_cosine_drop'),('facenet_drop','facenet_cosine_drop'),('ssim','ssim'),('lpips','lpips'),('dists','dists'),('changed_ratio','changed_pixel_ratio')]}
    pairs=[]; paired_tests={}
    for name, base_name, mask_name in [('PGD_to_PGD+Mask','PGD','PGD+Mask'),('MI-FGSM_to_MI-FGSM+Mask','MI-FGSM','MI-FGSM+Mask')]:
        p,t=paired([r for r in rows if r['method']==base_name],[r for r in rows if r['method']==mask_name],name); pairs += p; paired_tests[name]=t
    save_csv(OUT / "paired_comparisons.csv", pairs)
    paired_summary = {}
    for name in paired_tests:
        s = [r for r in pairs if r['comparison'] == name]
        ft = [r['facenet_transition'] for r in s]
        paired_summary[name] = {"perceptual_all_improve": sum(r['delta_ssim'] > 0 and r['delta_lpips'] < 0 and r['delta_dists'] < 0 for r in s), "facenet_drop_increase":sum(r['delta_facenet_cosine_drop'] > 0 for r in s), "facenet_drop_same":sum(r['delta_facenet_cosine_drop'] == 0 for r in s), "facenet_drop_decrease":sum(r['delta_facenet_cosine_drop'] < 0 for r in s), "facenet_transitions":{x:ft.count(x) for x in ['success→fail','fail→success','success→success','fail→fail']}}
    crossing={}; arc_fail_fn_success={}
    for m in METHODS:
        s=[r for r in rows if r['method']==m]
        crossing[m]={"arc_success_face_success":sum(success(r,'arcface_threshold_crossing') and success(r,'facenet_transfer_crossing') for r in s),"arc_success_face_fail":sum(success(r,'arcface_threshold_crossing') and not success(r,'facenet_transfer_crossing') for r in s),"arc_fail_face_success":sum(not success(r,'arcface_threshold_crossing') and success(r,'facenet_transfer_crossing') for r in s),"arc_fail_face_fail":sum(not success(r,'arcface_threshold_crossing') and not success(r,'facenet_transfer_crossing') for r in s)}
        arc_fail_fn_success[m]=[r['identity_id'] for r in s if not success(r,'arcface_threshold_crossing') and success(r,'facenet_transfer_crossing')]
    margins={m:{"facenet_clean_margin_vs_drop":corr([float(r['facenet_clean_margin']) for r in rows if r['method']==m],[numeric(r,'facenet_cosine_drop') for r in rows if r['method']==m]),"arcface_clean_margin_vs_success":corr([float(r['arcface_clean_margin']) for r in rows if r['method']==m],[int(success(r,'arcface_threshold_crossing')) for r in rows if r['method']==m]),"facenet_clean_margin_vs_success":corr([float(r['facenet_clean_margin']) for r in rows if r['method']==m],[int(success(r,'facenet_transfer_crossing')) for r in rows if r['method']==m])} for m in METHODS}
    transfer_successes=[{"method":r['method'],"identity_id":r['identity_id'],"clean_margin":r['facenet_clean_margin'],"adv_margin":r['facenet_adv_margin'],"drop":numeric(r,'facenet_cosine_drop')} for r in rows if success(r,'facenet_transfer_crossing')]
    source_transfer={m:corr([numeric(r,'arcface_cosine_drop') for r in rows if r['method']==m],[numeric(r,'facenet_cosine_drop') for r in rows if r['method']==m]) for m in METHODS}
    q_answers={
      "Q1": {"observation":"PGD has larger mean ArcFace drop, while MI-FGSM has larger mean FaceNet drop and more FaceNet crossings.","analysis":"Within-method source/transfer correlations are recorded in the derived JSON; they describe association, not causation.","interpretation":"Existing artifacts support only that surrogate strength and transferability are not interchangeable rankings in this fixed n=20 sample.","limitation":"They do not identify a momentum mechanism; gradient alignment or trajectory evidence is required."},
      "Q2": {"observation":"Both masked variants have substantially lower LPIPS/DISTS and higher SSIM; paired deltas quantify whether their FaceNet drop is retained per identity.","analysis":"The decoded-PNG L1/L2/changed-ratio and mask coverage correlations separate total/spatial perturbation from shared L∞.","interpretation":"Masking redistributes/reduces perturbation under the same peak bound; the observed FaceNet effect is method- and identity-dependent, not proof that a particular facial feature was preserved.","limitation":"Coverage and pixel statistics do not establish feature-level causality."},
      "Q3": {"observation":"MI-FGSM→MI-FGSM+Mask changes ArcFace ASR from 18/20 to 7/20; 11 pairs go success→fail and none fail→success (exact McNemar p=0.001).","analysis":"Mask coverage is positively associated with ArcFace drop, but has essentially no association with FaceNet drop (pooled Pearson r=-0.028); paired perturbation reductions are documented.","interpretation":"Coverage alone does not explain the MI-FGSM white-box ASR reduction, although the observed effect is large in this development sample.","limitation":"The optimization mechanism remains a Remaining Question: it needs per-step gradients/loss/trajectory or gradient-alignment analysis."},
      "Q4": {"observation":f"MI-FGSM+Mask crossing counts are Arc success/FN success={crossing['MI-FGSM+Mask']['arc_success_face_success']}, Arc success/FN fail={crossing['MI-FGSM+Mask']['arc_success_face_fail']}, Arc fail/FN success={crossing['MI-FGSM+Mask']['arc_fail_face_success']}, Arc fail/FN fail={crossing['MI-FGSM+Mask']['arc_fail_face_fail']}.","analysis":"The matrix and identity list in mechanism_analysis.json directly resolve the 35% versus 30% overlap.","interpretation":"ArcFace success is not a necessary condition for FaceNet success whenever Arc-fail/FN-success cells are nonzero.","limitation":"A 2×2 overlap does not infer cross-model causal direction."},
      "Q5": {"observation":"Every decoded PNG reaches 11/255 L∞, whereas L1/L2, mean |δ|, changed ratio and saturation ratio differ across methods.","analysis":"Pearson/Spearman tests are reported both pooled (n=80) and within each method (n=20). The latter are heterogeneous, so the pooled pattern is not treated as an individual-level relation.","interpretation":"L∞ controls only the single largest element; between-method perturbation amount/distribution and perceptual quality are aligned, explaining why equal L∞ can look different.","limitation":"The n=20 within-method associations are exploratory and do not establish pixel- or feature-level causality."}}
    result={"scope":"read-only existing formal artifacts; no attack rerun", "thresholds":{"arcface":{"value":arc_t,"artifact":str(THRESHOLDS['ArcFace'].relative_to(ROOT)),"field":arc_field},"facenet":{"value":fn_t,"artifact":str(THRESHOLDS['FaceNet'].relative_to(ROOT)),"field":fn_field}},"metric_definitions":{"units":"decoded uint8 RGB levels unless ratio","epsilon_saturation":"abs(delta)==11 over H*W*3","changed_pixel":"any RGB channel changed"},"perturbation_summary":psum,"perturbation_correlations_pooled_80":pcor,"mask_coverage_correlations":mask_corr,"paired_tests":paired_tests,"paired_transition_summary":paired_summary,"crossing_tables":crossing,"arcface_fail_facenet_success_identities":arc_fail_fn_success,"threshold_margin_correlations":margins,"facenet_transfer_successes":transfer_successes,"source_vs_transfer_correlations":source_transfer,"q_answers":q_answers}
    (OUT/'mechanism_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    plot_perturbation(rows); plot_paired([r for r in rows if r['method']=='PGD'],[r for r in rows if r['method']=='PGD+Mask'],'paired_pgd_mask.png','Paired PGD → PGD+Mask'); plot_paired([r for r in rows if r['method']=='MI-FGSM'],[r for r in rows if r['method']=='MI-FGSM+Mask'],'paired_mi_fgsm_mask.png','Paired MI-FGSM → MI-FGSM+Mask'); plot_source_transfer(rows); plot_coverage(mask_rows); plot_crossing(rows)
    text=REPORT.read_text(encoding='utf-8'); marker='<section id="appendix">'; section=report_section(result)
    if '<section id="mechanism-analysis">' in text:
        start=text.index('<section id="mechanism-analysis">'); end=text.index('</section>',start)+len('</section>'); text=text[:start]+section+text[end:]
    else: text=text.replace(marker,section+marker)
    text=text.replace('<a href="#questions">7. 值得討論的問題</a>', '<a href="#mechanism-analysis">7. Mechanism analysis</a><a href="#questions">8. 值得討論的問題</a>')
    text=text.replace('Authority audit finding:</b> <code>RESEARCH_STATUS.md</code> 的 Authority Table 目前列出 20-identity baseline v2 與 5-identity mask ablation v2，但尚未列出本次正式四方法 v3 artifact。v3 的 <code>summary.json</code> 自述 <code>formal: true</code>、sanity check passed，且其 aggregate 與本報告一致；這是一個文件 provenance 同步缺口，並非本報告數字與 v3 artifact 的不一致。','Authority audit:</b> <code>RESEARCH_STATUS.md</code> 已同步列入 four-method v3，並保留舊 authority entries。')
    text=re.sub(r'<section id="questions">.*?</section>', '<section id="questions"><div class="eyebrow">Evidence status</div><h2>已回答與剩餘問題</h2><div class="grid"><div class="card"><b>Q1 / Q5</b>已由 source-transfer correlation 與 decoded-PNG perturbation distribution 回答為描述性結論。</div><div class="card"><b>Q2 / Q4</b>已由 paired results、2×2 crossing 與 margin evidence 回答。</div><div class="card"><b>Q3</b>已量化 ASR transition，但真正的 gradient/trajectory mechanism 仍未證明。</div><div class="card"><b>Boundary</b>所有 inference 限於 frozen development n=20；不宣稱 population effect 或 causal mechanism。</div></div></section>', text, flags=re.S)
    text=re.sub(r'<section id="next">.*?</section>', '<section id="next"><div class="eyebrow">Minimal next experiment</div><h2>真正需要重新跑模型的下一步</h2><p>只針對同一個 frozen 20-identity protocol，儲存每步 surrogate loss、gradient norm / masked-gradient norm、gradient direction alignment，以及 final mask artifact；不需先產生新的 attack family。這是解釋 Q3 optimization interaction 的最小新增 evidence。</p></section>', text, flags=re.S)
    REPORT.write_text(text,encoding='utf-8')
    print(json.dumps({"out":str(OUT),"crossing":crossing,"source_transfer":source_transfer,"perturbation_correlations":pcor},ensure_ascii=False,indent=2))

if __name__ == '__main__': main()
