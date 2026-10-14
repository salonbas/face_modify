"""Benchmark Summary Report：靜態 HTML + matplotlib 圖表。HTML 只讀 summary.json。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None or v == "":
        return "N/A"
    if isinstance(v, str):
        return v
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if x != x:
        return "NaN"
    if abs(x - round(x)) < 1e-12 and abs(x) >= 1:
        return str(int(round(x)))
    return f"{x:.{digits}f}"


def _pct(v: Any) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{100.0 * float(v):.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def _cfg_label(c: dict[str, Any]) -> str:
    attack = c.get("attack", "")
    eps = c.get("eps")
    steps = c.get("steps")
    if attack == "fgsm" or steps in (None, "", 1):
        return f"{attack} eps={_fmt(eps, 3)}"
    return f"{attack} eps={_fmt(eps, 3)} steps={steps}"


def write_charts(out_dir: Path, summary: dict[str, Any]) -> dict[str, str]:
    charts_dir = Path(out_dir) / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    per = summary.get("per_config") or []
    dist = summary.get("distributions") or {}
    paths: dict[str, str] = {}

    def _save(fig, name: str) -> None:
        p = charts_dir / name
        fig.tight_layout()
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths[name] = f"charts/{name}"

    # epsilon vs white-box success / victim drop (per attack, mean over steps)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for attack, marker in (("fgsm", "o"), ("pgd_full", "s")):
        xs, ys = [], []
        by_eps: dict[float, list[float]] = {}
        for c in per:
            if c.get("attack") != attack:
                continue
            rate = (c.get("all") or {}).get("whitebox_success_rate")
            if rate is None:
                continue
            by_eps.setdefault(float(c["eps"]), []).append(float(rate))
        for eps in sorted(by_eps):
            xs.append(eps)
            ys.append(float(np.mean(by_eps[eps])))
        if xs:
            ax.plot(xs, ys, marker=marker, label=attack)
    ax.set_xlabel("epsilon")
    ax.set_ylabel("white-box success rate")
    ax.set_title("Epsilon vs white-box success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "eps_vs_whitebox.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for attack, marker in (("fgsm", "o"), ("pgd_full", "s")):
        by_eps: dict[float, list[float]] = {}
        for c in per:
            if c.get("attack") != attack:
                continue
            drop = (c.get("all") or {}).get("victim_cosine_drop_mean")
            if drop is None:
                continue
            by_eps.setdefault(float(c["eps"]), []).append(float(drop))
        xs, ys = [], []
        for eps in sorted(by_eps):
            xs.append(eps)
            ys.append(float(np.mean(by_eps[eps])))
        if xs:
            ax.plot(xs, ys, marker=marker, label=attack)
    ax.set_xlabel("epsilon")
    ax.set_ylabel("victim cosine drop (mean)")
    ax.set_title("Epsilon vs victim cosine drop")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "eps_vs_victim_drop.png")

    # PGD steps
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    by_steps: dict[int, list[float]] = {}
    for c in per:
        if c.get("attack") != "pgd_full":
            continue
        v = (c.get("all") or {}).get("surrogate_cosine_mean")
        if v is None:
            continue
        by_steps.setdefault(int(c["steps"]), []).append(float(v))
    xs = sorted(by_steps)
    ys = [float(np.mean(by_steps[s])) for s in xs]
    if xs:
        ax.plot(xs, ys, marker="^", color="#2ca02c")
    ax.set_xlabel("PGD steps")
    ax.set_ylabel("surrogate cosine (mean)")
    ax.set_title("PGD steps vs surrogate cosine")
    ax.grid(True, alpha=0.3)
    _save(fig, "steps_vs_surrogate_cosine.png")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    by_steps = {}
    for c in per:
        if c.get("attack") != "pgd_full":
            continue
        v = (c.get("all") or {}).get("victim_cosine_drop_mean")
        if v is None:
            continue
        by_steps.setdefault(int(c["steps"]), []).append(float(v))
    xs = sorted(by_steps)
    ys = [float(np.mean(by_steps[s])) for s in xs]
    if xs:
        ax.plot(xs, ys, marker="^", color="#d62728")
    ax.set_xlabel("PGD steps")
    ax.set_ylabel("victim cosine drop (mean)")
    ax.set_title("PGD steps vs victim cosine drop")
    ax.grid(True, alpha=0.3)
    _save(fig, "steps_vs_victim_drop.png")

    def _hist(values, title, name, xlabel):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        arr = np.asarray(values, dtype=np.float64) if values else np.asarray([])
        if arr.size:
            ax.hist(arr, bins=min(30, max(8, int(np.sqrt(arr.size)))), color="#4c6a92", edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        _save(fig, name)

    _hist(dist.get("surrogate_cosine"), "Surrogate cosine distribution", "dist_surrogate_cosine.png", "cosine")
    _hist(dist.get("victim_cosine"), "Victim cosine distribution", "dist_victim_cosine.png", "cosine")
    _hist(dist.get("victim_cosine_drop"), "Victim cosine-drop distribution", "dist_victim_drop.png", "1 - cosine")
    _hist(dist.get("victim_euclidean"), "Victim Euclidean-distance distribution", "dist_victim_euclidean.png", "euclidean")

    return paths


def write_benchmark_report_html(out_dir: Path, summary: Optional[dict[str, Any]] = None) -> Path:
    """只讀 summary.json 呈現；不做指標重算。"""
    out_dir = Path(out_dir)
    summary_path = out_dir / "summary.json"
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as f:
            s = json.load(f)
    elif summary is not None:
        s = summary
    else:
        raise FileNotFoundError(f"找不到 summary.json：{summary_path}")

    ex = s.get("executive") or {}
    per = s.get("per_config") or []
    charts = s.get("charts") or {}
    examples = s.get("examples") or {}
    limitations = s.get("limitations") or []
    failures = s.get("failure_counts") or {}
    rq = s.get("research_question") or ""

    strongest = ex.get("strongest_transfer_configuration") or {}
    weakest = ex.get("weakest_transfer_configuration") or {}

    def img(name: str, alt: str) -> str:
        src = charts.get(name, f"charts/{name}")
        return f'<figure><img src="{src}" alt="{alt}"/><figcaption>{alt}</figcaption></figure>'

    cfg_rows = []
    for c in per:
        a = c.get("all") or {}
        w = c.get("whitebox_success_only") or {}
        cfg_rows.append(
            "<tr>"
            f"<td>{_cfg_label(c)}</td>"
            f"<td>{a.get('n_attempted', '')}</td>"
            f"<td>{a.get('n_whitebox_success', '')}</td>"
            f"<td>{_pct(a.get('whitebox_success_rate'))}</td>"
            f"<td>{_fmt(a.get('victim_cosine_drop_mean'))}</td>"
            f"<td>{_fmt(w.get('victim_cosine_drop_mean'))}</td>"
            f"<td>{_fmt(a.get('victim_euclidean_mean'))}</td>"
            f"<td>{_fmt(a.get('linf_mean'))}</td>"
            "</tr>"
        )

    fail_items = "".join(f"<li><code>{k}</code>: {v}</li>" for k, v in sorted(failures.items()))
    lim_items = "".join(f"<li>{x}</li>" for x in limitations)

    def example_block(key: str, title: str) -> str:
        info = examples.get(key) or {}
        if not info:
            return f"<p class='muted'>{title}: not available.</p>"
        folder = info.get("dir", f"examples/{key}")
        meta = info.get("meta") or {}
        return f"""
        <div class="example">
          <h3>{title}</h3>
          <p class="muted">
            attack={meta.get("attack", "")}
            eps={_fmt(meta.get("eps"), 3)}
            steps={meta.get("steps", "")}
            surrogate cosine={_fmt(meta.get("surrogate_cosine"))}
            victim cosine={_fmt(meta.get("victim_cosine"))}
          </p>
          <div class="imgs">
            <figure><img src="{folder}/original.png" alt="Original"/><figcaption>Original</figcaption></figure>
            <figure><img src="{folder}/adversarial.png" alt="Adversarial"/><figcaption>Adversarial</figcaption></figure>
            <figure><img src="{folder}/perturbation.png" alt="Perturbation"/><figcaption>Perturbation</figcaption></figure>
          </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Large-scale Transfer Benchmark v0</title>
<style>
  :root {{ --bg:#f6f5f2; --card:#fff; --text:#1b1b1b; --muted:#5d5d5d; --line:#d9d6cf; --accent:#2f5d50; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:"Segoe UI","Noto Sans TC","PingFang TC",sans-serif; background:var(--bg); color:var(--text); line-height:1.55; }}
  main {{ max-width:1080px; margin:0 auto; padding:32px 20px 72px; }}
  h1 {{ color:var(--accent); margin:0 0 8px; }}
  h2 {{ margin:36px 0 12px; padding-bottom:6px; border-bottom:1px solid var(--line); }}
  .lead,.muted {{ color:var(--muted); }}
  .cards {{ display:grid; grid-template-columns:repeat(2,1fr); gap:10px; }}
  @media (max-width:800px) {{ .cards {{ grid-template-columns:1fr; }} }}
  .card {{ background:var(--card); border:1px solid var(--line); padding:12px 14px; }}
  .card dt {{ color:var(--muted); font-size:0.88rem; }}
  .card dd {{ margin:4px 0 0; font-weight:700; font-size:1.05rem; }}
  table {{ width:100%; border-collapse:collapse; background:var(--card); border:1px solid var(--line); font-size:0.92rem; }}
  th,td {{ padding:8px 10px; border-bottom:1px solid var(--line); text-align:left; }}
  figure {{ margin:0; background:var(--card); border:1px solid var(--line); padding:8px; }}
  figure img {{ width:100%; height:auto; display:block; }}
  figcaption {{ text-align:center; color:var(--muted); font-size:0.88rem; margin-top:6px; }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
  .imgs {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }}
  @media (max-width:800px) {{ .grid2,.imgs {{ grid-template-columns:1fr; }} }}
  .note {{ background:var(--card); border:1px solid var(--line); padding:12px 14px; }}
  footer {{ margin-top:40px; color:var(--muted); font-size:0.85rem; }}
</style>
</head>
<body>
<main>
  <h1>Large-scale Transfer Benchmark v0</h1>
  <p class="lead">{rq}</p>

  <h2>1. Executive Summary</h2>
  <div class="cards">
    <div class="card"><dt>Dataset size</dt><dd>{ex.get("dataset_size", "N/A")}</dd></div>
    <div class="card"><dt>Valid experiments</dt><dd>{ex.get("n_valid", "N/A")}</dd></div>
    <div class="card"><dt>Attack configurations</dt><dd>{ex.get("n_attack_configurations", "N/A")}</dd></div>
    <div class="card"><dt>Surrogate</dt><dd>{ex.get("surrogate", "")}</dd></div>
    <div class="card"><dt>Victim</dt><dd>{ex.get("victim", "")}</dd></div>
    <div class="card"><dt>Overall white-box success rate</dt><dd>{_pct(ex.get("overall_whitebox_success_rate"))}</dd></div>
    <div class="card"><dt>Victim Transfer Effect (mean cosine drop)</dt><dd>{_fmt(ex.get("victim_mean_cosine_drop"))} ({ex.get("victim_transfer_effect_qualitative", "")})</dd></div>
    <div class="card"><dt>Strongest transfer-effect configuration</dt><dd>{_cfg_label(strongest) if strongest else "N/A"}<br/><span class="muted">drop={_fmt(strongest.get("victim_cosine_drop_mean"))}</span></dd></div>
    <div class="card"><dt>Weakest transfer-effect configuration</dt><dd>{_cfg_label(weakest) if weakest else "N/A"}<br/><span class="muted">drop={_fmt(weakest.get("victim_cosine_drop_mean"))}</span></dd></div>
  </div>
  <p class="note">Victim threshold is uncalibrated. Binary Transfer Success Rate is not reported.
  Displayed value: {ex.get("transfer_success_rate_display", "N/A — victim threshold not calibrated")}</p>
  <p class="muted">Attempted rows: {ex.get("n_completed_rows", "N/A")} · Failed: {ex.get("n_failed", "N/A")} · Invalid L∞: {ex.get("n_invalid", "N/A")}</p>

  <h2>2. Attack Comparison</h2>
  <table>
    <thead>
      <tr>
        <th>Configuration</th><th>Attempts</th><th>White-box success</th><th>WB rate</th>
        <th>Victim cosine drop (all)</th><th>Victim cosine drop (WB only)</th>
        <th>Victim Euclidean</th><th>L∞ mean</th>
      </tr>
    </thead>
    <tbody>
      {''.join(cfg_rows)}
    </tbody>
  </table>

  <h2>3. Epsilon Analysis</h2>
  <div class="grid2">
    {img("eps_vs_whitebox.png", "epsilon vs white-box success rate")}
    {img("eps_vs_victim_drop.png", "epsilon vs victim cosine drop")}
  </div>

  <h2>4. PGD Steps Analysis</h2>
  <div class="grid2">
    {img("steps_vs_surrogate_cosine.png", "steps vs surrogate cosine")}
    {img("steps_vs_victim_drop.png", "steps vs victim cosine drop")}
  </div>
  <p class="muted">These plots present observed values only. More steps are not assumed to be stronger.</p>

  <h2>5. Distribution</h2>
  <div class="grid2">
    {img("dist_surrogate_cosine.png", "Surrogate cosine")}
    {img("dist_victim_cosine.png", "Victim cosine")}
    {img("dist_victim_drop.png", "Victim cosine drop")}
    {img("dist_victim_euclidean.png", "Victim Euclidean distance")}
  </div>

  <h2>6. Representative Examples</h2>
  {example_block("strongest_victim_effect", "Strong victim effect")}
  {example_block("weakest_victim_effect", "Weak victim effect")}
  {example_block("median_victim_effect", "Median case")}
  {example_block("strongest_surrogate_attack", "Strongest surrogate attack")}
  {example_block("weakest_surrogate_attack", "Weakest surrogate attack")}
  {example_block("selected_failure", "Selected failure")}

  <h2>7. Limitations</h2>
  <ul>{lim_items}</ul>
  <p class="note">This benchmark is an evidence accumulation step. Findings apply only to the current dataset, surrogate/victim pair, attack settings, and evaluation metric. They are not a final black-box conclusion.</p>

  <h2>Run accounting</h2>
  <ul>
    <li>Planned: {ex.get("n_planned", "N/A")}</li>
    <li>Completed rows: {ex.get("n_completed_rows", "N/A")}</li>
    <li>Valid: {ex.get("n_valid", "N/A")}</li>
    <li>Failed: {ex.get("n_failed", "N/A")}</li>
  </ul>
  <p>Failure reasons</p>
  <ul>{fail_items or "<li>None recorded.</li>"}</ul>

  <footer>
    Generated from summary.json — this page does not recompute metrics.<br/>
    Source of truth: <code>summary.json</code> and <code>raw_results.jsonl</code>
  </footer>
</main>
</body>
</html>
"""
    path = out_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path
