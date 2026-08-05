"""
Transfer Evaluation Experiment 輸出契約與正式 HTML 報告。

results/transfer/<run>/
  original.png
  adversarial.png
  perturbation.png
  metrics.json
  metrics.csv
  surrogate_embedding.npy
  victim_embedding.npy
  report.html
  config.json          # 與 metrics.json.config 同步，便於單獨讀取

HTML 只呈現 metrics.json 已算好的數值，不做任何計算。
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from advface.attacks.common import _bgr_to_rgb_float, _save_noise_map
from advface.evaluation.transfer import TransferEvalResult, build_conclusion
from advface.experiments.output import write_config_json


def write_transfer_metrics_json(out_dir: Path, metrics: dict[str, Any]) -> Path:
    path = Path(out_dir) / "metrics.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def write_transfer_metrics_csv(out_dir: Path, metrics: dict[str, Any]) -> Path:
    path = Path(out_dir) / "metrics.csv"
    rows = [
        {
            "role": "surrogate",
            "model": metrics["surrogate"]["model"],
            "cosine_before": metrics["surrogate"]["cosine_before"],
            "cosine_after": metrics["surrogate"]["cosine_after"],
            "delta": metrics["surrogate"]["delta"],
            "euclidean_distance": metrics["surrogate"].get("euclidean_distance", ""),
            "success": int(bool(metrics["surrogate"]["success"])),
            "threshold": "" if metrics["surrogate"].get("threshold") is None else metrics["surrogate"]["threshold"],
            "transfer_observed": "",
        },
        {
            "role": "victim",
            "model": metrics["victim"]["model"],
            "cosine_before": metrics["victim"]["cosine_before"],
            "cosine_after": metrics["victim"]["cosine_after"],
            "delta": metrics["victim"]["delta"],
            "euclidean_distance": metrics["victim"].get("euclidean_distance", ""),
            "success": int(bool(metrics["victim"]["success"])),
            "threshold": "" if metrics["victim"].get("threshold") is None else metrics["victim"]["threshold"],
            "transfer_observed": (
                ""
                if metrics["victim"].get("transfer_observed") is None
                else int(bool(metrics["victim"]["transfer_observed"]))
            ),
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "role",
                "model",
                "cosine_before",
                "cosine_after",
                "delta",
                "euclidean_distance",
                "success",
                "threshold",
                "transfer_observed",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    return path


def save_transfer_images(
    out_dir: Path,
    original_bgr: np.ndarray,
    adversarial_bgr: np.ndarray,
) -> dict[str, str]:
    out_dir = Path(out_dir)
    orig_path = out_dir / "original.png"
    adv_path = out_dir / "adversarial.png"
    pert_path = out_dir / "perturbation.png"
    cv2.imwrite(str(orig_path), original_bgr)
    cv2.imwrite(str(adv_path), adversarial_bgr)
    _save_noise_map(
        _bgr_to_rgb_float(adversarial_bgr) - _bgr_to_rgb_float(original_bgr),
        pert_path,
        original_bgr,
    )
    return {
        "original": orig_path.name,
        "adversarial": adv_path.name,
        "perturbation": pert_path.name,
    }


def save_embedding_pair(path: Path, result: TransferEvalResult) -> Optional[Path]:
    if result.embedding_original is None or result.embedding_adversarial is None:
        return None
    arr = np.stack(
        [result.embedding_original, result.embedding_adversarial],
        axis=0,
    ).astype(np.float32)
    np.save(path, arr)
    return path


def build_transfer_metrics(
    *,
    run_id: str,
    experiment_name: str,
    date_iso: str,
    image_path: str,
    image_stem: str,
    surrogate: TransferEvalResult,
    victim: TransferEvalResult,
    attack: dict[str, Any],
    images: dict[str, str],
    config: dict[str, Any],
) -> dict[str, Any]:
    conclusion = build_conclusion(surrogate, victim)
    return {
        "experiment": {
            "name": experiment_name,
            "run_id": run_id,
            "date": date_iso,
            "image": image_path,
            "image_stem": image_stem,
        },
        "attack": attack,
        "surrogate": surrogate.to_metrics_dict(),
        "victim": victim.to_metrics_dict(),
        "images": images,
        "config": config,
        "conclusion": conclusion,
    }


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "N/A"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if x != x:
        return "NaN"
    return f"{x:.{digits}f}"


def write_report_html(out_dir: Path, metrics: dict[str, Any]) -> Path:
    """
    由 metrics.json 內容產生靜態 report.html。
    HTML／inline CSS 只負責呈現，不做計算。
    """
    out_dir = Path(out_dir)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.is_file():
        with metrics_path.open("r", encoding="utf-8") as f:
            m = json.load(f)
    else:
        m = metrics

    exp = m.get("experiment") or {}
    if isinstance(exp, str):
        exp = {"name": exp, "run_id": m.get("run_id", ""), "date": m.get("date", "")}

    s = m["surrogate"]
    v = m["victim"]
    a = m["attack"]
    imgs = m["images"]
    c = m["conclusion"]
    cfg = m.get("config") or {}

    run_id = exp.get("run_id") or m.get("run_id", "")
    date_str = exp.get("date") or m.get("date", "")
    image_label = exp.get("image") or cfg.get("source_image") or ""
    victim_name = v.get("model") or cfg.get("victim_model") or m.get("victim_model", "")
    experiment_title = exp.get("name") or "Transfer Evaluation Experiment"

    transfer_status = c.get("transfer", "Not Observed")
    whitebox_success = "Successful" if s.get("success") else "Failed"

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{experiment_title} — {run_id}</title>
<style>
  :root {{
    --bg: #f7f7f5;
    --card: #ffffff;
    --text: #1a1a1a;
    --muted: #5c5c5c;
    --line: #d8d8d4;
    --accent: #2f5d50;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: "Segoe UI", "Noto Sans TC", "PingFang TC", sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.55;
  }}
  main {{
    max-width: 920px;
    margin: 0 auto;
    padding: 32px 20px 64px;
  }}
  h1 {{
    font-size: 1.75rem;
    margin: 0 0 8px;
    color: var(--accent);
  }}
  h2 {{
    font-size: 1.2rem;
    margin: 36px 0 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--line);
  }}
  .lead {{ color: var(--muted); margin: 0 0 24px; }}
  .meta {{
    display: grid;
    grid-template-columns: 160px 1fr;
    gap: 6px 12px;
    background: var(--card);
    border: 1px solid var(--line);
    padding: 16px 18px;
  }}
  .meta dt {{ color: var(--muted); }}
  .meta dd {{ margin: 0; font-weight: 600; word-break: break-all; }}
  .pipeline {{
    background: var(--card);
    border: 1px solid var(--line);
    padding: 20px 18px;
    font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
    white-space: pre;
    overflow-x: auto;
    font-size: 0.92rem;
  }}
  .imgs {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }}
  @media (max-width: 700px) {{
    .imgs {{ grid-template-columns: 1fr; }}
  }}
  figure {{
    margin: 0;
    background: var(--card);
    border: 1px solid var(--line);
    padding: 10px;
  }}
  figure img {{
    width: 100%;
    height: auto;
    display: block;
    background: #eee;
  }}
  figcaption {{
    margin-top: 8px;
    text-align: center;
    font-size: 0.9rem;
    color: var(--muted);
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--card);
    border: 1px solid var(--line);
  }}
  th, td {{
    text-align: left;
    padding: 10px 12px;
    border-bottom: 1px solid var(--line);
    vertical-align: top;
  }}
  th {{ width: 28%; color: var(--muted); font-weight: 600; }}
  .hint {{
    display: block;
    margin-top: 4px;
    color: var(--muted);
    font-weight: 400;
    font-size: 0.88rem;
  }}
  .status {{
    margin-top: 12px;
    padding: 12px 14px;
    background: var(--card);
    border: 1px solid var(--line);
    font-weight: 700;
  }}
  .conclusion {{
    background: var(--card);
    border: 1px solid var(--line);
    padding: 16px 18px;
  }}
  .conclusion p {{ margin: 0 0 10px; }}
  .conclusion p:last-child {{ margin-bottom: 0; }}
  footer {{
    margin-top: 40px;
    color: var(--muted);
    font-size: 0.85rem;
  }}
</style>
</head>
<body>
<main>
  <h1>{experiment_title}</h1>
  <p class="lead">White-box PGD → Surrogate / Victim Evaluation → Presentation Report</p>

  <h2>1. Experiment Information</h2>
  <dl class="meta">
    <dt>Image</dt><dd>{image_label}</dd>
    <dt>Attack</dt><dd>{a.get("type", "")}</dd>
    <dt>eps</dt><dd>{_fmt(a.get("eps"), 3)}</dd>
    <dt>steps</dt><dd>{a.get("steps", "")}</dd>
    <dt>Victim Model</dt><dd>{victim_name}</dd>
    <dt>Date</dt><dd>{date_str}</dd>
    <dt>Run ID</dt><dd>{run_id}</dd>
  </dl>

  <h2>2. Pipeline</h2>
  <div class="pipeline">Original
    │
    ▼
PGD Full Attack
    │
    ▼
Adversarial Image
    │
    ├───────────────┐
    ▼               ▼
Surrogate        Victim
(ArcFace)       (FaceNet)
    │               │
    ▼               ▼
Embedding      Embedding
    │               │
    ▼               ▼
Cosine Compare  Cosine Compare</div>

  <h2>3. Image Comparison</h2>
  <div class="imgs">
    <figure>
      <img src="{imgs.get("original", "original.png")}" alt="Original"/>
      <figcaption>Original</figcaption>
    </figure>
    <figure>
      <img src="{imgs.get("adversarial", "adversarial.png")}" alt="Adversarial"/>
      <figcaption>Adversarial</figcaption>
    </figure>
    <figure>
      <img src="{imgs.get("perturbation", "perturbation.png")}" alt="Perturbation"/>
      <figcaption>Perturbation</figcaption>
    </figure>
  </div>

  <h2>4. White-box Result</h2>
  <table>
    <tr>
      <th>Cosine Before</th>
      <td>{_fmt(s.get("cosine_before"))}
        <span class="hint">攻擊前乾淨圖的自我相似度；越接近 1，代表 embedding 越相似。</span>
      </td>
    </tr>
    <tr>
      <th>Cosine After</th>
      <td>{_fmt(s.get("cosine_after"))}
        <span class="hint">原圖與對抗圖 embedding 的 Cosine Similarity；越接近 1，代表越相似。</span>
      </td>
    </tr>
    <tr>
      <th>Delta</th>
      <td>{_fmt(s.get("delta"))}
        <span class="hint">Cosine Before − Cosine After；越大代表攻擊拉開相似度越多。</span>
      </td>
    </tr>
    <tr>
      <th>Euclidean Distance</th>
      <td>{_fmt(s.get("euclidean_distance"))}
        <span class="hint">原圖與對抗圖 embedding 的歐氏距離；越大代表差異越大。</span>
      </td>
    </tr>
    <tr>
      <th>Threshold</th>
      <td>{_fmt(s.get("threshold"))}
        <span class="hint">該模型判定「是否同一人」的門檻。</span>
      </td>
    </tr>
    <tr>
      <th>Attack Success</th>
      <td>{"Yes" if s.get("success") else "No"}
        <span class="hint">若 Cosine After 低於 threshold，視為白盒攻擊成功。</span>
      </td>
    </tr>
  </table>
  <p class="status">White-box Attack：{whitebox_success}<br/><span style="font-weight:500;color:var(--muted)">Model: {s.get("model", "")}</span></p>

  <h2>5. Victim Result</h2>
  <table>
    <tr>
      <th>Cosine Before</th>
      <td>{_fmt(v.get("cosine_before"))}
        <span class="hint">攻擊前乾淨圖的自我相似度；越接近 1，代表 embedding 越相似。</span>
      </td>
    </tr>
    <tr>
      <th>Cosine After</th>
      <td>{_fmt(v.get("cosine_after"))}
        <span class="hint">原圖與對抗圖 embedding 的 Cosine Similarity；越接近 1，代表越相似。</span>
      </td>
    </tr>
    <tr>
      <th>Delta</th>
      <td>{_fmt(v.get("delta"))}
        <span class="hint">Cosine Before − Cosine After；越大代表對 victim 拉開相似度越多。</span>
      </td>
    </tr>
    <tr>
      <th>Euclidean Distance</th>
      <td>{_fmt(v.get("euclidean_distance"))}
        <span class="hint">原圖與對抗圖 embedding 的歐氏距離；越大代表差異越大。</span>
      </td>
    </tr>
    <tr>
      <th>Threshold</th>
      <td>{_fmt(v.get("threshold"))}
        <span class="hint">正式 verification threshold；尚未建立時顯示 N/A，不硬套 ArcFace 門檻。</span>
      </td>
    </tr>
    <tr>
      <th>Transfer Status</th>
      <td>{transfer_status}
        <span class="hint">是否在 victim 上觀測到可轉移的攻擊效果（Observed / Not Observed）。</span>
      </td>
    </tr>
  </table>
  <div class="status">Victim Transfer：{transfer_status}<br/><span style="font-weight:500;color:var(--muted)">Model: {v.get("model", "")}</span></div>

  <h2>6. Summary</h2>
  <div class="conclusion">
    <p><strong>White-box Attack：</strong> {c.get("whitebox", "")}</p>
    <p><strong>Victim Transfer：</strong> {c.get("transfer", "")}</p>
    <p><strong>Conclusion：</strong> {c.get("body", "")}</p>
    <p>{c.get("closing", "")}</p>
  </div>

  <footer>
    Generated from metrics.json — values are display-only; no computation in this page.<br/>
    Source of truth: <code>metrics.json</code>
  </footer>
</main>
<script type="application/json" id="metrics-data">
{json.dumps(m, ensure_ascii=False, indent=2)}
</script>
</body>
</html>
"""
    path = out_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


def save_transfer_run(
    *,
    out_dir: Path,
    original_bgr: np.ndarray,
    adversarial_bgr: np.ndarray,
    surrogate: TransferEvalResult,
    victim: TransferEvalResult,
    config: dict[str, Any],
    attack: dict[str, Any],
    run_id: str,
    experiment: str,
    image_path: str,
    image_stem: str,
    victim_model_name: str | None = None,
) -> dict[str, Path]:
    """寫入完整 transfer experiment output contract。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 相容舊呼叫：若未傳 image，從 config 取
    if not image_path:
        image_path = str(config.get("source_image", ""))
    if not image_stem:
        image_stem = Path(image_path).stem if image_path else ""

    date_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    images = save_transfer_images(out_dir, original_bgr, adversarial_bgr)

    # 確保 config 含 victim 名稱
    if victim_model_name and "victim_model" not in config:
        config = {**config, "victim_model": victim_model_name}

    metrics = build_transfer_metrics(
        run_id=run_id,
        experiment_name=experiment,
        date_iso=date_iso,
        image_path=image_path,
        image_stem=image_stem,
        surrogate=surrogate,
        victim=victim,
        attack=attack,
        images=images,
        config=config,
    )

    write_config_json(out_dir, config)
    metrics_json = write_transfer_metrics_json(out_dir, metrics)
    metrics_csv = write_transfer_metrics_csv(out_dir, metrics)

    sur_emb = save_embedding_pair(out_dir / "surrogate_embedding.npy", surrogate)
    vic_emb = save_embedding_pair(out_dir / "victim_embedding.npy", victim)
    report = write_report_html(out_dir, metrics)

    paths: dict[str, Path] = {
        "config": out_dir / "config.json",
        "metrics_json": metrics_json,
        "metrics_csv": metrics_csv,
        "report": report,
    }
    if sur_emb is not None:
        paths["surrogate_embedding"] = sur_emb
    if vic_emb is not None:
        paths["victim_embedding"] = vic_emb
    return paths
