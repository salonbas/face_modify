# Research Status

區分「平台能做什麼」與「研究已經得到什麼」。本檔是 status authority。

## Infrastructure

已具備：

- White-box FGSM、PGD crop、PGD Full、MI-FGSM（ArcFace / InsightFace `buffalo_l`）
- Embedding 評估介面：InsightFace、FaceNet（VGGFace2）
- Canonical `run_experiment`（攻擊 → surrogate / victims 評估 → `ExperimentResult`）
- Single-image transfer HTML（只讀 metrics）
- Batch benchmark：grid、checkpoint / resume、failure isolation、aggregate HTML
- Dataset contract：`data/benchmark/manifest.csv`

## Validated Research Results

正式白盒 evidence（內容未重跑、未改寫）：

- FGSM：`results/fgsm/sun_v1`、`results/fgsm/musk_v1`
- PGD crop：`results/pgd/sun_v1`、`results/pgd/pgd100_sun_v1`
- PGD Full：`results/pgd/pgdfull_sun_v1`、`results/pgd/pgdfull_musk_v1`、`results/pgd/pgdfull_musk_v2`
- Gaussian baseline 說明：`results/gaussian/`
- FGSM vs PGD 對照圖：`results/compare/`

白盒結論（既有）：在約 3–10/255 的 L∞ 下，InsightFace 自匹配 cosine 可降至 0.4 以下（見 `pgdfull_*` README / CSV）。

Transfer：僅完成單圖評估流程與平台能力，**沒有**正式 large-scale transfer 研究結論。

## Engineering Validation Only

- MI-FGSM implemented; minimal model smoke is pending the local runtime dependencies. No comparative research claim.

- Batch framework smoke / 中斷 pilot：只證明 runner 能跑，不是研究成果。
- `transfer_benchmark_v0_pilot` 已刪除，不要 resume、不要分析。
- Formal 100-image benchmark **尚未開始**。

## Open Research Questions

- Victim（FaceNet）verification threshold 的權威來源與校正
- Euclidean 距離的 baseline 分布
- 攻擊方法之間的 transferability
- 更多 victim models
- Targeted transfer
- Mapper 的貢獻（尚未接入，也尚未決定最終 abstraction）
- 最終報告 / presentation 設計
