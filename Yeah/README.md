# Adversarial Face Research Platform

本專案目前的正式研究主線是 LFW face-verification adversarial attack：以
ArcFace / InsightFace `buffalo_l` 作為 white-box surrogate，FaceNet VGGFace2
作為不參與梯度的 black-box victim，比較 PGD Full 與 MI-FGSM 的 verification
dodging、transfer 與 perceptual quality。

## 新組員先讀

1. [研究狀態與可引用結果](docs/RESEARCH_STATUS.md)
2. [正式實驗 protocol](docs/EXPERIMENT_PROTOCOL.md)
3. [Reproducibility 與 research asset setup](docs/REPRODUCIBILITY.md)
4. [指標定義](docs/METRICS.md)、[校準與門檻](docs/CALIBRATION_AND_THRESHOLDS.md)、[實驗設定](docs/EXPERIMENT_CONFIG.md)
5. [腳本與架構索引](docs/SCRIPT_INVENTORY.md)、[artifact inventory](docs/ARTIFACT_INVENTORY.md)、[clean-clone 實測紀錄](docs/CLEAN_CLONE_AUDIT.md)

`docs/RESEARCH_STATUS.md` 是正式研究結論與結果 provenance 的最高權威。
若文件、HTML report 或舊輸出與它不一致，請以它為準。

## 正式結果位置

- calibration：`results/calibration/{arcface_lfw_v1,facenet_lfw_v1}/`
- PGD source robustness：`results/robustness/lfw_verification_pgd_v1/`
- transfer：`results/transfer/{pgd_arcface_to_facenet_verification_v1,mi_fgsm_arcface_to_facenet_verification_v1}/`
- Perceptual Evaluation v2：`results/perceptual/lfw_development_transfer_v2/`

Frozen 20-person v3 four-method baseline 的主要 summary 位於
`results/transfer/lfw_attack_dev_20_four_method_v3/`。五模型 Development 100
仍在 preflight 階段，未執行正式 benchmark；Momentum ablation 與其結果分開。
`archive/` 與 `results/archive/` 的內容只供歷史追溯，**不得作為正式分析依據**。

## 開發與驗證

```bash
python3.12 -m venv .venv
bash scripts/install_research_dependencies.sh
.venv/bin/python scripts/setup_research_assets.py --accept-upstream-research-terms
bash scripts/setup_five_model_sources.sh
# 將 AdaFace / MagFace 作者權重放在 checkpoint_manifest.json 指定路徑；詳見下方文件。
.venv/bin/python scripts/setup_research_assets.py --verify-only
.venv/bin/python scripts/check_research_environment.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/generate_research_specs.py --check
```

`setup_research_assets.py` 會自動取得 LFW、InsightFace ArcFace／landmark、
FaceNet 與 MobileFaceNet，並驗證 SHA-256。AdaFace／MagFace 作者權重的
人工取得與校驗步驟見 [資產手冊](docs/REPRODUCIBILITY.md)。所有模型權重、
原始圖片、cache 都保留在 Git 之外。若缺任何資產，環境檢查會失敗。

## Repository workflow 與正式入口

`advface/` 放共用攻擊、模型載入、校準與評估邏輯；`scripts/` 放 CLI；
`data/datasets/lfw/` 的 CSV 固定 cohort 與 pair；`results/calibration/` 的 JSON
是門檻權威來源；`results/transfer/` 放各 protocol 的結果；`docs/` 放研究定義。
先讀 [實驗設定](docs/EXPERIMENT_CONFIG.md) 與
[門檻規格](docs/CALIBRATION_AND_THRESHOLDS.md)，再依 protocol 選入口：

| 任務 | 入口 |
| --- | --- |
| 安裝資產及檢查 | `setup_research_assets.py`、`check_research_environment.py` |
| v2 五模型校準 | `calibrate_lfw_v2_victims.py`；正式門檻已固定，勿隨意重跑 |
| v3 frozen 20 生成與評估 | `run_lfw_two_phase_v4.py --attack pgd_full\|mi_fgsm --phase generate\|evaluate\|regression` |
| v3 結果彙總與機制分析 | `aggregate_lfw_four_method_v3.py`、`analyze_lfw_four_method_mechanisms.py` |
| 五模型 Development 100 | `run_lfw_five_model_benchmark.py --preflight`；目前 gate 為 hard stop |
| 1 人端到端 smoke | `smoke_lfw_five_model_pgd_mask.py --max-identities 1` |

跑 smoke 前先確認 [split audit](data/datasets/lfw/evaluation_splits/v2/audit.json)
和五模型 v2 門檻；smoke 只讀 Development 前一人，不能當成 100-person
正式結果。歷史 v1 五人、v2 20 人、v3 frozen 20、Momentum ablation 與
五模型 v2 Development/OOS 使用不同 cohort 或門檻，不能合併統計。
所有 35 個入口的角色、保留理由與相容路徑見
[腳本索引](docs/SCRIPT_INVENTORY.md)。

Perceptual Evaluation v2 只從已持久化的正式 artifact 重建評估，不會重跑 attack：

```bash
.venv/bin/python scripts/evaluate_lfw_perceptual_v2.py
```

探索性單圖、sun / musk、FGSM、smoke、舊 benchmark 與 historical meeting 材料
已退出 active research navigation；其可追溯材料見 [archive/README.md](archive/README.md)。
