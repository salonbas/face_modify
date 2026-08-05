# Adversarial Robustness — Face Analysis

以 **InsightFace**（`buffalo_l` / ArcFace）作為目標模型，研究白盒對抗性擾動對人臉辨識系統的影響。

實作從高斯基準 → FGSM → PGD（裁切）→ **PGD full（全圖，推薦）**，逐步推進攻擊效果與隱蔽性。

> **目前範圍**：僅白盒攻擊（surrogate 與 victim 為同一 ArcFace 權重的不同後端）。  
> **尚未包含**：黑箱攻擊、transfer attack、query-based attack、交通號誌攻擊。

> **核心成果（legacy results）**：在每像素變動僅 3–10/255 的條件下，使 InsightFace 自匹配 cosine 降至 0.4 門檻以下。詳見 `results/pgd/pgdfull_*`。

---

## 專案架構

```text
.
├── scripts/
│   ├── run_attack.py         # FGSM / PGD / PGD-full（--mode）
│   ├── run_gaussian.py       # 高斯噪聲基準
│   ├── verify_attack.py      # InsightFace 驗收
│   ├── attack_fgsm.py        # deprecated → run_attack.py
│   └── attack_gaussian.py    # deprecated → run_gaussian.py
├── advface/
│   ├── config.py             # 常數、project_root、providers
│   ├── paths.py              # results/<attack>/<run>/
│   ├── image_io.py
│   ├── models/
│   │   ├── insightface_app.py   # 評估：完整 FaceAnalysis
│   │   └── arcface_torch.py     # 攻擊：可微分 ArcFace
│   ├── attacks/
│   │   ├── common.py
│   │   ├── gaussian.py
│   │   ├── fgsm.py
│   │   └── pgd.py
│   ├── evaluation/
│   │   ├── similarity.py
│   │   └── attack_result.py
│   └── experiments/
│       └── output.py         # 統一輸出命名 + config.json
├── data/raw/
├── results/                  # 實驗輸出（含 legacy）
├── archive/docs/             # 過時文件
├── tests/
├── requirements.txt
└── README.md
```

### 兩條模型路徑（請勿混淆）

| 時機 | 模組 | 用途 |
|------|------|------|
| **攻擊時** | `advface.models.arcface_torch` | ONNX→Torch，對齊臉／全圖可微分，對輸入求梯度 |
| **評估時** | `advface.models.insightface_app` | 完整 FaceAnalysis（偵測→對齊→embedding）驗收 cosine |

---

## 環境安裝

```bash
cd Adversarial-Robustness-Face-Analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -e .          # 可選：以套件方式安裝 advface
```

### 模型權重

權重由 InsightFace 下載至 `~/.insightface/models/buffalo_l/`（含 `w600k_r50.onnx`、`det_10g.onnx` 等）。  
本 repo **不**存放權重本體。

裝置：

```bash
# 預設 CPU
export ADVFACE_PROVIDERS=cuda,cpu   # 可選 GPU
export ADVFACE_VERBOSE=1            # 顯示 InsightFace 載入 log
export ADVFACE_ROOT=/path/to/repo   # 可選：強制專案根目錄
```

---

## 攻擊方法

### 1. 高斯噪聲（基準）

```bash
python scripts/run_gaussian.py --img data/raw/sun.png --run-name gauss1
```

### 2. FGSM

```bash
python scripts/run_attack.py --mode fgsm --img data/raw/sun.png --run-name fgsm1
```

### 3. PGD crop（有接縫）

```bash
python scripts/run_attack.py --mode pgd --steps 20 --img data/raw/sun.png --run-name pgd20
```

### 4. PGD full（**推薦**）

```bash
python scripts/run_attack.py --mode pgd_full --steps 100 \
  --img data/raw/sun.png \
  --eps-list "0.005,0.008,0.010,0.012,0.015,0.020" \
  --run-name pgdfull_sun
```

---

## 新輸出命名規格

每次新 run 寫入 `results/<fgsm|pgd|gaussian>/<run-name>/`：

| 檔案 | 說明 |
|------|------|
| `base_<stem>.png` | 原圖備份 |
| `adv_<mode>_<stem>_eps_<eps>.png` | 對抗圖 |
| `<mode>_metrics.csv` | `eps,eps_255,steps,cosine,success` |
| `<mode>_cosine_chart.png` | cosine vs eps |
| `config.json` | mode、影像路徑與 hash、eps、steps、seed、model、det_size、provider、run_name |
| `<mode>_noise_eps_<eps>.png` | 可選擾動熱圖 |

`mode` 為 `fgsm` / `pgd` / `pgd_full`。

---

## Legacy results

`results/pgd/pgdfull_sun_v1`、`results/pgd/pgdfull_musk_v2` 等為重構前產出，**檔名與 CSV 欄位可能與新規格不同**。  
請勿覆蓋；驗證時請直接指定檔案路徑。詳見各目錄 README 與 `MIGRATION_NOTES.md`。

### 已知成功案例（legacy）

**sun.png / pgd_full / steps=100**

| eps | Cosine | 結果 |
|-----|--------|------|
| 0.012 | 0.339 | 成功 |
| 0.020 | -0.038 | 成功 |

**musk1.jpg / pgd_full / steps=200**

| eps | Cosine | 結果 |
|-----|--------|------|
| 0.040 | 0.347 | 成功 |

---

## 驗證

```bash
# 單張（legacy 路徑範例）
python scripts/verify_attack.py \
  --orig data/raw/sun.png \
  --adv  results/pgd/pgdfull_sun_v1/pgdfull_sun_eps_0.012.png

# 批次（新命名可用 --pattern "adv_*.png"）
python scripts/verify_attack.py \
  --orig data/raw/sun.png \
  --adv-dir results/pgd/some_new_run/ \
  --pattern "adv_*.png"
```

---

## 測試

```bash
pytest
```

---

## 文件

- `PROJECT_AUDIT.md` — 重構前盤點
- `MIGRATION_NOTES.md` — 腳本／import／輸出命名遷移
- `REFACTOR_REPORT.md` — 本輪重構報告
- `archive/docs/` — 過時本機指令與舊說明

---

## 依賴

見 `requirements.txt`。核心：`insightface`, `onnxruntime`, `onnx2torch`, `torch`, `opencv-python`, `matplotlib`。  
`pyproject.toml` 負責套件 metadata；執行時依賴以 `requirements.txt` 為準。
