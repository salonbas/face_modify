# Adversarial Face Research Platform

人臉辨識對抗攻擊研究平台：白盒攻擊、跨模型 transfer evaluation、單次實驗與 batch benchmark。

這是研究平台，不是一堆互不相關的實驗腳本。新增攻擊或 victim 時，應插入既有介面，而不是另寫一條 pipeline。

---

## Current Capabilities

- White-box：FGSM、PGD（crop）、PGD Full（全圖，推薦）、MI-FGSM（全圖）
- Transfer evaluation：ArcFace / InsightFace surrogate → FaceNet victim
- 單次實驗（圖 + HTML report）
- Batch benchmark（grid、checkpoint / resume、failure isolation、aggregate HTML）
- 統一 metrics 與 reporting（report 只讀結果，不重跑攻擊）

## Architecture

```text
advface/
├── models/          # EmbeddingModel：get_embedding(image)；各模型自管前處理
├── attacks/         # 攻擊實作 + registry（apply_attack）
├── evaluation/      # cosine / euclidean / success；與模型無關的擾動度量
├── experiments/     # canonical run_experiment + 單次輸出 / HTML
├── benchmark/       # grid、checkpoint、聚合、失敗隔離（不實作攻擊）
├── config.py        # 常數與路徑
├── paths.py         # results/<family>/<run>/
└── image_io.py
```

資料流：`image + attack config → adversarial image → surrogate / victims evaluation → ExperimentResult`。

Single-image 是 `1 × 1`；batch 是 `N × M`。兩者都呼叫 `run_experiment`。

---

## Canonical Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

權重：InsightFace 下載至 `~/.insightface/models/buffalo_l/`。FaceNet 權重在專案 `.cache/torch/`。Repo 不存放權重本體。

```bash
export ADVFACE_PROVIDERS=cuda,cpu   # 可選
export ADVFACE_VERBOSE=1
export ADVFACE_ROOT=/path/to/repo
```

### Single experiment — white-box sweep（多 eps、圖表）

```bash
python scripts/run_attack.py --mode fgsm --img data/raw/sun.png --run-name fgsm_sun
python scripts/run_attack.py --mode pgd_full --steps 100 --img data/raw/sun.png --run-name pgdfull_sun
python scripts/run_gaussian.py --img data/raw/sun.png --run-name gauss1
```

### Single experiment — transfer（攻擊 + surrogate/victim + HTML）

```bash
python scripts/run_transfer.py --image data/raw/sun.png --attack pgd_full --eps 0.040 --steps 200
python scripts/run_transfer.py --image data/raw/sun.png --attack fgsm --eps 0.040
python scripts/run_transfer.py --image data/raw/sun.png --attack mi_fgsm --eps 0.040 --steps 10 --momentum 1.0
```

### Batch experiment

```bash
python scripts/run_benchmark.py \
    --manifest data/benchmark/manifest.csv \
    --max-images 100 \
    --attacks fgsm pgd_full \
    --eps 0.005 0.01 0.02 0.03 0.04 \
    --pgd-steps 20 50 100 200 \
    --victim facenet_vggface2 \
    --run-name transfer_benchmark_v0
```

Dataset manifest：`python scripts/prepare_benchmark_dataset.py --max-images 200`

驗收既有對抗圖（不重新攻擊）：

```bash
python scripts/verify_attack.py --orig data/raw/sun.png --adv results/pgd/pgdfull_sun_v1/pgdfull_sun_eps_0.012.png
```

### Tests

```bash
pytest
# 含模型的 integration smoke（寫入 pytest tmp，不污染正式 results/）
pytest -m integration
```

---

## Current Research State

**Infrastructure（已具備）**

白盒 FGSM / PGD / PGD Full、ArcFace surrogate、FaceNet victim、單圖 transfer、batch runner（checkpoint / resume / failures）、統一 metrics、HTML report。

**Validated research evidence（值得保留的結果）**

`results/fgsm/*_v1`、`results/pgd/sun_v1`、`results/pgd/pgd100_sun_v1`、`results/pgd/pgdfull_*`、`results/gaussian/`、`results/compare/`。

已知白盒成功案例（legacy，未重跑）：

- sun.png / PGD Full / steps=100：eps=0.012 cosine=0.339；eps=0.020 cosine=-0.038
- musk1.jpg / PGD Full / steps=200：eps=0.040 cosine=0.347

**不是研究成果**

中斷的 `transfer_benchmark_v0_pilot` 已刪除。未跑 100-image formal benchmark。不得從 smoke / 中斷 pilot 下研究結論。

**尚未實作**

targeted attack、mapper、query-based black-box、DI / TI / ensemble、calibrated victim threshold。

權威狀態見 [`docs/RESEARCH_STATUS.md`](docs/RESEARCH_STATUS.md)。

---

## Adding an attack later

1. 在 `advface/attacks/` 實作
2. 在 `advface/attacks/registry.py` 註冊（`AttackSpec`）
3. 用 `scripts/run_transfer.py --attack <name>` 單圖測試，再用 `scripts/run_benchmark.py --attacks ...` 批量跑

不必改 evaluation、CSV、checkpoint、HTML 核心。
