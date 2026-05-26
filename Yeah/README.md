# Adversarial Robustness — Face Analysis

以 **InsightFace**（`buffalo_l` / ArcFace）作為目標模型，研究對抗性擾動對人臉辨識系統的影響。
實作從高斯基準→ FGSM → PGD → 全圖無接縫 PGD，逐步推進攻擊效果與隱蔽性。

> **核心成果**：在每像素變動僅 3–10/255（人眼幾乎不可見）的條件下，
> 使 InsightFace 的自匹配 cosine 從 ~1.0 降至 0.35 以下（門檻 0.4），
> 達成「欺騙 AI 辨識、人眼無法察覺」的目標。

---

## 專案結構

```
.
├── scripts/
│   ├── attack_gaussian.py    # 高斯噪聲基準實驗
│   ├── attack_fgsm.py        # FGSM / PGD / 全圖PGD 攻擊（--mode 切換）
│   └── verify_attack.py      # 用 InsightFace 驗證攻擊是否成功
├── advface/
│   ├── config.py
│   ├── paths.py
│   ├── image_io.py
│   ├── insightface_backend.py
│   ├── metrics.py
│   └── attacks/
│       ├── gaussian.py
│       └── fgsm.py           # FGSM + PGD + 全圖PGD（pgd_full）
├── data/raw/                 # 原始圖片
├── results/
│   ├── fgsm/                 # FGSM 實驗結果
│   ├── pgd/                  # PGD / pgd_full 實驗結果
│   └── gaussian/             # 高斯基準結果
├── requirements.txt
└── README.md
```

---

## 環境安裝

```bash
cd Adversarial-Robustness-Face-Analysis
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

---

## 攻擊方法一覽

### 1. 高斯噪聲（基準）

在每個像素加上隨機 N(0, σ²) 噪聲。**結論：InsightFace 對隨機噪聲相當穩健**，cosine 幾乎不受影響，印證了「有方向性的梯度攻擊」才是關鍵。

```bash
python scripts/attack_gaussian.py --img data/raw/sun.png --run-name gauss1
```

### 2. FGSM（單步梯度攻擊）

對 InsightFace 使用的 ArcFace ONNX（`w600k_r50.onnx`）透過 `onnx2torch` 轉換為可微分 PyTorch 模型，對 112×112 對齊人臉計算梯度，單步更新：

```
x_adv = x - eps×255 × sign(∂cosine/∂x)
```

```bash
python scripts/attack_fgsm.py --mode fgsm --img data/raw/sun.png --run-name fgsm1
```

### 3. PGD（多步迭代，裁切後貼回）

FGSM 的多步版本，共 N 步，每步步長 = eps×255/N，每步後投影回 L∞ ball。
效果比 FGSM 強約 10 倍，但對齊臉貼回原圖時有**矩形接縫**，肉眼可見。

```bash
python scripts/attack_fgsm.py --mode pgd --steps 20 --img data/raw/sun.png --run-name pgd20
```

### 4. pgd_full（全圖 PGD，**推薦**）

關鍵改進：以 PyTorch `affine_grid` + `grid_sample` 取代 OpenCV 裁切，
讓梯度直接流回整張圖的 delta，**完全消除接縫問題**。

```
delta 維度 = 原圖大小
forward: x_adv → 可微分仿射裁切 → ArcFace → cosine
backward: 梯度流回 delta（整張圖）
```

```bash
python scripts/attack_fgsm.py --mode pgd_full --steps 100 \
  --img data/raw/sun.png \
  --eps-list "0.005,0.008,0.010,0.012,0.015,0.020" \
  --run-name pgdfull_sun
```

---

## 實驗結果

### sun.png（pgd_full，steps=100）

| eps | 每像素最大變動 | Cosine | 結果 |
|-----|-------------|--------|------|
| 0.005 | 1.3/255 | 0.716 | 攻擊失敗 |
| 0.010 | 2.6/255 | 0.440 | 攻擊失敗 |
| **0.012** | **3.1/255** | **0.339** | **攻擊成功 ✓** |
| 0.015 | 3.8/255 | 0.181 | 攻擊成功 ✓ |
| 0.020 | 5.1/255 | -0.038 | 攻擊成功 ✓ |

### musk1.jpg（pgd_full，steps=200）

| eps | 每像素最大變動 | Cosine | 結果 |
|-----|-------------|--------|------|
| 0.030 | 7.6/255 | 0.413 | 攻擊失敗 |
| **0.040** | **10.2/255** | **0.347** | **攻擊成功 ✓** |

> **觀察**：不同人臉對攻擊的抵抗能力不同（musk 需要更大的 eps），
> 這本身也是值得研究的課題。

---

## 驗證攻擊效果

```bash
# 單張驗證
python scripts/verify_attack.py \
  --orig data/raw/sun.png \
  --adv  results/pgd/pgdfull_sun_v1/pgdfull_sun_eps_0.012.png

# 批次驗證整個資料夾
python scripts/verify_attack.py \
  --orig data/raw/sun.png \
  --adv-dir results/pgd/pgdfull_sun_v1/ \
  --pattern "pgdfull_sun_eps_*.png"
```

---

## InsightFace 裝置設定

```bash
# 預設 CPU（最通用）
# 啟用 GPU（需 CUDA 環境）
export ADVFACE_PROVIDERS=cuda,cpu

# 顯示 InsightFace 載入詳細 log（除錯用）
export ADVFACE_VERBOSE=1
```

---

## 依賴

見 `requirements.txt`。核心套件：`insightface`, `onnxruntime`, `onnx2torch`, `torch`, `opencv-python`, `matplotlib`。
