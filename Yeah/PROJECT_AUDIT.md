# PROJECT AUDIT — Adversarial Robustness Face Analysis

> 產出日期：2026-08-05  
> 範圍：完整 repository 程式碼盤點與架構審查（**僅分析，未修改任何程式或實驗結果**）  
> 排除：`.git`、`.venv`、`__pycache__`、InsightFace 權重本體、大量輸出圖片內容本身

---

## 1. Executive Summary

本專案是一個規模精簡、結構已初步模組化的 **InsightFace 白盒人臉 embedding 對抗攻擊實驗平台**（約 14 個 Python 檔、~1150 LOC）。核心攻擊鏈已打通：

- **Victim / Surrogate（白盒）**：InsightFace `buffalo_l` 的 ArcFace `w600k_r50.onnx`，經 `onnx2torch` 轉成可微分 PyTorch 模型。
- **攻擊**：FGSM、裁切臉 PGD（`run_pgd`）、全圖可微分 PGD（`run_pgd_full`，推薦）。
- **目標**：untargeted **self-match dodging** — 最小化對抗圖與原圖 embedding 的 cosine similarity。
- **驗收**：再用完整 InsightFace `FaceAnalysis` pipeline（偵測→對齊→embedding）計算 cosine；門檻約 `0.4`。

**結論（延伸性）**：專案**可以**繼續延伸到黑箱 transfer demo，且不必先做大型重構。但目前與 InsightFace / ArcFace / 112×112 對齊強綁定，並存在**輸出檔名與文件／舊結果不一致**、**缺少 experiment config 保存**、**評估指標偏窄**等問題。建議以「薄介面 + 最小 transfer script」推進黑箱 Demo，而非一次到位的完整分層重構。

**最嚴重的三個問題（摘要）**：

1. **輸出命名已與文件、舊實驗結果脫節**（High）：現行程式寫出的 CSV／圖檔名與 `README.md`、`instruction.txt`、`results/*/README` 及既有 `results/` 不符，重跑後驗證指令會失效。
2. **攻擊最佳化路徑 ≠ 驗收路徑，且缺少攻擊過程內部 cosine 記錄**（High）：攻擊對 Torch ArcFace + 固定 landmark 最佳化，驗收走完整偵測；部分實驗（尤其 musk）出現 cosine 隨 eps **非單調**，可信度需謹慎解讀。
3. **可重現性不足**（High）：未保存完整 experiment config、無測試、無影像品質指標（PSNR/SSIM/L∞ 實測）、幾乎只用單圖實驗。

---

## 2. Repository Map

### 2.1 目錄結構（精簡）

```text
Adversarial-Robustness-Face-Analysis/
├── README.md                 # 專案說明與指令（部分過時）
├── instruction.txt           # 本機常用指令備忘
├── COMMANDS.local.md         # 過時的本機設定筆記（引用已不存在的腳本）
├── PROJECT_AUDIT.md          # 本報告
├── requirements.txt
├── pyproject.toml            # setuptools 套件宣告（dependencies 為空）
├── advface/                  # 核心套件
│   ├── __init__.py
│   ├── config.py
│   ├── paths.py
│   ├── image_io.py
│   ├── insightface_backend.py
│   ├── metrics.py
│   └── attacks/
│       ├── __init__.py
│       ├── _arcface.py       # ONNX→Torch ArcFace 載入
│       ├── gaussian.py
│       ├── fgsm.py
│       └── pgd.py
├── scripts/
│   ├── attack_gaussian.py
│   ├── attack_fgsm.py        # 真正入口：fgsm / pgd / pgd_full
│   └── verify_attack.py
├── data/
│   ├── raw/                  # 少量測試圖
│   └── adversarial/          # 空（僅 .gitkeep）
├── models/                   # 空（權重實際在 ~/.insightface）
├── notebooks/                # 空目錄
└── results/                  # 實驗輸出（gitignore）
    ├── fgsm/{sun_v1,musk_v1}/
    ├── pgd/{sun_v1,pgd100_sun_v1,pgdfull_sun_v1,pgdfull_musk_v1,pgdfull_musk_v2}/
    ├── gaussian/             # 僅 README，無實際 run 輸出
    └── compare/              # 有比較圖；產生腳本缺失
```

### 2.2 重要檔案盤點

| 路徑 | 用途 | 被誰引用 | 輸入 | 輸出 | 重要符號 | 狀態 |
|------|------|----------|------|------|----------|------|
| `scripts/attack_fgsm.py` | **主入口**：CLI 切換 fgsm/pgd/pgd_full | 使用者執行 | `--img`, `--mode`, `--eps-list`, `--steps`… | `results/{fgsm\|pgd}/<run>/` | `main()` | **Core** |
| `scripts/attack_gaussian.py` | 高斯噪聲基準實驗入口 | 使用者執行 | eps 範圍、seed | `results/gaussian/<run>/` | `main()` | Experiment |
| `scripts/verify_attack.py` | 攻擊後驗收（cosine / L2） | 使用者執行 | `--orig`, `--adv` / `--adv-dir` | stdout | `verify_pair()` | Core / Utility |
| `advface/attacks/pgd.py` | PGD / PGD-full 核心與 demo 存檔 | `attack_fgsm.py` | BGR 圖、eps list、steps | 對抗圖、CSV、chart | `run_pgd`, `run_pgd_full`, `run_pgd_*_demo`, `_save_pgd_outputs` | **Core** |
| `advface/attacks/fgsm.py` | FGSM 核心、共用工具、demo | `attack_fgsm.py`, `pgd.py` | 同上 | FGSM 結果 | `run_fgsm`, `run_fgsm_demo`, `parse_eps_list`, `_paste_aligned_patch` | **Core** |
| `advface/attacks/_arcface.py` | ArcFace ONNX→Torch 載入與快取 | `fgsm.py`, `pgd.py` | `~/.insightface/.../w600k_r50.onnx` | Torch `nn.Module` | `load_arcface_torch` | **Core** |
| `advface/attacks/gaussian.py` | 高斯實驗邏輯 | `attack_gaussian.py` | 圖、eps 整數範圍 | CSV、折線圖、可選噪聲圖 | `run_gaussian_noise_experiment` | Experiment |
| `advface/insightface_backend.py` | FaceAnalysis 建立、選臉、抽 embedding | 幾乎所有攻擊／驗證 | BGR / path | 512-d embedding | `create_face_app`, `pick_best_face`, `get_embedding_from_bgr` | **Core** |
| `advface/metrics.py` | 相似度／距離 | attacks、verify | vectors | float | `cosine_similarity`, `euclidean_distance` | Core |
| `advface/image_io.py` | 讀圖、stem | attacks | path | BGR ndarray | `load_bgr`, `image_stem` | Utility |
| `advface/config.py` | 常數與 provider | scripts、backend | env `ADVFACE_*` | 常數 | `FACE_MODEL_NAME`, `DEFAULT_*`, `insightface_providers` | Configuration |
| `advface/paths.py` | run 輸出目錄 | scripts | attack_type, run_name | Path | `make_run_dir` | Utility |
| `advface/__init__.py` | 套件版本 `0.2.0` | — | — | — | `__version__` | Utility |
| `advface/attacks/__init__.py` | 空套件標記 | — | — | — | — | Utility |
| `requirements.txt` | 依賴 | pip | — | — | — | Configuration |
| `pyproject.toml` | package metadata；`dependencies=[]` | setuptools | — | — | — | Configuration |
| `README.md` | 文件 | — | — | — | — | Configuration（部分過時） |
| `instruction.txt` | 指令備忘 | — | — | — | — | Utility |
| `COMMANDS.local.md` | 舊 Windows 指令；引用 `insightface_compare.py`（不存在） | — | — | — | — | **Legacy** |
| `data/raw/*.jpg|png` | 實驗圖片 | scripts | — | — | sun, musk1/2, face2/3 | Data |
| `data/adversarial/` | 預留；空 | — | — | — | — | Data（未使用） |
| `models/` | 預留；空（權重在 home） | — | — | — | — | Unknown / 未使用 |
| `notebooks/` | 空 | — | — | — | — | Unknown |
| `results/**` | 實驗產物 | verify、人工檢視 | — | png/csv | — | Output |
| `results/compare/COMPARE_README.txt` | 引用不存在的 `scripts/compare_attacks.py` | — | — | — | — | Legacy / 文件漂移 |
| `results/*/README.txt` | 各攻擊說明；檔名慣例偏舊版 | — | — | — | — | Configuration / Legacy |

### 2.3 特別標註（任務要求的關鍵位置）

| 關注點 | 位置 |
|--------|------|
| **專案入口點** | `scripts/attack_fgsm.py`（`--mode fgsm\|pgd\|pgd_full`）；基準：`scripts/attack_gaussian.py`；驗收：`scripts/verify_attack.py` |
| **模型載入** | 評估用：`advface/insightface_backend.py::create_face_app`；攻擊用：`advface/attacks/_arcface.py::load_arcface_torch`（`w600k_r50.onnx`） |
| **圖片 preprocessing** | `cv2.imread` BGR（`image_io.load_bgr`）；攻擊時 `COLOR_BGR2RGB`；對齊：`insightface.utils.face_align.norm_crop2(..., image_size=112)`；ArcFace 正規化：`(x - 127.5) / 127.5` |
| **embedding 提取（攻擊）** | `F.normalize(model(...), dim=1)`（`fgsm.py` / `pgd.py`） |
| **embedding 提取（驗收）** | `face.normed_embedding`（`get_embedding_from_bgr`） |
| **similarity** | `advface/metrics.py::cosine_similarity`；驗收另算 `euclidean_distance` |
| **PGD 實作** | `advface/attacks/pgd.py::run_pgd`, `run_pgd_full` |
| **loss** | `(_emb(x) * emb_ref).sum()` ≡ 正規化後 cosine |
| **參數** | `eps`（[0,1] 比例，實際像素 `eps*255`）；`step_size = eps_px / steps`；`steps` CLI 預設 **20**（非寫死 200）；det_size 預設 `(640,640)` |
| **圖片輸出與評估** | `_save_pgd_outputs` / `run_fgsm_demo`；另 `verify_attack.py` |
| **Hard-coded** | 模型名 `buffalo_l`、ONNX 檔名、112、127.5、threshold `0.4`、預設圖 `data/raw/sun.png`、CSV/圖檔命名字串 |

### 2.4 資料與模型（非程式）

| 資產 | 說明 |
|------|------|
| `data/raw/sun.png` | 主要成功案例圖（大檔 ~5.7MB） |
| `data/raw/musk1.jpg` | musk 實驗圖；`pgdfull_musk_v2` 在 eps=0.040、steps=200 達 cosine≈0.347 |
| `data/raw/musk2.jpg` | **與 musk1 完全相同**（MD5 一致）→ Duplicate |
| `data/raw/face2.jpg`, `face3.jpg` | 存在；程式預設幾乎不用（`DEFAULT_IMAGE2` 未被引用） |
| `~/.insightface/models/buffalo_l/` | 實際權重：`w600k_r50.onnx`（辨識）、`det_10g.onnx`（偵測）等 |
| `results/pgd/pgdfull_*` | 核心研究成果；**不要覆蓋** |

---

## 3. Current Pipeline

### 3.1 端到端流程（以推薦的 `pgd_full` 為準）

執行指令（文件慣例）：

```bash
python scripts/attack_fgsm.py --mode pgd_full --steps 200 \
  --img data/raw/musk1.jpg \
  --eps-list "0.010,0.020,0.030,0.040" \
  --run-name pgdfull_musk
```

**實際呼叫鏈與順序：**

1. **CLI 解析** — `scripts/attack_fgsm.py::main`  
   - `ensure_project_dirs()`（`config.py`）  
   - `make_run_dir(attack_type="pgd", run_name=...)`（`paths.py`）→ `results/pgd/<run>/`  
   - 呼叫 `run_pgd_full_demo(...)`（`advface/attacks/pgd.py`）

2. **載入圖片** — `load_bgr(img_path)`（`image_io.py`）  
   - OpenCV `IMREAD_COLOR` → **BGR uint8**

3. **Seed（弱）** — `torch.manual_seed(0)`, `np.random.seed(0)`（無 `cudnn.deterministic`；PGD 本身無隨機起步）

4. **載入 FaceAnalysis（偵測用）** — `create_face_app(det_size)`  
   - `FaceAnalysis(name="buffalo_l", providers=...)`  
   - 預設 provider：`CPUExecutionProvider`（可用 `ADVFACE_PROVIDERS=cuda,cpu`）

5. **人臉偵測與對齊矩陣** — `run_pgd_full` 內：  
   - `pick_best_face(app.get(img_bgr))`  
   - `face_align.norm_crop2(..., image_size=112)` → 取得仿射矩陣 `M`（此模式不使用裁切像素本身做攻擊）

6. **全圖 tensor + 可微分裁切**  
   - `x_full`：BGR→RGB，float，`[1,3,H,W]`，值域約 `[0,255]`  
   - `_cv2M_to_torch_theta(M, H, W)` → `affine_grid` → 固定 `grid`  
   - **landmark / grid 在整個攻擊中固定**（以乾淨圖對齊為準）

7. **載入可微分 ArcFace** — `load_arcface_torch(device)`  
   - 複製 ONNX 到 temp → `onnx2torch.convert` → `eval()`、`requires_grad_(False)`  
   - 全域快取 `_TORCH_ARC`

8. **參考 embedding**  
   - `_emb_full(x)`：`grid_sample` → `(crop-127.5)/127.5` → model → `F.normalize`  
   - `emb_ref = _emb_full(x_full).detach()`（乾淨圖）

9. **對每個 eps 做 PGD**（見第 4 節）  
   - 輸出 `attacked_bgr`（uint8 BGR）

10. **驗收與存檔** — `_save_pgd_outputs`  
    - 再抽 `emb_base = get_embedding_from_bgr(app, img_bgr)`（完整 InsightFace）  
    - 對每張 adv 再 `get_embedding_from_bgr` → `cosine_similarity`  
    - 寫 CSV、PNG、matplotlib 折線圖（threshold 線 0.4）

11. **（可選）事後驗證** — `scripts/verify_attack.py`  
    - 再次載入模型、讀原圖／對抗圖、算 cosine 與 Euclidean

### 3.2 FGSM / 裁切 PGD 差異（同入口不同函式）

| 模式 | 函式 | 擾動作用域 | 貼回方式 | 預設 steps |
|------|------|------------|----------|------------|
| fgsm | `run_fgsm` | 112×112 crop | `_paste_aligned_patch`（有接縫） | 1（單步） |
| pgd | `run_pgd` | 112×112 crop | 同上 | CLI 預設 20 |
| pgd_full | `run_pgd_full` | 全圖 `delta` | `x+delta`（無接縫） | CLI 預設 20；文件建議 100/200 |

### 3.3 攻擊類型判定

| 問題 | 答案（依程式） |
|------|----------------|
| Targeted 或 Untargeted？ | **Untargeted**（對自身 embedding 推遠，無目標身份） |
| 目標是什麼？ | **拉遠 embedding**（降低 self-match cosine），不是改分類 logits（ArcFace 此處當特徵提取器） |
| 每步優化的 loss？ | \(L = \langle e(x), e(x_0)\rangle\)（L2-normalized → cosine）；對 `x`/`delta` **最小化** |
| 是否標準 PGD？ | **大致符合 L∞ PGD（sign 梯度 + 投影）**；但無 random start、step size 固定為 `eps/steps`、無 momentum；屬常見教學／實作變體，非完整 Madry 預設配置 |
| 200 iterations 是否寫死？ | **否**。CLI `--steps` 預設 20；README/instruction 對 musk 建議 200、對 sun 建議 100 |
| Epsilon 是否限制？ | **是**：crop 模式投影到 `[x0±eps*255]`；full 模式 `delta.clamp(±eps*255)`，再 `clamp(x+delta, 0, 255)` |
| 存檔數值範圍？ | 攻擊中 float `[0,255]` → `astype(uint8)` → `cv2.imwrite`；再讀回為量化後的 uint8，**與攻擊中 float 狀態不完全一致** |

---

## 4. PGD Implementation Review

### 4.1 核心更新式（`run_pgd` / `run_pgd_full`）

對正規化 embedding，loss：

```python
(_emb(x_adv) * emb_ref).sum().backward()
```

更新（最小化 cosine）：

```python
x_adv = x_adv.detach() - step_size * x_adv.grad.detach().sign()
# 或 delta -= step_size * sign(grad)
```

投影：

- crop：`x_adv ∈ [x0 - eps_px, x0 + eps_px] ∩ [0, 255]`
- full：`delta ∈ [-eps_px, eps_px]`，`x_adv = clamp(x_full + delta, 0, 255)`

`step_size = eps_px / steps`（例如 eps=0.04、steps=200 → 每步約 0.051 像素）

### 4.2 與標準 PGD 的符合度

| 項目 | 狀態 | 說明 |
|------|------|------|
| 對輸入求梯度 | OK | `requires_grad_(True)` 在 `x_adv`/`delta` |
| 模型權重凍結 | OK | `p.requires_grad_(False)` |
| sign 更新 | OK | 最小化 cosine 用減號，方向正確 |
| L∞ projection | OK | 以原圖為中心（crop）或以 delta ball（full） |
| 像素 clip | OK | `[0,255]` |
| 每步清梯度 | OK | 每步新建 requires_grad 的 tensor，無明顯 accumulation |
| retain_graph | 未使用 | 合理 |
| Random start | 無 | 常見可選項；非錯誤 |
| 與 Madry 預設 α | 不同 | 此處 α=`ε/T`；Madry 常固定 α（如 2/255）並允許多次撞邊界 |

### 4.3 需注意／需確認的實作細節

1. **攻擊 loss 在 Torch 路徑；報告的 cosine 在 InsightFace 路徑**  
   - `_save_pgd_outputs` 才算 cosine；攻擊迴圈內 `PgdResult.cosine = nan`。  
   - 這使「優化目標」與「論文式指標」可能有數值落差（通常仍同向）。

2. **`_cv2M_to_torch_theta` 正確性** — **需要確認**  
   - 全圖攻擊依賴 OpenCV 仿射 ↔ PyTorch `affine_grid` 轉換。程式有推導註解，但本輪未做數值對齊測試（例如比較 `grid_sample` crop 與 `norm_crop2` 像素差）。若轉換有偏，梯度仍可能有效（結果顯示攻擊成功），但「無接縫全圖」的幾何精度需驗證。

3. **uint8 量化**  
   - 存檔後再驗收，可能讓邊界成功案例的 cosine 略回升。

4. **musk 結果非單調**（實驗現象，可能與實作／偵測交互有關）  
   - `pgdfull_musk_v1`：eps 0.012→0.516，0.015→0.750  
   - `pgdfull_musk_v2`：eps 0.010→0.625，0.015→0.745  
   - 可能原因：重新偵測 landmark、局部最優、或全圖擾動改變偵測框。**需要確認**根因。

---

## 5. Correctness Risks

| ID | 嚴重度 | 問題 | 位置 | 理由 |
|----|--------|------|------|------|
| C1 | **High** | 輸出檔名／CSV 欄位與文件、舊結果不一致 | `pgd.py::_save_pgd_outputs`, `fgsm.py::run_fgsm_demo` vs `results/**`, `README.md`, `instruction.txt` | 現碼寫 `{prefix}_metrics.csv`、`{prefix}_chart.png`、`{prefix}_base_{stem}.png`、`{prefix}_{stem}_{eps}.png`；舊結果為 `*_cosine_metrics.csv`、`*_cosine_line_chart.png`、`base_*.png`、`*eps_*.png`；CSV 欄位 `cosine` vs `cosine_similarity`。重跑會讓文件中的 verify 路徑失效。 |
| C2 | **High** | 攻擊路徑與驗收路徑不一致且未記錄攻擊內 cosine | `pgd.py` / `fgsm.py` vs `get_embedding_from_bgr` | 白盒優化 Torch+固定對齊；驗收重新偵測。缺少 side-by-side 記錄，難以診斷失敗是「攻擊不夠」還是「偵測漂移」。 |
| C3 | **High** | 實驗可重現資訊不足 | demos 未寫 config.json | 未系統性保存 eps、steps、seed、model、commit、provider、影像 hash。 |
| C4 | **Medium** | RGB/BGR 轉換多處手動 | `fgsm.py`, `pgd.py`, `image_io.py` | 目前攻擊路徑有顯式 BGR→RGB；InsightFace 吃 BGR。邏輯看起來一致，但易在擴充第二模型時搞錯。 |
| C5 | **Medium** | 存檔量化可能削弱攻擊 | `astype(np.uint8)` + `cv2.imwrite` | 浮點對抗樣本量化後效果可能變差；未做「記憶體內 vs 讀檔後」對照。 |
| C6 | **Medium** | `_cv2M_to_torch_theta` 未驗證 | `pgd.py:130-148` | 幾何轉換若有系統偏差，full 攻擊的梯度對齊假設不完整。 |
| C7 | **Medium** | cosine 非單調（musk） | `results/pgd/pgdfull_musk_v*` | 削弱「更大 eps → 更強攻擊」敘事；需排查偵測穩定性。 |
| C8 | **Medium** | 幾乎單圖／少圖實驗 | `data/raw` 少量圖；musk1=musk2 | 不足以支持泛化結論；自動駕駛遷移敘事目前僅類比。 |
| C9 | **Medium** | 無 PSNR/SSIM/LPIPS/實測 L∞ | 全 repo | 「人眼幾乎不可見」僅靠目視與 eps 敘述，缺量化證據。 |
| C10 | **Medium** | `project_root()` 依賴 cwd | `config.py:41-43` | 不在 repo 根目錄執行會把目錄建錯處；scripts 只修了 `sys.path`。 |
| C11 | **Low** | FGSM 無相對 x0 的顯式投影以外的多步邏輯 | `fgsm.py` | 單步 FGSM 可接受；但與 PGD 工具函式耦合在同一檔。 |
| C12 | **Low** | Provider 預設 CPU、攻擊可用 CUDA | `config.insightface_providers`, `_arcface` | 偵測在 CPU、梯度在 GPU 時行為仍可用，但效能／數值環境需記錄。 |
| C13 | **Low** | 成功門檻 0.4 hard-coded 多處 | `verify_attack.py`, charts, `GAUSSIAN_FAIL_THRESHOLD` | 與 InsightFace 常見驗證門檻相近，但未引用官方設定來源。 |
| C14 | **Low** | 無 random seed 完整控制 | demos | CUDA 非確定性可能造成微小差異。 |
| C15 | **Cosmetic** | README 結構圖過時 | `README.md` | 仍寫 PGD 在 `fgsm.py`；實際已拆到 `pgd.py`。 |
| C16 | **Cosmetic** | `COMMANDS.local.md` 指向不存在腳本 | `insightface_compare.py` | 易誤導。 |
| C17 | **High（文件／工具缺失）** | `compare_attacks.py` 不存在 | `results/compare/COMPARE_README.txt` | 比較圖存在但無法從 repo 重現產生流程。 |

**關於「gradient / detach / normalization / clipping 錯誤」的總評**：  
目前 PGD/FGSM 的梯度方向、凍結權重、L∞ 投影與 `[0,255]` clip **未發現 Critical 級別的明顯實作錯誤**；既有 sun/musk 成功數字也支持「攻擊大致有效」。主要風險在 **評估管線一致性、輸出契約漂移、可重現性與指標完整度**，而非 sign 正負號反了這類致命錯誤。

**Data leakage / 同模型攻擊與評估**：  
白盒設定下，surrogate 與 victim 本質是同一 ArcFace 權重（Torch vs ORT 後端）。這對白盒研究可接受，但若宣稱「對 InsightFace 系統攻擊」，需講清楚：梯度來自辨識子網路，不含偵測器端到端可微。黑箱 transfer 時必須換獨立 victim，否則沒有黑箱意義。

---

## 6. Architecture Review

### 6.1 目前耦合情況

單次 `pgd_full` demo 實際混合了：

| 職責 | 是否混在同一流程 | 位置 |
|------|------------------|------|
| config | 部分集中、部分散落 | `config.py` + CLI + 魔術數字 |
| model loading | 兩套 | `_arcface.py` + `insightface_backend.py` |
| preprocessing | 散落 | attacks 內聯 |
| attack algorithm | 尚可 | `run_pgd*` / `run_fgsm` |
| loss | 內聯一字元組 | 無獨立 objectives 模組 |
| evaluation | 與 demo I/O 綁死 | `_save_pgd_outputs` |
| visualization | 同檔 | matplotlib in attacks |
| experiment runner | CLI scripts | `attack_fgsm.py` |
| file I/O | 同 demo | cv2/csv |

整體評價：**對目前畢專規模「勉強合格、略擠」**——比單檔 prototype 好，但離可替換 victim/attack 的研究平台還差一層薄介面。

### 6.2 理想分層 vs 現況映射

建議的目標結構與現有檔案對應：

| 目標層 | 現有可遷入 |
|--------|------------|
| `configs/` | `advface/config.py` + 未來 YAML/JSON run config |
| `src/models/` | `_arcface.py`, `insightface_backend.py` → adapter |
| `src/attacks/` | `fgsm.py`, `pgd.py`, `gaussian.py`（純演算法） |
| `src/data/` | `image_io.py` + 對齊／normalize |
| `src/objectives/` | 抽出 cosine / targeted losses |
| `src/evaluation/` | `metrics.py` + 影像品質 + success 判定 |
| `src/experiments/` | demos 的「跑實驗+存檔」部分 |
| `scripts/` | 薄 CLI |
| `tests/` | **目前不存在** |
| `outputs/` | 即 `results/` |

### 6.3 應抽出的 interface（最小集合）

```text
VictimModel / FeatureExtractor
  - preprocess(image) -> tensor
  - embed(tensor) -> embedding
  - embed_numpy(bgr) -> embedding   # 黑箱查詢友好

Attack
  - run(model, image, objective, budget) -> AdvExample

Objective
  - loss(emb_adv, emb_ref | emb_target) -> scalar

Evaluator
  - similarity, image_quality, success(threshold)
```

### 6.4 不值得過度設計的部分

- 完整 plugin 系統、多後端 registry、Hydra 大型配置。
- 現在就抽象「交通號誌模型」的完整 dataset pipeline（尚無該資料與模型）。
- 把 gaussian / fgsm / pgd 硬塞進複雜 class hierarchy。

### 6.5 對畢專最合理的重構深度

**建議：漸進式「薄重構」，不要大搬家。**

1. 先統一輸出契約與 config 落盤（不改演算法）。  
2. 再抽出 `FeatureExtractor` protocol，讓 transfer demo 能插第二個模型。  
3. 攻擊函式保持函式式即可。  
4. 等黑箱 Demo 跑通、需求變清楚後，再決定是否搬到 `src/` 目錄樹。

---

## 7. Dead Code and Technical Debt

### 可以安全刪除（建議人工再確認一次後）

| 項目 | 理由 |
|------|------|
| `data/raw/musk2.jpg` | 與 `musk1.jpg` MD5 相同 |
| `COMMANDS.local.md` 中對 `insightface_compare.py` 的段落 | 腳本不存在（檔案本身 gitignore，可本地清理） |

### 建議封存

| 項目 | 理由 |
|------|------|
| `results/pgd/sun_v1`（裁切 PGD，有接縫） | 歷史對照有用；非最終方法 |
| `results/pgd/pgdfull_musk_v1` | 被 v2 取代的探索 run；保留作為失敗／非單調案例 |
| `results/compare/` 舊圖 | 產生腳本已失蹤；封存產物即可 |

### 應該合併

| 項目 | 理由 |
|------|------|
| `fgsm.py` 與 `pgd.py` 的 demo 存檔邏輯 | CSV/圖表/noise map 高度重複 |
| `parse_eps_list`, `_paste_aligned_patch`, noise map | 已在 fgsm，被 pgd import；可再抽到 `attacks/utils.py` 或 `advface/viz.py` |
| `verify_attack.py` 與 `_save_pgd_outputs` 的 cosine 驗收 | 重複載入與重複指標 |

### 應該保留但重新命名

| 項目 | 建議 |
|------|------|
| `scripts/attack_fgsm.py` | 實際是 multi-attack launcher → 如 `scripts/run_attack.py` |
| 輸出檔名 | 與文件統一：`*_cosine_metrics.csv` 或全面改文件 |
| `pgd` vs `pgd_full` 檔名 prefix | 舊結果用 `pgdfull_`（無底線）混用，需統一 |

### 無法確認，需要人工確認

| 項目 | 原因 |
|------|------|
| `get_embedding_from_path` | 目前無引用；可能預留給外部 notebook |
| `DEFAULT_IMAGE2` | 未使用；可能預留 pair 實驗 |
| `data/adversarial/`, `models/` | 空目錄意圖不明（本地權重 vs 產出存放） |
| `euclidean_distance` 除 verify 外用途 | 功能保留合理，但無標準化報告 |
| `compare_attacks.py` 是否曾存在於其他 branch | 本工作區無此檔 |
| `_cv2M_to_torch_theta` 數值正確性 | 需寫小測試對齊 OpenCV warp |
| gaussian 實驗是否跑過但未提交 | `results/gaussian/` 僅 README |

### 其他技術債清單

- 未使用：`get_embedding_from_path`、`DEFAULT_IMAGE2`（僅定義）
- 重複 preprocessing / similarity：攻擊檔與 verify 各自實作
- Hard-coded path：`DEFAULT_IMAGE = "data/raw/sun.png"`；ONNX 路徑依 `~/.insightface`
- 魔術數字：112、127.5、0.4、255
- Debug：無大量 commented-out code；有可控的 stdout 抑制（`ADVFACE_VERBOSE`）
- Notebook：目錄空，無不可重現 notebook
- 測試：無
- `pyproject.toml` dependencies 為空，與 `requirements.txt` 分裂
- `advface/attacks/__init__.py` 註解仍像「之後新增 PGD」，但 PGD 已存在

---

## 8. Black-box Extension Readiness

### 8.1 Demo 目標對照

> 在 surrogate 上產 adv，丟給未參與梯度的 victim，測 transferability。

| 評估項 | 判斷 |
|--------|------|
| 可直接重用 | 圖片 I/O、PGD/FGSM 產樣、cosine 指標、`verify_attack` 流程、`results/` 目錄慣例 |
| 與 InsightFace 綁太死 | `_arcface.load_arcface_torch`、`FaceAnalysis`、112 crop、512-d、`(x-127.5)/127.5` |
| 能否分離 surrogate / victim | **目前不能**；攻擊與驗收都指向同一 buffalo_l 權重 |
| 不同 preprocessing | 需 per-model preprocess；adv 應以「共用像素圖（uint8 RGB/BGR 約定清楚）」為交換格式 |
| 不同 embedding 維度 | **不要**直接比跨模型 embedding；應比「各自 self-match」或「各自與 gallery 的判定」 |
| 兩個人臉模型應比什麼 | 建議：各模型上 `cos(e(orig), e(adv))`；以及是否跌破該模型門檻（dodging）。Targeted impersonation 需目標身份圖 |
| targeted vs untargeted transfer | **先做 untargeted dodging transfer**（與現有 loss 一致，實作最小） |
| 現有 PGD adv 能否直接測另一模型 | **可以作為第一步**：把 `results/pgd/pgdfull_*/*.png` 丟進第二模型算 self-match — 零改攻擊程式 |
| 需新增的最小功能 | 第二模型 loader；transfer 評估 script；成功定義；簡單結果表 |

### 8.2 準備度評分（主觀）

| 面向 | 分數 (1–5) | 說明 |
|------|------------|------|
| 白盒攻擊可重用性 | 4 | `run_pgd_full` 可產樣 |
| 模型可替換性 | 2 | 無 adapter |
| 評估可擴充性 | 2 | 僅 cosine |
| 實驗管理 | 2 | 無 config 落盤 |
| 黑箱 Demo 可行性（短時間） | 4 | 用現有 adv 圖 + 新 evaluate script 即可起步 |

---

## 9. Minimal Black-box Demo Plan

**原則**：不大型重構；優先重用；短時間可跑通。

### 步驟 A（0.5–1 天）：零攻擊改動的 transfer 探測

1. 選定已有成功 adv，例如：  
   - `results/pgd/pgdfull_sun_v1/pgdfull_sun_eps_0.012.png`  
   - `results/pgd/pgdfull_musk_v2/pgdfull_musk1_eps_0.040.png`
2. 新增 **一個** script（建議名：`scripts/eval_transfer.py`，本輪不實作）：  
   - 載入 orig + adv  
   - Victim A：現有 `create_face_app`（buffalo_l）→ cosine  
   - Victim B：另一人臉模型（候選見下）→ cosine  
3. 輸出表格：orig/adv 路徑、A/B cosine before（orig-orig=1）、after、是否跌破門檻。

**Victim B 候選（由易到難）**：

1. InsightFace 另一 pack（若可得不同 recognition 權重）— **需要確認**環境是否方便取得  
2. 另一開源人臉辨識（如不同 ArcFace 實作 / MagFace / FaceNet）— 需額外依賴  
3. 同一 ONNX 但只證明「ORT vs Torch」— **不算黑箱**，勿當正式 Demo

### 步驟 B（1–2 天）：用現有 PGD 當場產樣再 transfer

1. 呼叫既有 `run_pgd_full`（surrogate = Torch buffalo_l）  
2. 記憶體內或存檔後送 Victim B  
3. 記錄 surrogate 攻擊成功與否、victim 是否 transfer 成功

### 步驟 C：Demo 輸出清單（最低）

- original image  
- adversarial image  
- surrogate attack result（cosine、success）  
- victim result（cosine、success）  
- similarity before/after（各模型 self-match）  
- `transfer_success` 布林（建議定義：surrogate 成功 **且** victim 也跌破其門檻）

### 成功定義建議（寫進 Demo README）

```text
surrogate_success := cos_s(orig, adv) < T_s
victim_success    := cos_v(orig, adv) < T_v
transfer_success  := surrogate_success and victim_success
```

門檻可先都用 0.4，再註明 Victim B 應使用其慣用門檻（**需要確認**各模型官方建議）。

### 明確不做（本階段）

- Query-based 黑箱優化  
- 交通號誌模型  
- 完整 `src/` 目錄搬家  
- 批次百張資料集

---

## 10. Refactoring Proposal

### 10.1 建議採納的分層（精簡版，而非一次建齊範例樹）

```text
advface/
  models/
    base.py                 # Protocol: embed_bgr / embed_tensor
    insightface_app.py      # 現 insightface_backend
    arcface_torch.py        # 現 _arcface
  attacks/
    fgsm.py / pgd.py        # 只留純演算法（吃 model protocol）
  objectives/
    embedding.py            # cosine minimize / targeted maximize
  eval/
    similarity.py
    quality.py              # PSNR/SSIM/Linf（後加）
    success.py
  experiment/
    runner.py               # 存 config + metrics + paths
  image_io.py / config.py / paths.py
scripts/
  run_attack.py             # 現 attack_fgsm.py
  eval_transfer.py          # 黑箱 Demo
  verify_attack.py
configs/
  pgd_full_sun.yaml         # 可選
```

### 10.2 現有檔案遷移建議（未來執行時）

| 現有 | 建議 |
|------|------|
| `_arcface.py` | `models/arcface_torch.py` |
| `insightface_backend.py` | `models/insightface_app.py` |
| `fgsm.py` 工具函式 | `attacks/common.py` 或 `advface/viz.py` |
| demo 存檔段 | `experiment/runner.py` |
| `metrics.py` | 擴成 `eval/` |

### 10.3 如何避免「為架構耽誤研究」

1. **規則**：沒有第二個 victim 之前，不搬家目錄。  
2. 先加 `eval_transfer.py` + 小 `Protocol`，用 adapter 包舊函式。  
3. 任何重構必須能重跑 `pgdfull_sun` 得到同趨勢結果（允許 uint8 誤差）。  
4. 舊 `results/` 只讀封存；新 run 用新命名契約並寫 `config.json`。

---

## 11. Prioritized Action List

### P0 — 黑箱 Demo 前必須處理

1. **釐清並固定「輸出檔名契約」**（改碼或改文件二選一，避免半新半舊）。  
2. **定義 transfer 成功標準與 victim 模型選型**（含依賴是否裝得動）。  
3. **用現有 adv 圖先做 Victim B 的 self-match 探測**，確認 Demo 故事是否說得通。  
4. **不要覆蓋**既有 `results/pgd/pgdfull_*` 成功案例。

### P1 — 建議近期處理

1. 攻擊結束時同時記錄 **Torch 內 cosine** 與 **InsightFace 驗收 cosine**。  
2. 每次 run 寫入 `config.json`（img hash、eps、steps、seed、providers、mode、git commit 若有）。  
3. 增加簡單影像指標：實測 L∞、L2、PSNR（SSIM 可後加）。  
4. 將 `scripts/attack_fgsm.py` 更名／文件化為多模式入口。  
5. 修復 README 結構圖（PGD 在 `pgd.py`）。  
6. 驗證 `_cv2M_to_torch_theta`（小單元測試）。  
7. 刪除或標註 `musk2` 重複檔。

### P2 — 正式擴大研究前

1. 多圖／小資料集批次 runner。  
2. `FeatureExtractor` adapter + 第二、第三模型。  
3. 補齊或移除 `compare_attacks` 文件承諾。  
4. 基本 pytest（metrics、projection、affine 對齊）。  
5. `requirements.txt` 與 `pyproject.toml` 依賴對齊。  
6. 調查 musk 非單調 cosine 根因。

### P3 — 可延後

1. 完整 `project/src/...` 目錄美化。  
2. Query-based 黑箱、交通號誌遷移。  
3. Hydra/MLflow 等實驗平台。  
4. LPIPS、大規模可視化 dashboard。  
5. Notebook 教學化。

---

## 12. Open Questions

1. **`_cv2M_to_torch_theta` 與 OpenCV `norm_crop2` 的像素級誤差有多大？**（需要確認）  
2. **musk 實驗 cosine 非單調的主因是偵測漂移還是優化不穩？**（需要確認）  
3. **黑箱 Victim B 選定哪一個模型？** 授權、權重、preprocess 是否可在現有環境安裝？（需要確認）  
4. **輸出命名要以「相容舊 results」為準，還是接受 breaking change 並更新文件？**（需要產品／報告決策）  
5. **`compare_attacks.py` 是否曾存在於其他機器或 branch？** 比較圖如何產生？（需要確認）  
6. **Gaussian 實驗是否有未納入 repo 的結果？** 目前僅 README。（需要確認）  
7. **成功門檻 0.4 是否應對齊 InsightFace 官方 verification threshold（模型相關）？**（需要確認）  
8. **長期目標若轉向交通號誌：是否保留 face pipeline 作為方法論載體，另開 vision 模組？**（研究方向）  
9. **攻擊時 `ADVFACE_PROVIDERS` 與 Torch CUDA 裝置不一致時，論文／報告要如何描述硬體設定？**（需要確認）  
10. **是否需要 targeted impersonation（讓 A 被認成 B）作為下一階段？** 與現有 untargeted dodging 故事不同。

---

## Appendix A — 既有關鍵實驗結果（摘錄，未改動）

**sun.png / pgd_full / steps=100**（`results/pgd/pgdfull_sun_v1/pgd_full_cosine_metrics.csv`）：

| eps | cosine | 解讀（門檻 0.4） |
|-----|--------|------------------|
| 0.012 | 0.339 | 成功 |
| 0.020 | -0.038 | 成功 |

**musk1.jpg / pgd_full / steps=200**（`results/pgd/pgdfull_musk_v2/pgd_full_cosine_metrics.csv`）：

| eps | cosine | 解讀 |
|-----|--------|------|
| 0.030 | 0.413 | 失敗 |
| 0.040 | 0.347 | 成功 |

---

## Appendix B — 引用索引（核心程式位置）

| 主題 | 檔案:符號 |
|------|-----------|
| CLI 入口 | `scripts/attack_fgsm.py:main` |
| PGD crop | `advface/attacks/pgd.py:run_pgd` |
| PGD full | `advface/attacks/pgd.py:run_pgd_full` |
| PGD 存檔 | `advface/attacks/pgd.py:_save_pgd_outputs` |
| FGSM | `advface/attacks/fgsm.py:run_fgsm` |
| ArcFace Torch | `advface/attacks/_arcface.py:load_arcface_torch` |
| FaceAnalysis | `advface/insightface_backend.py:create_face_app` |
| Embedding | `advface/insightface_backend.py:get_embedding_from_bgr` |
| Cosine | `advface/metrics.py:cosine_similarity` |
| 驗收 | `scripts/verify_attack.py:verify_pair` |
| 預設參數 | `advface/config.py` |

---

*本報告僅供架構與研究規劃；未對 repository 程式碼或 `results/` 做任何修改。*
