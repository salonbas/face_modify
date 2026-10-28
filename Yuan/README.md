# Yuan：MI-FGSM reference 比較實驗

本資料夾由 `Yeah/advface` 複製研究程式，修改僅在 Yuan。保留 t.txt。
複製整個小型 advface package 是為了保留 attack、model、evaluation 的相依關係；
不複製歷史報告、資料集、結果或大型權重。

## 實驗

- self：最小化 cos(ArcFace(B_adv), ArcFace(B))，沿用原本 baseline。
- pair：最小化 cos(ArcFace(B_adv), ArcFace(A))；A 使用自己的 landmarks 對齊。
- 兩組固定相同 B、epsilon、steps、alpha、momentum、seed，無 Mask、無 random start。
- FaceNet 只做評估；ArcFace 與 FaceNet 分別使用正式 far_1e-2 校正門檻。
- 每組 PNG 儲存並讀回後，才計算 A–B_adv 驗證結果與實際 L∞。
- 一個 identity 的兩組都完成才加入統計；失敗另存 summary.json。
- ASR 分母為各模型 clean A–B 通過門檻的完成 pairs，並同時列出 requested / completed。
- 所有輸出為 exploratory，不能直接取代 Yeah 正式結果。

## 使用

從專案根目錄執行（使用已裝好研究依賴的 Python）：

```powershell
python Yuan/scripts/compare_reference.py --check-only
python Yuan/scripts/compare_reference.py --limit 1 --steps 5
python Yuan/scripts/compare_reference.py
```

5 steps 是工程 smoke，不能與 200 steps 正式數字比較。
默认使用相鄰 Yeah 作為只讀 data-root；也可指向有完整資料的 Yeah：

```powershell
python Yuan/scripts/compare_reference.py --data-root C:/senior_project/face_modify/Yeah --check-only
```

必要輸入：

- `<data-root>/data/datasets/lfw/evaluation_splits/attack_dev.csv`，欄位為
  identity_id、reference_image、probe_image；影像路徑相對 data-root。
- split 列出的全部 A/B 照片（或以 --limit 限制 development 樣本）。
- `<data-root>/results/calibration/{arcface_lfw_v1,facenet_lfw_v1}/`
  下的 thresholds.json 與 summary.json。
- InsightFace 完整模型放在 `<data-root>/models/insightface/buffalo_l/`。
- FaceNet 權重放在 `<data-root>/models/facenet/20180402-114759-vggface2.pt`。
- Python：torch、torchvision、numpy、OpenCV、InsightFace、onnxruntime、onnx2torch、
  facenet-pytorch 等。`requirements-source.txt` 是 Yeah 原始環境快照，
  不是本機已驗證的 Windows 安裝組合；不要直接假設所有鎖定版本都可安裝。

腳本不會自動下载模型。輸出固定在 `Yuan/results/reference_<timestamp>/`，包含
config.json、results.csv、summary.json 與兩組 B_adv PNG。
目前僅有 RGB MAE，不宣稱已評估 SSIM、LPIPS、DISTS 或不可察覺性。

## 檢查

```powershell
python -m unittest discover -s Yuan/tests -p test_reference_runner.py -v
python -m pytest Yuan/tests/test_mi_fgsm.py Yuan/tests/test_pair_reference.py
```

第一組不需要攻擊依賴，驗證 CSV、ASR 分母與統計；第二組是從 Yeah 複製的
MI-FGSM baseline 測試及獨立 A 對齊、epsilon 檢查，需要 pytest 和研究環境。
缺少這些依賴時，模型相關測試會 skip，不能當成已通過。
2026-10-13 已建立 `Yuan/.venv`、接入 Yeah 的完整資料與權重，並通過 12 項測試。
Windows 環境使用官方 InsightFace 0.7.3 sdist，省略未使用的 C++ mesh renderer；
FaceAnalysis、face_align、辨識 ONNX 權重保持原版。
onnx2torch 改以 ModelProto 輸入避免 Windows NamedTemporaryFile 鎖定。
完整實驗與 smoke 分開保存，結果位置見 `results/reference_<timestamp>/`。

本機使用方式（專案根目錄）：

```powershell
./Yuan/.venv/Scripts/python.exe Yuan/scripts/compare_reference.py --check-only
./Yuan/.venv/Scripts/python.exe Yuan/scripts/compare_reference.py --limit 1 --steps 5
./Yuan/.venv/Scripts/python.exe Yuan/scripts/compare_reference.py --threads 4
./Yuan/.venv/Scripts/python.exe Yuan/scripts/summarize_reference.py Yuan/results/reference_<timestamp>
```

CUDA 可用時自動使用 GPU；每組輸出也記錄初始／最終 objective cosine 與初始梯度 L1。
實際套件版本、程式雜湊與比較報告由 summarize_reference.py 保存。

## 2026-10-13 完整比較

本機結果：`results/reference_20261013_145019_577894/REPORT.md`。
20/20 identities 完成、0 failures；12 項測試與 40 張 PNG 的擾動檢查通過。
MI-FGSM：eps=0.04、200 steps、alpha=0.0002、momentum=1；RTX 3050、Torch 2.5.1+cu121。

| Objective | ArcFace | FaceNet crossing / 20 | FaceNet crossing / clean-valid 19 |
| --- | ---: | ---: | ---: |
| self-reference | 20/20 (100%) | 8/20 (40%) | 8/19 (42.1%) |
| A-reference | 20/20 (100%) | 13/20 (65%) | 13/19 (68.4%) |

Richard Myers 原始 FaceNet A–B cosine=0.186393，低於 0.404789 門檻，
不屬於可被攻擊造成跨界的 clean-valid pair；全 20 人比例與有效樣本比例分開列出。
新 loss 在 20/20 人均下降，平均 objective cosine=0.665900 → -0.481600。
這是固定 development sample 的單次比較；SSIM/LPIPS/DISTS 尚未評估。
Windows 相容性修補與套件版本見該 run 的 REPORT.md、requirements-tested.txt、insightface-build.json。
先前 smoke 與中止的 CPU run 保留在獨立目錄，不納入本表。

## 加入 mask 的比較

執行：`./Yuan/.venv/Scripts/python.exe Yuan/scripts/compare_reference.py --mask --threads 4`。
每個 identity 的 self/pair 共用同一張 B 的 mask；設定為 68-point landmarks +
SLIC（n_segments=100、compactness=10、sigma=0），tv_weight=0。
mask 僅遮罩梯度，A-reference 與 self-reference 的其他參數完全相同。
landmark 模型直接共用已載入的本機 FaceAnalysis，避免重複下載和載入模型。

完整報告：`results/reference_mask_20261013_154854_692028/MASK_COMPARISON.md`。
20/20 完成、0 failures、13 項測試通過；所有 40 張 masked PNG 的 mask 外差值為 0。

| Objective | Mask | ArcFace /20 | FaceNet /20 | SSIM RGB |
| --- | --- | ---: | ---: | ---: |
| self | 無 | 20/20 (100%) | 8/20 (40%) | 0.934743 |
| A-reference | 無 | 20/20 (100%) | 13/20 (65%) | 0.934300 |
| self | 有 | 8/20 (40%) | 6/20 (30%) | 0.974616 |
| A-reference | 有 | 20/20 (100%) | 9/20 (45%) | 0.974454 |

FaceNet clean-valid 為 19 人（Richard Myers 原始 pair 不通過）：mask self 6/19=31.6%，
mask A-reference 9/19=47.4%。SSIM 統一從 80 張已保存 PNG 原生 RGB 重算；
LPIPS、DISTS 未評估，不能將較高 SSIM 等同於人眼不可察覺。
這是固定 development sample 的單次觀察，不能推廣為一般成功率。

四組評估可重建（不重跑 attack）：

```powershell
./Yuan/.venv/Scripts/python.exe Yuan/scripts/compare_mask_report.py --baseline Yuan/results/reference_20261013_145019_577894 --masked Yuan/results/reference_mask_20261013_154854_692028
```

## MI-FGSM + DI

DI 使用固定尺寸 ArcFace 變體：對 B_adv 的可微分對齊 crop 做 RGB 正規化後，
以 p=0.7 隨機縮小到原尺寸的 90%–100%，隨機 padding 回 112×112。
padding 的正規化值為 0（RGB=127.5）；A/reference 不做隨機變換。
每步先從 DI loss 回傳到整張 B 的梯度，再套用可選 mask、L1 normalization、
momentum、sign update 與 epsilon 投影。成功率仍由未變換的 decoded PNG 評估。
這是 [DI 論文](https://arxiv.org/abs/1803.06978) 的 fixed-input-size adaptation，
不是原論文 ImageNet 設定的逐字重現。

```powershell
./Yuan/.venv/Scripts/python.exe Yuan/scripts/compare_reference.py --di --threads 4
./Yuan/.venv/Scripts/python.exe Yuan/scripts/compare_reference.py --di --mask --threads 4
```

可指定 `--di-probability` 與 `--di-min-scale`；`--seed` 同時固定獨立 CPU DI RNG。
self/pair、mask/unmasked 共用相同 seed 與 DI 抽樣設定。
預設不加 `--di` 時保留原攻擊行為且不消耗 DI RNG。
config.json 保存 DI 參數，results.csv 保存實際變換的步數及原始／最終未變換的 objective cosine。
21 項測試涵蓋 disabled identity、RNG reproducibility、gradient flow、A 的獨立對齊、
DI+mask 外像素不變與 epsilon budget。

### 2026-10-14 DI 完整結果

無 mask：`results/reference_di_20261014_153851_517219/`。
有 mask：`results/reference_di_mask_20261014_155754_945242/`。
八組完整報告在後者的 `DI_COMPARISON.md`；數值在 `di_comparison.json`、
`di_ssim_metrics.csv`。兩個 DI run 均為 20/20 completed、0 failures。

以下四組都使用 A-reference：

| Method | ArcFace /20 | FaceNet /20 | FaceNet /clean-valid 19 | SSIM RGB |
| --- | ---: | ---: | ---: | ---: |
| MI-FGSM | 20/20 (100%) | 13/20 (65%) | 13/19 (68.4%) | 0.934300 |
| MI-FGSM + DI | 20/20 (100%) | 14/20 (70%) | 14/19 (73.7%) | 0.933692 |
| MI-FGSM + Mask | 20/20 (100%) | 9/20 (45%) | 9/19 (47.4%) | 0.974454 |
| MI-FGSM + DI + Mask | 20/20 (100%) | 12/20 (60%) | 12/19 (63.2%) | 0.974080 |

DI 無 mask 新增 Jennifer Rodriguez 成功；DI+Mask 新增 Meryl Streep、Paul Bremer、
Jennifer Rodriguez；兩種 A-reference 比較均無 success→fail。
原 self-reference 的 FaceNet /20 則為無 mask 40%→50%、有 mask 30%→40%；
其逐人得失與白箱數值詳見完整報告。

21 項測試通過。從全部 160 張保存 PNG 重算 SSIM，並核對 tensor epsilon、
serialized epsilon+quantization、所有 mask 外 delta=0、clean cosine 與 frozen split 一致。
DI seed=0 在每組 200 步中實際使用 126 次非 identity 縮放/pad。
這是固定 20 人、單一 seed／DI 參數設定的比較，不是最佳化後的普遍成功率；
LPIPS/DISTS 未評估。SSIM 比較顯示加 DI 後略降，不能宣稱 DI 同時改善影像品質。

重建八組報告（不重跑攻擊）：

```powershell
./Yuan/.venv/Scripts/python.exe Yuan/scripts/compare_di_report.py --baseline Yuan/results/reference_20261013_145019_577894 --baseline-mask Yuan/results/reference_mask_20261013_154854_692028 --di Yuan/results/reference_di_20261014_153851_517219 --di-mask Yuan/results/reference_di_mask_20261014_155754_945242
```

## MI-FGSM + DI + TI（2026-10-26）

TI 參考 [CVPR 2019 TI 論文](https://arxiv.org/abs/1904.02884)，在原圖空間的
raw gradient 上套用正規化 Gaussian depthwise convolution；RGB 各通道獨立。
固定首輪設定為 kernel=5×5、sigma=1 原圖像素；邊界採 replicate padding，
這些是本次明確記錄的 implementation choices。

順序：**DI loss backward → TI 平滑 → mask → L1 normalization → momentum → sign → epsilon projection**。
TI 不直接模糊原圖。mask 放在 TI 之後，因此平滑後仍不允許 mask 外更新。
不加 `--ti` 時不平滑梯度；kernel-size=1 是 exact identity control。

```powershell
./Yuan/.venv/Scripts/python.exe Yuan/scripts/compare_reference.py --di --ti --pair-only --threads 4
./Yuan/.venv/Scripts/python.exe Yuan/scripts/compare_reference.py --di --ti --mask --pair-only --threads 4
```

可調 `--ti-kernel-size`（正奇數）、`--ti-sigma`（正且有限）。
`--pair-only` 只跑目前主用的 A-reference；不加則保留 self/pair 兩組行為。
TI 也可單獨以 `--ti` 使用，不必開啟 DI。
本輪未依 FaceNet 分數調參，使用相同 frozen 20 人、eps=0.04、200 steps、
alpha=0.0002、momentum=1、DI p=0.7、DI scale=0.9–1、seed=0。

| Method（全為 A-reference） | ArcFace /20 | FaceNet /20 | FaceNet /clean-valid 19 | SSIM RGB | RGB MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| MI-FGSM + DI | 20/20 (100%) | 14/20 (70%) | 14/19 (73.7%) | 0.933692 | 2.267205 |
| MI-FGSM + DI + TI | 20/20 (100%) | 16/20 (80%) | 16/19 (84.2%) | 0.933951 | 2.434873 |
| MI-FGSM + DI + Mask | 20/20 (100%) | 12/20 (60%) | 12/19 (63.2%) | 0.974080 | 0.984219 |
| MI-FGSM + DI + TI + Mask | 20/20 (100%) | 12/20 (60%) | 12/19 (63.2%) | 0.974968 | 1.013600 |

TI 無 mask 新增 Michelle Collins、Jorge Rodolfo Canicoba Corral 成功，無成功轉失敗。
有 mask 的成功身分集合不變。SSIM 雖小幅增加，RGB MAE 亦增加，不能宣稱整體畫質改善。
Richard Myers 原始 FaceNet pair 不通過門檻，分別列出全20與有效19人的比例。

本輪兩個 run 各 20/20 完成、0 failures。36 項測試通過；80 張保存 PNG 的
tensor / serialized epsilon 檢查、mask 外 delta=0、clean cosine、DI 抽樣次數與
mask 一致性檢查均通過。LPIPS/DISTS 未評估，結果限於固定樣本與單一 seed/參數。

無 mask：`results/reference_di_ti_20261026_235225_545727/`。
有 mask：`results/reference_di_ti_mask_20261026_235415_117069/`。
完整比較在後者 `TI_COMPARISON.md`、`ti_comparison.json`、`ti_ssim_metrics.csv`。

重建比較（不重跑 attack）：

```powershell
./Yuan/.venv/Scripts/python.exe Yuan/scripts/compare_ti_report.py --di Yuan/results/reference_di_20261014_153851_517219 --di-mask Yuan/results/reference_di_mask_20261014_155754_945242 --ti Yuan/results/reference_di_ti_20261026_235225_545727 --ti-mask Yuan/results/reference_di_ti_mask_20261026_235415_117069
```
