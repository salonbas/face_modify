# Refactor Report — Phase 1 (White-box cleanup)

## 1. Summary

完成第一階段保守重構：模型／攻擊共用／評估／實驗輸出分層，統一新 run 命名與 `config.json`，修正 `project_root`，清理重複圖與過時文件，並加入最小 pytest。演算法與既有 `results/pgd/pgdfull_*` 研究結果未改動。

## 2. Before / After Structure

**Before（精簡）**

```text
advface/{config,paths,image_io,insightface_backend,metrics,attacks/{_arcface,fgsm,pgd,gaussian}}
scripts/{attack_fgsm,attack_gaussian,verify_attack}
models/ notebooks/ data/adversarial/   # 空
```

**After**

```text
advface/
  models/{insightface_app,arcface_torch}
  attacks/{common,fgsm,pgd,gaussian}
  evaluation/{similarity,attack_result}
  experiments/output.py
  config.py paths.py image_io.py
  (+ metrics.py / insightface_backend.py / attacks/_arcface.py 相容層)
scripts/{run_attack,run_gaussian,verify_attack,+deprecated wrappers}
tests/
archive/docs/
```

## 3. Files Moved

| From | To |
|------|----|
| `advface/insightface_backend.py`（邏輯） | `advface/models/insightface_app.py` |
| `advface/attacks/_arcface.py`（邏輯） | `advface/models/arcface_torch.py` |
| `advface/metrics.py`（邏輯） | `advface/evaluation/similarity.py` |
| `COMMANDS.local.md` | `archive/docs/` |
| `instruction.txt` | `archive/docs/` |
| `results/compare/COMPARE_README.txt` | `archive/docs/` |

## 4. Files Renamed

| From | To |
|------|----|
| `scripts/attack_fgsm.py` | `scripts/run_attack.py`（舊名變 deprecated wrapper） |
| `scripts/attack_gaussian.py` | `scripts/run_gaussian.py`（同上） |

## 5. Files Deleted

- `data/raw/musk2.jpg`（MD5 = `musk1.jpg`）
- 空目錄：`models/`、`notebooks/`、`data/adversarial/`

## 6. Files Archived

- `archive/docs/COMMANDS.local.md`
- `archive/docs/instruction.txt`
- `archive/docs/COMPARE_README.txt`

## 7. Shared Logic Extracted

- `advface/attacks/common.py`：`parse_eps_list`、貼回對齊臉、noise map、L∞ 投影 helpers
- `advface/experiments/output.py`：統一檔名、CSV、chart、`config.json`、驗收存檔
- `advface/evaluation/attack_result.py`：`AttackResult` dataclass + success 判定

## 8. Output Contract

新 run：

```text
base_<stem>.png
adv_<mode>_<stem>_eps_<eps>.png
<mode>_metrics.csv          # eps,eps_255,steps,cosine,success
<mode>_cosine_chart.png
config.json
```

`config.json` 含：attack_mode、source_image、source_image_hash、eps_list、steps、seed、model_name、detector_size、provider、run_name、similarity_threshold。

Legacy `results/` **未**重新命名。

## 9. Config Changes

- `project_root()`：優先 `ADVFACE_ROOT`，否則由 `advface/config.py` 路徑推導
- `make_run_dir()`：基於 `project_root()`，不依賴 cwd
- 集中常數：`ALIGNED_FACE_SIZE=112`、`PIXEL_MAX=255`、`ARCFACE_INPUT_MEAN/SCALE=127.5`、`SIMILARITY_THRESHOLD=0.4`、`ARCFACE_ONNX_FILENAME`
- `ensure_project_dirs()` 不再建立空的 `models/`、`notebooks/`、`data/adversarial/`
- 套件版本 → `0.3.0`

## 10. Test Results

```text
python -m pytest -q
..............                                                         [100%]
14 passed in ~2.1s
```

涵蓋：metrics、PGD projection、paths／cwd、affine OpenCV↔Torch 對齊（通過，MAE 在容差內）。

## 11. Smoke Test Results

| Run | 結果 |
|-----|------|
| FGSM `refactor_smoke_test` eps=0.01 | cosine≈0.866；產出新檔名 + `config.json` |
| PGD full `refactor_smoke_test_pgd` steps=2 eps=0.01 | cosine≈0.747；全圖 L∞≤3（budget≈2.55，uint8） |
| Gaussian `refactor_smoke_test` eps=10 | cosine≈0.971 |
| Legacy verify `pgdfull_sun_eps_0.012.png` | cosine≈0.3391（與附錄一致） |

未覆蓋：`pgdfull_sun_v1`、`pgdfull_musk_v2`。

## 12. Compatibility Notes

- 舊 CLI wrapper 可用（印 deprecated）。
- 舊 import 路徑仍 re-export。
- 新／舊 output 檔名並存；verify 支援直接路徑與過濾非對抗圖。
- FGSM／PGD crop 貼回後，全圖 L∞ 可能因仿射重採樣大於 crop 上的 eps（原行為）；`pgd_full` 約束正常。

## 13. Remaining Technical Debt

- 攻擊路徑（Torch+固定 landmark）與驗收路徑（完整偵測）仍分離；缺少攻擊內 cosine 並記
- musk 非單調 cosine 根因未查
- 無 PSNR/SSIM／實測 L∞ 報告欄
- `compare_attacks.py` 仍缺失
- `get_embedding_from_path`、`DEFAULT_IMAGE2` 保留未用
- `pyproject.toml` dependencies 仍空（刻意；以 requirements.txt 為準）
- Gaussian 尚未寫 `config.json`（可下一輪對齊）
- 單圖實驗為主，資料集擴充未做

## 14. Recommended Next Step

黑箱 Demo 最適合從：

1. **既有 adv 圖** + 新 `scripts/eval_transfer.py`（讀 orig/adv → victim A/B self-match）  
2. 或呼叫 `run_pgd_full` 產樣後，以 **uint8 BGR 圖** 為交換格式丟第二模型  

先抽薄 `FeatureExtractor` protocol 包住 `get_embedding_from_bgr`，勿一次大搬家。本輪**未**實作黑箱。
