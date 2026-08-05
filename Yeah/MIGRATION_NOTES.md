# Migration Notes

## Scripts

| 舊 | 新 |
|----|----|
| `scripts/attack_fgsm.py` | `scripts/run_attack.py` |
| `scripts/attack_gaussian.py` | `scripts/run_gaussian.py` |
| `scripts/verify_attack.py` | 同名（已更新 import／過濾新檔名） |

舊 script 仍可執行，但會印出 `[DEPRECATED]` 並轉呼叫新入口。

## Imports

| 舊 | 新 |
|----|----|
| `advface.insightface_backend` | `advface.models.insightface_app` |
| `advface.attacks._arcface` | `advface.models.arcface_torch` |
| `advface.metrics` | `advface.evaluation.similarity` |
| `advface.attacks.fgsm.parse_eps_list` 等共用工具 | `advface.attacks.common` |

舊路徑仍保留為相容層（re-export），建議新程式改用新路徑。

## Output naming

### 新 run（重構後）

```text
base_<stem>.png
adv_<mode>_<stem>_eps_<eps>.png
<mode>_metrics.csv          # eps,eps_255,steps,cosine,success
<mode>_cosine_chart.png
config.json
```

`mode` ∈ `{fgsm, pgd, pgd_full}`；gaussian 使用 `gaussian_metrics.csv` / `gaussian_cosine_chart.png`。

### Legacy results（不受影響）

`results/pgd/pgdfull_*`、`results/fgsm/*_v1` 等**未重新命名、未搬移、未覆蓋**。  
舊檔名例如 `pgdfull_sun_eps_0.012.png`、`pgd_full_cosine_metrics.csv` 仍可用 `verify_attack.py --adv <path>` 驗證。

## 已移除

| 項目 | 原因 |
|------|------|
| `data/raw/musk2.jpg` | 與 `musk1.jpg` MD5 完全相同 |
| 空目錄 `models/`、`notebooks/`、`data/adversarial/` | 無用途且易誤解 |

## 已封存

| 項目 | 位置 |
|------|------|
| `COMMANDS.local.md` | `archive/docs/` |
| `instruction.txt` | `archive/docs/` |
| `results/compare/COMPARE_README.txt` | `archive/docs/` |

## Breaking changes（文件層）

- 主入口腳本名稱變更（有 deprecated wrapper）。
- **新**實驗輸出檔名與 CSV 欄位變更；舊 results 不變。
- `project_root()` 改由套件路徑推導，不再依賴 cwd。
