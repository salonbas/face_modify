# Research Status

本檔是正式研究狀態與結果 provenance 的唯一入口。需要引用數字時，先查
下方 Authority Table；`results/archive/`、單圖輸出與 smoke 不能作為正式
baseline 或結論。

## 研究目標

研究 face-verification adversarial attack。目前主線是 **black-box
transferability** 與 **imperceptibility**：在 ArcFace surrogate 上產生擾動，
再觀察未參與梯度的 FaceNet victim。另有待延伸的 **targeted impersonation**
分支；目前正式結果均為 untargeted verification dodging，不可混稱。

## 正式實驗 Protocol

- **Dataset / scope：** LFW；目前是固定的 5 個 development identities（每種
  attack `n=5`，PGD + MI-FGSM 合計 5+5 個 attack observations），不是 general
  transfer-rate estimate。identity 與影像對定義見
  `data/datasets/lfw/evaluation_splits/attack_dev.csv`。
- **正式五人：** Michelle Collins、Meryl Streep、Jolanta Kwasniewski、Dino de
  Laurentis、Cyndi Thompson。
- **A/B pair：** 同一 identity 的 `*_0001.jpg` 為 reference **A**，`*_0002.jpg`
  為 probe **B**；攻擊 B 得 **B_adv**，評估 clean `A–B` 與 adversarial
  `A–B_adv` cosine。這是 verification dodging，不是 identity-to-identity
  impersonation。
- **模型角色：** ArcFace / InsightFace `buffalo_l` 是 white-box surrogate；
  FaceNet VGGFace2 是 black-box victim（`victim_gradient_participation=false`）。
- **門檻：** acceptance rule 為 cosine `>= threshold` 即 same identity；兩模型皆
  採 LFW development-train 校正的 `far_1e-2` operating point：ArcFace
  **0.140721**、FaceNet **0.404789**。不要使用 legacy 0.4 或 exploratory
  FAR=1e-3 門檻取代它們。
- **擾動預算：** `configured_epsilon=0.04`（= 10.2/255）是 tensor attack budget；
  `actual_linf` 是序列化 PNG 後 decoded RGB uint8 的最大差。正式樣本的
  tensor L∞ 為約 0.040000，實際序列化 L∞ 為 **11/255 = 0.043137**；後者因
  8-bit quantization 可以略大於 configured epsilon，兩者不可混用。
- **正式 perceptual metrics：** SSIM（native RGB）、LPIPS（AlexNet，jointly
  resize 到 max 256）、DISTS（VGG16，jointly resize 到 max 256）。

## Authority Table

| 正式項目 | 唯一權威來源 | 用途 / 邊界 |
| --- | --- | --- |
| ArcFace calibration | `results/calibration/arcface_lfw_v1/{thresholds.json,summary.json}` | ArcFace threshold 與 calibration protocol。 |
| FaceNet calibration | `results/calibration/facenet_lfw_v1/{thresholds.json,summary.json}` | FaceNet threshold 與 calibration protocol。 |
| PGD source metrics | `results/robustness/lfw_verification_pgd_v1/{results.csv,summary.json,config.json}` | PGD 的 ArcFace source / white-box metrics；此 run 未持久化 B_adv。 |
| PGD regenerated transfer artifacts | `results/transfer/pgd_arcface_to_facenet_verification_v1/` | 同設定後續重生的 PGD B_adv 與 FaceNet transfer measurements；不是 source run 原始 artifact。 |
| MI-FGSM formal results | `results/transfer/mi_fgsm_arcface_to_facenet_verification_v1/` | MI-FGSM source、victim metrics 與其持久化 B_adv。 |
| Perceptual Evaluation v2 | `results/perceptual/lfw_development_transfer_v2/{perceptual_metrics.csv,aggregate_summary.json,report_v2.html}` | 正式 SSIM / LPIPS / DISTS、serialized L∞，以及跨來源的最終 5+5 比較。 |
| 20-identity development baseline v2 | `results/transfer/lfw_attack_dev_20_baseline_v2/{pgd_full,mi_fgsm}/` | frozen `attack_dev.csv` 全部 20 identities；每個 observation 的 ArcFace、FaceNet、actual L∞、SSIM、LPIPS、DISTS 都由同一 decoded serialized `B_adv` 計算。 |
| 5-identity PGD vs landmark-mask ablation v2 | `results/transfer/lfw_attack_dev_mask_ablation_v2/{pgd_full,pgd_landmark_superpixel_mask}/` | 固定原 v1 五人、200 steps、`tv_weight=0` 的完整 PGD 與 PGD+Mask paired comparison；mask ordering 是 overlay 驗證的 implementation assumption。 |
| 20-identity formal four-method v3 | `results/transfer/lfw_attack_dev_20_four_method_v3/{summary.json,unified_summary.csv,per_identity_results.csv}` | frozen `attack_dev.csv` 的 PGD、MI-FGSM、PGD+Mask、MI-FGSM+Mask 各 20 identities；`formal: true`、20/20 valid、serialized B_adv / ε-plus-quantization / mask-outside checks passed。後續 mechanism analysis 只能讀取此 artifact，不得重生 B_adv。 |
| Gradient trajectory instrumentation | `results/diagnostics/lfw_gradient_trajectory_v1/` | **diagnostic / mechanism analysis，不是新的 benchmark authority。** 可選 `diagnostics=True` 僅保存 detached statistics、trajectory 與 mask artifact；正式 benchmark 預設為關閉且行為不變。 |

**PGD provenance 必讀：** PGD 的 ArcFace source metrics 來自 robustness run，
但 perceptual 評估的 `B_adv` 是 transfer run 以相同 configuration 後續重生的
artifact。不得敘述為「perceptual 的 B_adv 原本由 robustness run 持久化」。

## 目前正式結果（LFW fixed 5+5）

所有數字是各 attack 的五個 identity 平均；ASR 是該固定 development sample
上的 boundary-crossing proportion，非可泛化成功率。兩法均為 ε=0.04、200
steps，且 ArcFace source ASR 均為 5/5。

| Attack | actual L∞ (serialized) | ArcFace cosine drop / ASR | FaceNet cosine drop / transfer ASR | SSIM ↑ | LPIPS ↓ | DISTS ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PGD Full | 11/255 = 0.043137 | 1.013154 / 5/5 (100%) | 0.165169 / 0/5 (0%) | 0.966591 | 0.044879 | 0.049447 |
| MI-FGSM | 11/255 = 0.043137 | 0.763412 / 5/5 (100%) | 0.216153 / 1/5 (20%) | 0.932117 | 0.088657 | 0.069428 |

解讀限於這五人：PGD 的 source drop 較大且 perceptual scores 較佳；MI-FGSM
在 FaceNet 的平均 verification drop 較大，並只在 Cyndi Thompson 觀察到 1 次
victim threshold crossing。這些不是方法優劣的 general claim。

## 20-identity development baseline v2（完整；不取代 v1）

`attack_dev.csv` 的全部 20 identities（未使用 `attack_test.csv` 或
`attack_reserve.csv`）均以 ε=0.04、200 steps 跑完。每個 identity 在獨立
subprocess 中完成攻擊、PNG serialization/reload、ArcFace/FaceNet evaluation 與
SSIM/LPIPS/DISTS，完成後 process 結束；兩個 run 都是 20/20 valid、0 failures、
0 observed OOM。數字為 20 identity 平均：

| Attack | ArcFace cosine drop / ASR | FaceNet cosine drop / transfer ASR | actual L∞ | SSIM ↑ | LPIPS ↓ | DISTS ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PGD Full | 0.919779 / 20/20 (100%) | 0.212559 / 4/20 (20%) | 11/255 | 0.967956 | 0.039712 | 0.048895 |
| MI-FGSM | 0.693174 / 18/20 (90%) | 0.271114 / 8/20 (40%) | 11/255 | 0.934606 | 0.081449 | 0.066979 |

這個較大但仍為 frozen development sample 的比較，維持 v1 的方向：PGD 有較大的
ArcFace source drop 且感知分數較佳；MI-FGSM 有較大的 FaceNet transfer drop 與
較多 victim boundary crossings。不可延伸為 general transfer-rate claim。

## Landmark-superpixel mask ablation v2（完整 5 identity）

先以 Michelle Collins 做 20-step engineering smoke；PGD 與 PGD+Mask 都成功，
mask PNG 的外部 delta 為 0。正式固定五人均使用 200 steps、ε=0.04、
`tv_weight=0`，完整 5/5、0 failures、0 observed OOM。PGD vs PGD+Mask 的
ArcFace drop 為 1.013154 vs 0.751933、FaceNet drop 為 0.165169 vs 0.161429、
FaceNet ASR 皆為 0/5；Mask 的 SSIM/LPIPS/DISTS 為 0.984227/0.015499/0.020583
（unmasked: 0.966591/0.044879/0.049447）。五筆 serialized PNG 的 mask 外
delta 均為 0。

`buffalo_l/1k3d68.onnx` 的輸出 artifact 沒有官方 semantic point-index mapping。
因此 17–26 eyebrows、27–35 nose、36–47 eyes、48–67 mouth 是
**implementation assumption**，不是僅由 68 個 output points 推論的事實；已用
`visualizations/<identity>/landmark_indices_overlay.png` 人工檢視，該 smoke
identity 的各 region 視覺上落在相應五官。所有正式 Mask output 也保存 original、
index overlay、initial polygon、SLIC、final mask、perturbation、B_adv。

## 已完成研究（不要重複）

- LFW 上 ArcFace 與 FaceNet 的 model-specific threshold calibration。
- ε / PGD Full（ε=0.04、200 steps）在 5 identity LFW development protocol 的
  source robustness / verification measurement。
- ArcFace → FaceNet 的 LFW PGD transfer（含重生 artifact 的 provenance 註記）。
- 同 protocol 的 MI-FGSM formal comparison。
- Perceptual Evaluation v2：SSIM、LPIPS、DISTS、serialized actual L∞ 的 5+5
  聚合與 provenance 檢查。

## Historical / 不得作為正式分析的資料

- `results/archive/smoke/`：runner / pipeline 的 engineering smoke outputs。
- `results/archive/historical_single_image/pgdfull_musk_v2/`：非 LFW、沒有 A/B
  verification pair 的 single-image legacy baseline。
- `results/archive/historical_meeting/meeting_2026_09_02/`：historical meeting
  snapshot；其中的舊數字與解讀只能追溯背景。
- `archive/docs/{FGSM_README.txt,PGD_README.txt}` 與
  `archive/legacy_results_notes/`：legacy single-image / old-pipeline 說明；可讀但
  不能放入正式 LFW 或 perceptual 統計。

## 目前研究分支：從哪裡接續

- **Targeted impersonation：** 建立獨立 target-identity A/B/target protocol 與
  success definition；不可沿用目前 untargeted ASR 的語義。
- **Invisible attack：** 已新增可重用的 `landmark_superpixel_mask` constraint
  module（InsightFace `buffalo_l/1k3d68.onnx` 的 68-point landmarks + SLIC）。
  它只遮罩每步 attack gradient，可由 `pgd_full`、`mi_fgsm` 與後續 attack 透過
  `AttackConfig.extra={"constraint": "landmark_superpixel_mask"}` 使用；
  `constraint=None` 與 `tv_weight=0` 保持 baseline 行為。其 5-pair smoke output
  必須在完整產生且通過 pairing / epsilon / perceptual sanity check 後才可加入
  Authority Table，未完成的 smoke 或 partial output 不得引用。
- **Transferability：** 從既有 ArcFace → FaceNet 5-identity result 擴展更多
  victim、identity / split 與 attack design；維持 calibrated threshold。
- **Robustness / real-world distortion：** 對既有 B_adv 加入壓縮、resize、
  crop、光照或其他實際失真，分開報告 source 與 victim 的 verification effects。
- **Experimental / combination branches：** DI / TI、ensemble、query-based、
  mapper 等候選組合應先獨立實驗；成功 merge 回主線後才可成為正式 protocol。

## 給新組員的使用規則

1. 不確定哪個結果能用時，以本文件的 Authority Table 為準。
2. 新實驗一律沿用正式 LFW protocol，除非研究問題明確要求更改；更改時要另存
   protocol、threshold 與 artifact provenance。
3. branch 實驗成功並 merge 回主線後，必須同步更新本文件的 Authority Table、
   正式結果與已完成研究。
4. smoke / exploratory / historical result 不得直接升格成正式結論；須有正式
   protocol、可追溯輸出與明確 authority 後才可納入。

## v2 expansion feasibility record (2026-09-30)

`data/datasets/lfw/evaluation_splits/attack_dev.csv` 的 frozen development
candidate split 只有 **20 identities**（含原 v1 的 5 identities）。本次已依明確
授權將全部 20 identities 建為 development baseline v2；沒有以
`attack_test.csv` 或 `attack_reserve.csv` 補樣本。n=20 仍不得描述成 30–50 或
general transfer-rate estimate。
