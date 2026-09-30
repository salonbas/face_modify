# Formal LFW Experiment Protocol

本文件定義目前可引用的 LFW 研究 protocol。研究狀態、結果與 artifact authority
以 [RESEARCH_STATUS.md](RESEARCH_STATUS.md) 為準；本文件不取代該 authority table。

## Scope and unit of analysis

- 任務為 **untargeted verification dodging**，不是 targeted impersonation。
- 使用 LFW funneled images。每個 observation 是同一 identity 的 A/B pair：
  `*_0001.jpg` 是 reference A，`*_0002.jpg` 是 probe B；只攻擊 B，產生
  `B_adv`，比較 clean `A-B` 與 adversarial `A-B_adv` cosine。
- 正式 v1 allowlist 是 Michelle Collins、Meryl Streep、Jolanta Kwasniewski、
  Dino de Laurentis、Cyndi Thompson，定義於
  `data/datasets/lfw/evaluation_splits/attack_dev.csv`。該 CSV 是較大的
  development candidate split；**只有此五人**屬於目前正式 v1 的固定樣本。
- 每個 attack 有 `n=5`，PGD 與 MI-FGSM 合併為 10 個 attack observations。
  這是 development comparison，不可宣稱為 general transfer-rate estimate。
- v2 expansion 使用同一 frozen `attack_dev.csv` 的全部 20 identities，不更改
  split 且不使用 test/reserve。它是較大的 development baseline，仍非 general
  transfer-rate estimate；v1 的固定五人不被覆寫。

## Models and decision rule

- ArcFace / InsightFace `buffalo_l` 是可微分的 white-box surrogate。
- FaceNet VGGFace2 是 black-box victim；`victim_gradient_participation=false`。
- cosine `>= threshold` 判為 same identity。門檻一律使用 official LFW
  development-train calibration 的 `far_1e-2` operating point：ArcFace
  `0.14072093367576602`、FaceNet `0.40478909015655523`。
- 不得以 legacy `0.4` 或 exploratory `far_1e-3` 門檻取代上述門檻。

## Attack contract

| Attack | Surrogate objective | Parameters |
| --- | --- | --- |
| PGD Full | existing ArcFace self-representation objective | `eps=0.04`、200 steps |
| MI-FGSM | ArcFace self-reference cosine minimization | `eps=0.04`、200 steps、`alpha=0.0002`、momentum `1.0`、per-image L1 gradient normalization、no random start |

`configured_epsilon=0.04` 是 tensor-space attack budget（10.2/255）。
另行報告 decoded RGB uint8 PNG 的 `actual_linf`；因 8-bit quantization，正式
artifact 為 11/255（0.043137），不得與 configured epsilon 混為同一數值。

## Metrics and reporting

- source / victim metrics：clean 與 adversarial A/B cosine、cosine drop，以及各
  模型 calibrated threshold 的 boundary crossing。ASR 僅指固定五人中的 crossing
  proportion。
- perceptual metrics：native RGB SSIM；LPIPS AlexNet（jointly resize 到 max
  256）；DISTS VGG16（jointly resize 到 max 256）；以及 serialized `actual_linf`。
- 所有正式結論都必須保留 attack、identity allowlist、threshold source、artifact
  provenance 與 sample count。

## Result provenance

- calibration：`results/calibration/arcface_lfw_v1/` 與
  `results/calibration/facenet_lfw_v1/`。
- PGD 的 ArcFace source metrics：`results/robustness/lfw_verification_pgd_v1/`。
  這個 source run 未持久化 `B_adv`。
- PGD transfer / perceptual image artifact：
  `results/transfer/pgd_arcface_to_facenet_verification_v1/` 是相同設定後續重生的
  `B_adv`，不可誤稱為 source robustness run 原始持久化 artifact。
- MI-FGSM metrics 與 `B_adv`：
  `results/transfer/mi_fgsm_arcface_to_facenet_verification_v1/`。
- 最終 5+5 perceptual aggregation：
  `results/perceptual/lfw_development_transfer_v2/`。
- v2 20-identity complete baseline：
  `results/transfer/lfw_attack_dev_20_baseline_v2/{pgd_full,mi_fgsm}/`。每個
  identity 由獨立 subprocess 執行；所有報告 metrics 都從同一 serialized PNG
  decode 後取得。
- v2 five-identity Mask ablation：
  `results/transfer/lfw_attack_dev_mask_ablation_v2/{pgd_full,pgd_landmark_superpixel_mask}/`；
  200 steps、ε=0.04、`tv_weight=0`。landmark index layout 是經 overlay 檢驗的
  implementation assumption，不是 1k3d68 artifact 已證實的官方 semantic mapping。

任何不在上述範圍的 single-image、smoke、historical 或 exploratory output，均不
可併入正式統計或用來支持正式結論。
