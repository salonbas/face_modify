# 畢業專題研究摘要

> 盤點日期：2026-10-15。正式性以 `docs/RESEARCH_STATUS.md` 的 Authority Table 為最高依據；以下未重跑訓練、推論或攻擊。

## 最新研究進度與任務界定

目前正式主線是 **LFW 人臉驗證（face verification）的 untargeted dodging attack**，不是 identification，也不是 targeted impersonation。每筆觀察取同一人的兩張 LFW funneled 影像：`*_0001.jpg` 為 reference A、`*_0002.jpg` 為 probe B；只攻擊 B 產生 B_adv，比較 A–B 與 A–B_adv 的 cosine similarity。目的在使原本應判為同一人的相似度跌破門檻，對應到門禁情境是「合法使用者可能被拒絕」，不能描述為冒充他人或破解門禁。來源：`docs/RESEARCH_STATUS.md`、`docs/EXPERIMENT_PROTOCOL.md`。

資料切分由 `data/datasets/lfw/evaluation_splits/split_config.json` 固定（seed 20260902）：development 20 人、test 100 人、reserve 208 人；正式四方法比較只用 `attack_dev.csv` 的 20 人，未用 test/reserve。因此結果是 frozen development sample 的比較，不能推廣為一般成功率。來源：上述 split config、`attack_dev.csv`、`results/transfer/lfw_attack_dev_20_four_method_v3/summary.json`。

## 方法與評估

- **模型角色：** ArcFace／InsightFace `buffalo_l`（`w600k_r50.onnx`）是可取得梯度的 white-box surrogate；FaceNet VGGFace2 是不參與梯度的 black-box victim。兩者皆輸出正規化 512 維 embedding。來源：兩模型 calibration `summary.json`、`docs/EXPERIMENT_PROTOCOL.md`。
- **判定：** cosine similarity `>= threshold` 為同一人。LFW development-train、FAR 1% 校正門檻為 ArcFace **0.140721**、FaceNet **0.404789**。來源：`results/calibration/{arcface_lfw_v1,facenet_lfw_v1}/thresholds.json`。
- **攻擊：** PGD 與 MI-FGSM，皆為 full-image、ε=0.04、200 steps、`tv_weight=0`；MI-FGSM 另用 alpha=0.0002、momentum=1.0。目標是降低 ArcFace 對 clean B self-reference 的 cosine。來源：`docs/EXPERIMENT_PROTOCOL.md`、`advface/attacks/{pgd.py,mi_fgsm.py}`。
- **Landmark-guided Mask：** 使用 68 點 landmarks，在眉、眼、鼻、口建立初始區域，再以 SLIC superpixel 擴張；每一步只將 gradient 乘上固定 mask，並非把遮罩畫到輸入影像。68 點語意索引是經 overlay 人工檢視的 implementation assumption，不是模型官方保證。來源：`advface/constraints/landmark_superpixel.py`、`docs/RESEARCH_STATUS.md`。
- **指標：** source／victim cosine drop 與 calibrated-threshold crossing（ASR）；影像品質用 SSIM（越高越好）、LPIPS、DISTS（越低越好）。configured ε=0.04 是 tensor budget；PNG 解碼後 actual L∞ 為 11/255=0.043137，因 8-bit quantization 可稍大，兩者不可混稱。

## 正式實驗數據（20 identities，每方法 n=20）

| 方法 | ArcFace cosine drop / ASR | FaceNet drop / transfer ASR | SSIM ↑ | LPIPS ↓ | DISTS ↓ |
|---|---:|---:|---:|---:|---:|
| PGD | 0.919779 / 20/20（100%） | 0.212559 / 4/20（20%） | 0.967956 | 0.039712 | 0.048895 |
| MI-FGSM | 0.693174 / 18/20（90%） | 0.271114 / 8/20（40%） | 0.934606 | 0.081449 | 0.066979 |
| PGD + Mask | 0.647885 / 16/20（80%） | 0.209295 / 4/20（20%） | **0.985195** | **0.013953** | **0.017759** |
| MI-FGSM + Mask | 0.498320 / 7/20（35%） | 0.221978 / 6/20（30%） | 0.974845 | 0.025242 | 0.024718 |

四法皆為 actual L∞ 11/255；clean cosine 平均 ArcFace 0.665798、FaceNet 0.718468。來源（正式彙總）：`results/transfer/lfw_attack_dev_20_four_method_v3/{summary.json,unified_summary.csv,per_identity_results.csv}`。

## 主要研究發現

1. **白箱效果強，但跨模型效果下降：** PGD 在 ArcFace 20/20 crossing，轉至 FaceNet 僅 4/20；顯示 source 成功不等於跨模型成功。
2. **方法存在效果／品質取捨：** 未遮罩 MI-FGSM 的 FaceNet drop 與 crossing 較高（0.271114、8/20），但 PGD 的 ArcFace drop 較大且 SSIM／LPIPS／DISTS 較佳。僅限本 development sample。
3. **Mask 顯著改善感知品質，但會犧牲攻擊效果：** PGD+Mask 的三項品質指標最佳，然而 ArcFace ASR 由 100% 降至 80%；MI-FGSM+Mask 更由 90% 降至 35%。這支持「限制擾動範圍帶來品質與有效性的 trade-off」，不能宣稱全面優於 baseline。

## 可用視覺素材

- 四方法正式數據圖：`docs/reports/meeting_2026_09_30/assets/analysis/{arcface_facenet_crossing.png,arcface_vs_facenet_drop.png,paired_pgd_mask.png,paired_mi_fgsm_mask.png,perturbation_distribution.png}`。
- Mask 流程圖像：`docs/reports/meeting_2026_09_30/assets/{original.png,landmark_indices_overlay.png,initial_polygon_mask.png,slic_segmentation.png,final_mask.png,perturbation.png,B_adv.png}`。
- 單例前後對照：`docs/reports/meeting_2026_09_30/assets/comparison/`。其中檔名含 `archived_mask`，適合作示意；不要用它單獨支持正式總體數字。
- 正式表格可由 `unified_summary.csv` 重建；個別樣本可由 `per_identity_results.csv` 重建。

## 尚未完成、無法確認或資料衝突

- 尚未在 Raspberry Pi 或其他嵌入式平台部署；推論延遲、RAM、功耗、溫度與 GPIO 門鎖測試皆為「待補充／未量測」。
- 尚未驗證 resize、JPEG compression、量化、相機重拍等實體／edge pipeline 對攻擊的影響；也沒有冒充他人的 targeted 結果。
- `summary.json` 指向 `results/transfer/lfw_attack_dev_20_baseline_v2/{pgd_full,mi_fgsm}`，但目前 checkout 未見這兩個目錄；四方法正式彙總與逐筆 CSV 存在，baseline 的原始 B_adv 圖則不可直接取用。另 MI-FGSM+Mask 子目錄未見獨立 `config.json`，其設定以頂層 formal summary 與逐筆 records 交叉確認。
- Repository 文件同時保留舊 5+5 結果；它是正式歷史 baseline，但本簡報應以較完整的 20 人四方法 v3 為主，勿混合母數。
