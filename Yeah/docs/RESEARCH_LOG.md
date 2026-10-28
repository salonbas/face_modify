# Research Log

- 2026-10-02 — 完成教授進度報告整理；目前剩餘核心問題為 Mask 對 MI-FGSM 的 white-box ASR 損失機制。
- 2026-10-02 — 建立 Mask × MI-FGSM gradient/trajectory diagnostic instrumentation；完整 frozen-20 diagnostic 待在可完成 200-step run 的環境執行。
- 2026-10-07 — 整理 Git reproducibility：納入 frozen split、calibration metadata 與正式研究產物，僅保留環境／快取／大型 raw assets 在 Git 外。
- 2026-10-07 — 將可重現實驗所需的 LFW / model assets 改為官方來源自動安裝至固定 repo-local 路徑；因上游 redistribution terms，不納入 Git LFS。
- 2026-10-28 — 完成 MI/PGD 更新式核對與 LFW split audit：μ=0 與 PGD+Mask 數學等價；現有 split 無可新增的未使用 development 身份，且 11 人與 calibration pair 定義重疊；本 checkout 缺 raw images／model weights，故未執行新 ablation。
