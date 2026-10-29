# Research Log

- 2026-10-02 — 完成教授進度報告整理；目前剩餘核心問題為 Mask 對 MI-FGSM 的 white-box ASR 損失機制。
- 2026-10-02 — 建立 Mask × MI-FGSM gradient/trajectory diagnostic instrumentation；完整 frozen-20 diagnostic 待在可完成 200-step run 的環境執行。
- 2026-10-07 — 整理 Git reproducibility：納入 frozen split、calibration metadata 與正式研究產物，僅保留環境／快取／大型 raw assets 在 Git 外。
- 2026-10-07 — 將可重現實驗所需的 LFW / model assets 改為官方來源自動安裝至固定 repo-local 路徑；因上游 redistribution terms，不納入 Git LFS。
- 2026-10-28 — 完成 MI/PGD 更新式核對與 LFW split audit：μ=0 與 PGD+Mask 數學等價；現有 split 無可新增的未使用 development 身份，且 11 人與 calibration pair 定義重疊；本 checkout 缺 raw images／model weights，故未執行新 ablation。
- 2026-10-29 — 完成兩位 MI+Mask momentum 對照（μ=0/0.5 恢復 ArcFace 成功）、凍結 LFW v2 development 100/OOS 228 並建立身份隔離的 ArcFace/FaceNet calibration v2；舊 split、threshold 與正式 v3 結果未改。
- 2026-10-29 — 五模型 benchmark Stage 0 preflight：既有四方法攻擊時間外推 100 人需約 182 分鐘（僅 attack.apply），且三個新 checkpoint 尚未取得；依 60 分鐘上限觸發 Hard Stop，未執行新攻擊或 OOS 評估。
## 2026-10-29 — five-model readiness safety gate

Profiled one Development identity on CPU and rejected a zero-TV graph change
after paired timing comparison. Downloaded and hashed official/author-linked
AdaFace, MobileFaceNet, and MagFace checkpoints; implemented their inference
adapters and independent Calibration v2 thresholds. Development 100 clean
eligibility retained its full denominator (83 jointly eligible). Two-person
PGD+Mask end-to-end smoke and 66 tests passed. The fail-closed CLI refuses
formal execute because attack-only runtime exceeds 60 minutes and the safe
full executor is unfinished. Status: NOT READY; no 100-person attack started.
