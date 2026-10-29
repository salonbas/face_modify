# 研究專案重構與 Clean Clone 任務回報

更新：2026-10-29。詳細命令與證據見 [CLEAN_CLONE_AUDIT.md](CLEAN_CLONE_AUDIT.md)。

## 已完成

- 原有 35 個 script 保留歷史相容，新增 `scripts/run_face_verification_benchmark.py` 作為功能導向入口。`configs/lfw_dev_20_v3.json` 與 `configs/lfw_dev_100_v2.json` 明確選擇 cohort、門檻來源及 runner；v2 100 人執行仍由 preflight gate 拒絕。不同歷史 protocol 的 runner 沒有草率合併或刪除。
- Python 3.12 安裝改為相容的 NumPy 1.26.4、PyTorch 2.2.2+cpu、FaceNet 2.6.0、Pillow 10.2.0；不再使用 `--no-deps`，全新 venv 的 `pip check` 與 imports 通過。
- LFW、ArcFace／landmark、FaceNet、MobileFaceNet 的自動下載及 SHA-256 驗證通過。LFW 的 Figshare 來源回傳 403 時，改用整包 SHA-256 與 scikit-learn 紀錄一致的公開鏡像；13,233 張 JPEG 逐檔驗證。補齊 InsightFace runtime 路徑與 MobileFaceNet detector。
- Git 已追蹤 v2 Development 100／OOS 228 split、五模型 calibration JSON/CSV、checkpoint manifest、Momentum ablation 小型記錄、正式 config、三份規格文件、安裝／檢查腳本及正式 PNG checksum manifest。五個歷史來源 CSV 的 Git 位元組現在與 v2 audit SHA-256 一致。第三方圖片、權重、cache 沒有提交。
- `METRICS.md`、`CALIBRATION_AND_THRESHOLDS.md`、`EXPERIMENT_CONFIG.md` 保留權威來源與驗證方式；`generate_research_specs.py --check` 通過。正式 v3 結果及教授報告未修改。

## GitHub Clean Clone 實測

從已推送 `origin/main` 的 `8c852f43202e12b8891e40104bac5616914ebc7c` 重新 clone，使用全新 Python 3.12 venv 與空白 HOME/cache。安裝、`pip check`、LFW 與自動模型 checksum、ArcFace／FaceNet／MobileFaceNet loader、七個 CLI `--help`、config／split／threshold 一致性及正式 v3 80 筆結果讀取通過。clone 測試為 **65 passed、1 skipped**；原工作目錄的新依賴 venv 為 **66 passed**。五模型 benchmark preflight 正確回報 `NOT READY`、exit 2；沒有執行 100 人 benchmark 或 OOS 攻擊。

**Clean clone 判定：NOT YET REPRODUCIBLE。** AdaFace 與 MagFace 作者權重需依 [REPRODUCIBILITY.md](REPRODUCIBILITY.md) 由 Google Drive 人工取得並以 manifest SHA-256 驗證；這個 clone 未取得兩者，因此五模型 200-step 新生成、SSIM／LPIPS／DISTS／L∞、threshold crossing 與數值容差驗收尚未通過。原環境的一人 smoke 只讀取已存在的 PNG，不能當成 clean-clone 證據。正式 PNG 的 hash 已記錄，但原圖 bytes 仍需私下備份；感知模型 cache 的精確 provenance 也待補。

**A. Repository Reproducibility：PARTIAL　B. Research Integrity：PASS　C. GitHub Sync：PUSHED**
