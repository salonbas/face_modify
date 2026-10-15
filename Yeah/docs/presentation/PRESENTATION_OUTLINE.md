# 20 分鐘簡報架構（14 頁）

> 題目：基於深度學習之人臉辨識對抗攻擊研究與嵌入式系統應用。總時間約 19:30，保留約 30 秒緩衝。研究成果與尚未實作的嵌入式規劃必須以不同色標示。

## Part 1｜畢業專題研究（約 12 分鐘）

### 1. 封面（0:30）
- **核心訊息：** 研究「人臉驗證遭受對抗擾動」及其對未來智慧門禁的意義。
- **內容：** 題目、課程、組員、指導老師、日期（姓名欄待補充）。
- **視覺：** 簡潔的人臉 embedding → similarity → accept/reject 線性圖；不要使用裝飾性 AI 人像。
- **來源：** 自製示意。

### 2. 為什麼要研究人臉辨識安全？（1:00）
- **核心訊息：** 門禁依賴相似度門檻；肉眼不明顯的擾動可能使合法使用者被拒絕。
- **內容：** 應用背景、adversarial example、已驗證風險界線：「拒絕本人」，不是冒充他人。
- **視覺：** A/B 正常通過與 A/B_adv 跌破門檻的二欄示意。
- **來源：** `docs/EXPERIMENT_PROTOCOL.md`。

### 3. 研究問題與整體架構（1:15）
- **核心訊息：** 在 ArcFace 產生攻擊，再測 FaceNet，觀察白箱效果、跨模型轉移與畫質。
- **內容：** LFW A(reference)／B(probe) → ArcFace surrogate → PGD/MI-FGSM ± Mask → B_adv → ArcFace/FaceNet + 品質指標。
- **視覺：** 一張由左到右流程圖；標出 victim 不參與梯度。
- **來源：** `docs/RESEARCH_STATUS.md`、`results/transfer/lfw_attack_dev_20_four_method_v3/summary.json`。

### 4. Face Verification 與兩個模型（1:15）
- **核心訊息：** 本研究是 1:1 verification，不是 1:N identification。
- **內容：** normalized 512-D embedding、cosine 判定；ArcFace threshold 0.140721，FaceNet 0.404789（FAR 1% 校正）。
- **視覺：** Verification vs Identification 小圖，右側列 source/victim 角色。
- **來源：** calibration `thresholds.json`／`summary.json`。

### 5. PGD 與 MI-FGSM：如何讓相似度下降（1:20）
- **核心訊息：** 兩者均迭代降低 ArcFace self-match cosine；MI-FGSM額外累積動量。
- **內容：** ε=0.04、200 steps；PGD=逐步更新與投影，MI-FGSM=L1 gradient normalization + momentum。避免完整數學推導。
- **視覺：** 兩條簡化更新流程；顯示 ε 邊界。
- **來源：** `docs/EXPERIMENT_PROTOCOL.md`、`advface/attacks/{pgd.py,mi_fgsm.py}`。

### 6. Landmark-guided Mask（1:20）
- **核心訊息：** 只在眉眼鼻口相關 superpixels 更新 gradient，換取更佳視覺品質。
- **內容：** 68 landmarks → 五官 polygon → SLIC → final mask → gradient × mask；索引語意為 implementation assumption。
- **視覺：** `original.png` → `landmark_indices_overlay.png` → `slic_segmentation.png` → `final_mask.png` → `B_adv.png`。
- **來源：** `docs/reports/meeting_2026_09_30/assets/`、constraint 程式。

### 7. 實驗設計與指標（1:20）
- **核心訊息：** frozen development 20 人、每法 n=20；ASR 是門檻 crossing，不是一般化成功率。
- **內容：** 四方法、共 80 attack observations；A=`0001`、B=`0002`；actual L∞=11/255；cosine drop/ASR、SSIM/LPIPS/DISTS。
- **視覺：** protocol 卡片 + 指標方向箭頭（SSIM↑；LPIPS/DISTS↓）。
- **來源：** split config、`attack_dev.csv`、formal v3 summary。

### 8. 結果與三個發現（4:00）
- **核心訊息：** 白箱成功不代表跨模型成功；MI-FGSM較會轉移；Mask改善品質但犧牲效果。
- **內容：** 放完整四方法表；突出 PGD ArcFace ASR 100%、MI-FGSM FaceNet ASR 40%、PGD+Mask SSIM 0.985195／LPIPS 0.013953／DISTS 0.017759。附註 development n=20/method。
- **視覺：** 左側 grouped bars（ArcFace ASR：100/90/80/35%；FaceNet：20/40/20/30%）；右側品質表或 `arcface_facenet_crossing.png` + `paired_*_mask.png`。
- **來源：** formal v3 `{summary.json,unified_summary.csv}`；既有分析 PNG。

## Part 2｜嵌入式系統應用規劃（約 5:45；尚未部署）

### 9. 應用情境：智慧人臉門禁（0:50）
- **核心訊息：** 把 verification pipeline 放到 edge，可降低影像外傳與網路依賴，但攻擊可能造成拒絕服務式風險。
- **內容：** 合法使用者註冊模板、現場 probe、通過門檻才開鎖；清楚標「概念設計／未部署」。
- **視覺：** 使用者—攝影機—edge—門鎖情境圖。
- **來源：** 自製規劃；無實測數據。

### 10. 可實作的門禁系統架構（1:40）
- **核心訊息：** 完整資料流可由現成嵌入式模組組成。
- **內容：** CSI/USB camera → frame capture → face detection/alignment → resize/normalize → embedding model → cosine 與 template → threshold → policy → GPIO → transistor/optocoupler relay/driver → electric strike；另含 event log、按鍵／管理介面、門磁與 fail-safe power design。
- **視覺：** 分層架構圖：感測、運算、決策、致動；GPIO 不直接驅動門鎖，電源隔離。
- **來源：** 本次合理工程規劃；平台型號、鎖具電壓與法規待補充。

### 11. 研究結果對門禁的安全意義（1:10）
- **核心訊息：** 已驗證的是數位影像上本人相似度下降；門禁影響仍是假設，需經 camera-to-edge 測試。
- **內容：** Evidence：ArcFace source 最多 100%、FaceNet transfer 最多 40% crossing（development n=20）；Potential：合法者拒絕、反覆失敗、可用性下降；Not shown：冒充、實體重拍、門鎖破解。
- **視覺：** 三欄「已驗證／合理風險／尚未證明」。
- **來源：** formal v3 summary。

### 12. Edge 限制與設計取捨（2:05）
- **核心訊息：** latency、RAM、功耗與模型壓縮會同時影響可用性與攻擊行為，必須量測。
- **內容：** Raspberry Pi 類 SBC 的 CPU/NPU選擇；偵測通常是主要負載之一；RAM含 runtime、模型、frame buffers；持續相機推論影響功耗／散熱；ONNX/TFLite、FP16/INT8、較小 backbone 可加速但可能改變 embedding、門檻與 transferability。列 KPI：端到端 latency、FPS、峰值 RAM、平均/峰值功耗、溫度、FAR/FRR、attack ASR。所有數值標待量測。
- **視覺：** trade-off 四角圖 + benchmark 表空格。
- **來源：** 工程規劃；不得填入未測數字。

## Part 3｜未來整合與結論（約 1:45）

### 13. 未來整合路線（1:00）
- **核心訊息：** 先建立 clean edge baseline，再逐步測試影像轉換與攻擊轉移。
- **內容：** (1) 部署與校正 threshold；(2) 比較模型 latency/RAM/power；(3) JPEG、resize、camera recapture、FP16/INT8 robustness；(4) ArcFace→edge model transfer；(5) 防禦與 liveness／重試策略。全標「未來工作」。
- **視覺：** 五階段 roadmap，不填日期。
- **來源：** `docs/RESEARCH_STATUS.md` 的未來分支 + 本次整合規劃。

### 14. 結論（0:45）
- **核心訊息：** 三句帶走：研究界線、正式發現、下一步。
- **內容：** verification dodging；source/transfer gap 與 mask trade-off；嵌入式部署需重新量測效能、門檻與攻擊韌性。
- **視覺：** 三張結論卡；頁尾「Q&A」。
- **來源：** formal v3 summary。

## 待組員補充清單

- 封面：組員、學號、課程、老師、日期與校系識別。
- 確認 LFW 人像於課堂簡報中的使用／標註方式；若不放臉圖，改用流程示意。
- 從現有素材挑選可公開的單例對照；目前 baseline v2 原始 B_adv 目錄在 checkout 缺失。
- 決定 Raspberry Pi 具體型號、camera、relay/driver、鎖具與電源方案；簡報現階段不得填效能數字。
- 未來實測表：端到端 latency、FPS、RAM、功耗、溫度、FAR/FRR 與 edge ASR。
- 口頭報告分工、動畫順序、引用格式與圖片授權註記。
