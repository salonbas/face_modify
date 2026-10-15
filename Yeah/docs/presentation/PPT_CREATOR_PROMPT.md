# 可直接貼入 Canva AI／PPT Creator 的 Prompt

請建立一份 **16:9、14 頁、繁體中文（臺灣用語）** 的大學資訊工程系課堂簡報，題目為「基於深度學習之人臉辨識對抗攻擊研究與嵌入式系統應用」，報告時間 20 分鐘。

## 全域規則

- 視覺風格：專業、簡潔、深藍／青綠科技色系，白底或深色底皆可，但全篇一致；大標題清楚，內文每頁以 3–5 點為限。
- 優先使用流程圖、比較表、bar chart、攻擊前後對照；不要使用無關的裝飾性 AI 人像，不要自行上網找研究結果。
- **不得改寫、四捨五入成不同結論、補造或推估下列鎖定數據。** 若版面不足，保留 6 位小數或 ASR 分數形式。
- 明確區分兩類內容：藍色標籤「Repository 已驗證」；橘色標籤「嵌入式規劃／尚未部署」。
- 本研究是 **1:1 face verification 的 untargeted dodging**：讓同一人的相似度下降，可能使合法使用者驗證失敗。不是 1:N identification；沒有證明 targeted impersonation、破解門禁、實體攻擊或嵌入式裝置攻擊。
- 不需要完整數學推導。英文術語第一次出現時附短中文解釋。
- 不要在投影片顯示 Repository 長路徑；路徑僅供製作者／組員找素材，可放 speaker notes 或頁尾小字「Source」。
- 未確定欄位必須原樣保留「待補充」；不得用常識數字代填。

## 鎖定研究事實與數據

- Dataset：LFW funneled；固定 `attack_dev.csv` development 20 identities。每種方法 n=20，四方法共 80 attack observations；未使用 test/reserve，不能稱一般化成功率。
- Pair：同一 identity 的 `*_0001.jpg` 是 reference A，`*_0002.jpg` 是 probe B；只攻擊 B，得到 B_adv。
- Surrogate：ArcFace／InsightFace `buffalo_l`，white-box；Victim：FaceNet VGGFace2，black-box 且不參與梯度。
- Acceptance：cosine similarity ≥ threshold 判定同一人；ArcFace threshold=0.140721、FaceNet threshold=0.404789，均為 LFW development-train FAR 1% operating point。
- 四法皆 ε=0.04、200 steps、`tv_weight=0`；序列化 PNG actual L∞=11/255=0.043137。MI-FGSM 使用 alpha=0.0002、momentum=1.0。
- 正式結果：

| 方法 | ArcFace drop | ArcFace ASR | FaceNet drop | FaceNet transfer ASR | SSIM | LPIPS | DISTS |
|---|---:|---:|---:|---:|---:|---:|---:|
| PGD | 0.919779 | 20/20（100%） | 0.212559 | 4/20（20%） | 0.967956 | 0.039712 | 0.048895 |
| MI-FGSM | 0.693174 | 18/20（90%） | 0.271114 | 8/20（40%） | 0.934606 | 0.081449 | 0.066979 |
| PGD + Mask | 0.647885 | 16/20（80%） | 0.209295 | 4/20（20%） | 0.985195 | 0.013953 | 0.017759 |
| MI-FGSM + Mask | 0.498320 | 7/20（35%） | 0.221978 | 6/20（30%） | 0.974845 | 0.025242 | 0.024718 |

- SSIM 越高越好；LPIPS、DISTS 越低越好。ASR 是此固定 development sample 中由 same-person acceptance 跨越門檻成 reject 的比例。

## 逐頁內容與版面

### 第 1 頁｜封面（30 秒）
標題：「基於深度學習之人臉辨識對抗攻擊研究與嵌入式系統應用」

副標：「從 ArcFace／FaceNet 對抗實驗到智慧門禁整合規劃」

下方欄位：「課程：待補充｜組員：待補充｜指導老師：待補充｜日期：待補充」

版面：中央標題；背景使用抽象的 embedding 節點、cosine gauge、門禁輪廓，勿生成逼真人臉。

### 第 2 頁｜問題：人臉驗證也會被微小擾動影響（60 秒）
核心句：「門禁不是只要『看見臉』，而是 embedding 相似度必須高於門檻。」

重點：
- 正常 A–B：同一人，相似度高於門檻 → Accept
- 攻擊 A–B_adv：相似度下降、跌破門檻 → Reject
- 本研究關注合法使用者被拒絕的 dodging 風險
- 未證明冒充他人、破解門禁或實體攻擊

版面：左右二欄 decision flow，以綠色 Accept、紅色 Reject 表示；不要填虛構 cosine 值。

### 第 3 頁｜研究目標與完整實驗流程（75 秒）
核心句：「在 ArcFace 產生擾動，再用 FaceNet 檢驗跨模型轉移。」

流程圖文字必須依序為：
`LFW 同一人 A / B pair` → `ArcFace white-box surrogate` → `PGD 或 MI-FGSM` → `可選 Landmark-guided Mask` → `B_adv` → 分支一 `ArcFace source evaluation`、分支二 `FaceNet black-box victim evaluation`、分支三 `SSIM / LPIPS / DISTS`。

在 FaceNet 旁標：「不參與 gradient」。頁角放藍色「Repository 已驗證」。

### 第 4 頁｜本研究是 Verification，不是 Identification（75 秒）
左側做二格比較：
- Verification（本研究）：1:1，問「A 與 B 是否同一人？」
- Identification（非本研究）：1:N，問「這個人是資料庫中的誰？」

右側模型卡：
- ArcFace：white-box surrogate，512-D normalized embedding，threshold 0.140721
- FaceNet VGGFace2：black-box victim，512-D normalized embedding，threshold 0.404789
- Rule：cosine ≥ threshold → same identity

頁尾小字：「Thresholds calibrated on LFW development-train at FAR 1%。」

### 第 5 頁｜PGD 與 MI-FGSM（80 秒）
核心句：「兩者都迭代降低 ArcFace self-match cosine，但更新策略不同。」

二欄內容：
- PGD：gradient sign 更新 → 投影回 L∞ ε-ball → 重複 200 steps
- MI-FGSM：L1-normalized gradient → momentum 累積 → sign 更新 → 投影

共同設定：ε=0.04、200 steps、full-image attack；MI-FGSM：alpha=0.0002、momentum=1.0。

版面：兩條簡短迴圈流程，不放完整公式；用邊框標示 ε constraint。

### 第 6 頁｜Landmark-guided Mask：把更新集中在五官區域（80 秒）
核心句：「Mask 限制每一步 gradient 的空間範圍，目標是提升不可察覺性。」

流程：`原圖` → `68 landmarks` → `眉／眼／鼻／口 polygon` → `SLIC superpixels` → `final mask` → `gradient × mask` → `B_adv`。

旁註：
- Mask 固定由 clean B 建立，不是把彩色遮罩貼在人臉上
- 68 點 region mapping 是經 overlay 檢查的 implementation assumption

手動插入素材（PPT Creator 不可讀取時保留灰色佔位框）：
`docs/reports/meeting_2026_09_30/assets/original.png`
`landmark_indices_overlay.png`、`slic_segmentation.png`、`final_mask.png`、`B_adv.png`

### 第 7 頁｜實驗設計與評估指標（80 秒）
用四張 protocol cards：
1. LFW frozen development：20 identities
2. A=`*_0001`；B=`*_0002`；只攻擊 B
3. 四方法 × 20 = 80 observations
4. ε=0.04；serialized actual L∞=11/255

下方指標列：
- Attack effect：cosine drop、threshold-crossing ASR
- Visual quality：SSIM ↑、LPIPS ↓、DISTS ↓

警示小字：「ASR 僅代表固定 development sample，不是一般化成功率。」

### 第 8 頁｜正式結果：效果、轉移性與畫質的取捨（240 秒）
此頁是研究重點，使用 60% 圖表、40% 結論。不得修改鎖定數據。

圖表 A：grouped bar chart，Y 軸 ASR（0–100%），四組依序 PGD、MI-FGSM、PGD+Mask、MI-FGSM+Mask；ArcFace series = 100、90、80、35；FaceNet series = 20、40、20、30。

圖表 B：精簡品質表：
- PGD：SSIM .967956 / LPIPS .039712 / DISTS .048895
- MI-FGSM：.934606 / .081449 / .066979
- PGD+Mask：.985195 / .013953 / .017759（以綠色標最佳）
- MI-FGSM+Mask：.974845 / .025242 / .024718

三個結論卡：
1. PGD：ArcFace 20/20，但 FaceNet 4/20 → 白箱成功不等於跨模型成功
2. MI-FGSM：FaceNet drop 0.271114、8/20 crossing，為四法最高
3. Mask：品質改善，但 ArcFace ASR 下降，呈現 imperceptibility–effectiveness trade-off

頁尾：「每方法 n=20；LFW frozen development；非 general claim。」

可手動插入：`docs/reports/meeting_2026_09_30/assets/analysis/arcface_facenet_crossing.png`、`paired_pgd_mask.png`、`paired_mi_fgsm_mask.png`。若未插入，依上述資料重建 chart。

### 第 9 頁｜應用情境：智慧人臉門禁（50 秒）
頁角放橘色「嵌入式規劃／尚未部署」。

內容：
- 使用者註冊 reference template
- 現場 camera 擷取 probe
- Edge device 本機產生 embedding 並比對
- 通過 threshold 才送出開鎖訊號
- 優點：減少影像外傳、降低網路依賴；限制：資源與安全需重新評估

版面：使用者 → camera → edge → lock 的情境圖；不要寫已完成 Raspberry Pi 部署。

### 第 10 頁｜可實作的嵌入式門禁架構（100 秒）
畫四層資料流架構：

1. 感測層：CSI／USB camera、補光（選配）、門磁／離開按鍵
2. 運算層：Raspberry Pi 類 SBC → capture → face detection → alignment → resize/normalize → embedding model
3. 決策層：stored template + cosine threshold → retry／deny／accept policy → encrypted event log
4. 致動層：GPIO → transistor/optocoupler relay 或專用 driver → electric strike／maglock；門鎖使用獨立合規電源

醒目工程註記：「GPIO 不可直接驅動門鎖；須隔離、driver、flyback protection 與 fail-safe/fail-secure 設計。」

待補充：SBC 型號、camera 型號、鎖具／電壓、enclosure、網路與消防／逃生需求。

### 第 11 頁｜對抗攻擊對門禁的意義：證據邊界（70 秒）
做三欄：

「已驗證」
- 數位 LFW 影像上的 same-person verification dodging
- ArcFace source ASR 最高 100%
- FaceNet transfer ASR 最高 40%
- n=20/method development sample

「合理安全風險」
- 合法使用者可能反覆驗證失敗
- 可用性下降、造成拒絕服務式干擾
- 模型差異會改變 transferability

「尚未證明」
- 冒充特定他人
- 相機重拍／實體貼片仍有效
- 已破解門禁或已在 edge 裝置成功

### 第 12 頁｜Edge 限制：速度、記憶體、功耗與模型壓縮（125 秒）
核心句：「同一模型搬到 edge 不只是變慢；前處理與量化也可能改變門檻和攻擊效果。」

用 trade-off 圖呈現：
- Compute／latency：face detection + alignment + embedding 的端到端時間
- Memory：runtime、模型、frame buffers、OS 共用 RAM
- Power／thermal：持續 camera inference、降頻、散熱
- Compression：小 backbone、ONNX/TFLite、FP16/INT8 可望加速，但可能改變 embedding distribution、accuracy、threshold、attack transfer

右下 benchmark 表全部保留待補充：
`平台｜模型｜端到端 latency｜FPS｜峰值 RAM｜平均/峰值功耗｜溫度｜FAR/FRR｜attack ASR`
第一列：`Raspberry Pi 型號：待補充｜其餘：待量測`。

禁止填入任何推估效能數字。

### 第 13 頁｜未來整合路線（60 秒）
畫五階段 roadmap，全部標「未來工作」：
1. Edge clean baseline：部署 detection + embedding，重新校正 threshold
2. Platform benchmark：比較模型 latency、RAM、power、thermal
3. Pipeline robustness：resize、JPEG、camera recapture、lighting
4. Compression study：FP16／INT8／小模型是否改變攻擊效果
5. Transfer & defense：ArcFace → edge model transfer、liveness、重試與 anomaly logging

不要加入完成日期或已完成符號。

### 第 14 頁｜結論（45 秒）
三張大卡：
1. 「研究界線」：完成的是 untargeted face-verification dodging，不是 impersonation
2. 「核心發現」：PGD 白箱效果強；MI-FGSM 較會轉移；Mask 改善畫質但犧牲 ASR
3. 「整合下一步」：在 edge 重新量測效能、門檻、壓縮與實體 pipeline 的 attack robustness

頁尾：「Q&A」及「組員／聯絡資訊：待補充」。

## 素材與資料來源（供製作者，不需做成額外頁）

- 正式數據：`results/transfer/lfw_attack_dev_20_four_method_v3/summary.json`、`unified_summary.csv`、`per_identity_results.csv`
- 研究定義：`docs/RESEARCH_STATUS.md`、`docs/EXPERIMENT_PROTOCOL.md`
- 門檻：`results/calibration/arcface_lfw_v1/thresholds.json`、`results/calibration/facenet_lfw_v1/thresholds.json`
- Mask 圖：`docs/reports/meeting_2026_09_30/assets/`
- 結果圖：`docs/reports/meeting_2026_09_30/assets/analysis/`

生成後請檢查：恰好 14 頁；全繁體中文；第 8 頁數值逐項一致；第 9–13 頁均清楚標示尚未部署／未來工作；任何「待補充」均不得擅自填寫。
