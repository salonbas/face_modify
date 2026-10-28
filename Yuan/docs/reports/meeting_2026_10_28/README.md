# 研究說明網頁：A-reference、DI、TI

開啟本資料夾的 `report.html`，向下捲動閱讀。這是一般文章式網頁，內容聚焦三個主題：

1. 對準驗證使用的 A：A-reference。
2. DI — Diverse Inputs：輸入多樣化。
3. TI — Translation-Invariant：平移不變梯度。

頁面包含任務與loss的必要背景、兩個實際圖片例子、DI縮放/padding互動示意、
TI Gaussian kernel與更新順序，以及四階段的結果比較。
圖片可點選放大；左側目錄可直接跳到主題；可列印整份文章。
不使用投影片換頁、計時、全螢幕或講稿切換介面。

圖片和數據內嵌，約1.3 MB，無需網路、伺服器或額外套件。
四個條件的 FaceNet 跨界比例為40%、65%、70%、80%（全20人分母），
另列clean-valid 19人分母，避免把原始驗證失敗計入攻擊成功。
建置時核對80張保存PNG，沒有重新生成攻擊。

`meeting_data.json` 是本頁數據快照。`qa/webpage-verification.json` 記錄瀏覽器檢查：
三個主題標題、無簡報控制項、離線圖片、DI示意、圖片放大、
1440/1366/768/390px版面及列印，未發現錯誤或外部網路請求。

## 修改與重建

內容版型：`Yuan/scripts/professor_web_template.html`。
數據建置程式：`Yuan/scripts/build_professor_meeting.py`，預設為page版面。

```powershell
C:/senior_project/face_modify/Yuan/.venv/Scripts/python.exe C:/senior_project/face_modify/Yuan/scripts/build_professor_meeting.py --yuan-root C:/senior_project/face_modify/Yuan
```

`transcript.md`、`transcript.html` 和原投影片版型是先前14頁版本的保留材料，
不會嵌入目前三主題的網頁，也未列為頁面的主要閱讀內容。
來源的實驗報告與程式均維持原樣。
