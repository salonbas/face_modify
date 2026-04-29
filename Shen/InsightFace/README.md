# InsightFace Face Recognition Colab

這個資料夾提供一份 Google Colab notebook，用來用 `deepinsight/insightface` 比對兩張照片中的臉是否可能是同一人。

## 使用方式

1. 到 Google Colab 開啟 `InsightFace_Face_Recognition_Colab.ipynb`。
2. 選 `Runtime` -> `Change runtime type` -> GPU。
3. 從上到下執行 notebook。
4. 在上傳圖片的 cell 選擇兩張照片。
5. Notebook 會顯示兩張照片的臉框、cosine similarity，以及是否可能為同一人的判斷。

## 判斷方式

- 使用 `FaceAnalysis(name="buffalo_l")` 偵測臉並抽取 face embedding。
- 每張照片如果有多張臉，會自動選擇面積最大的臉。
- 使用兩個 normalized embedding 的 cosine similarity 做比對。
- 預設 threshold 是 `0.35`，這只是快速 demo 的參考值，不適合直接當正式產品標準。

## 注意事項

- 第一次執行會自動下載 InsightFace 模型，可能需要一點時間。
- 建議使用清楚、正臉、單人照片，辨識結果會比較穩定。
- InsightFace 0.7.3 的 Cython 模組綁 NumPy 1.x C-API，而 Colab 預裝的 OpenCV/JAX/shap 等套件強制要求 NumPy 2.x。Notebook 第一個 cell 會把 NumPy、OpenCV、ONNX Runtime、InsightFace 一起鎖到相容版本，並**自動重啟 runtime**。
- 重啟後 Colab 會顯示「session crashed」之類的訊息，這是正常現象；請從第 2 段（匯入套件）的 cell 開始往下執行，不要再跑安裝 cell。
- 如果看到 `np._core._multiarray_umath has no attribute _blas_supports_fpe` 錯誤，代表沒有重啟 runtime，請從第一個 cell 重新跑一次。
- InsightFace 程式碼是 MIT License，但官方 README 說明部分訓練資料與模型僅供非商業研究用途；正式商用前請確認模型授權。
