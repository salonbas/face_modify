# DiffProtect Colab

這個資料夾提供一份 Google Colab notebook，用來在 GPU runtime 上執行 `joellliu/DiffProtect`。

## 使用方式

1. 到 Google Colab 開啟 `DiffProtect_Colab.ipynb`。
2. 選 `Runtime` -> `Change runtime type` -> GPU。
3. 從上到下執行 notebook。
4. 在上傳照片的 cell 上傳自拍照。
5. 執行完成後，notebook 會下載 `diffprotect_results.zip`。

## 注意事項

- DiffProtect 沒有官方 Colab，這份 notebook 會自動 clone 官方 repo 並做 Colab 相容性修補。
- 權重檔很大，Google Drive 可能限流；下載失敗時可用 notebook 內的手動上傳權重 cell。
- 建議上傳正臉、清楚、單人照片，臉部對齊成功率較高。
