# DPA 程式起點（非完整論文復現）

這個小程式用合成分類資料測試：把已訓練模型的參數與隨機初始化參數混合時，模型預測一致性與準確率如何改變。

**它不是 DPA 的 DPO/HMA 實作，也沒有產生人臉攻擊影像。** 它的用途是先驗證 Python/PyTorch 環境，以及熟悉「參數多樣化」這個概念。

## 執行方法（Windows PowerShell）

```powershell
cd .\yu\dpa_starter
python -m pip install torch
python .\dpa_parameter_study.py
```

會在 `results/dpa_parameter_diagnostic.csv` 產生表格。可以重新設定 `--seed`、`--epochs`、`--out`。

## 下一階段

1. 找到你現有 ArcFace 和 FaceNet 的推論入口，確認對齊、正規化和 embedding 尺度。
2. 使用有授權的測試資料建置固定的驗證配對與每個模型各自的閾值。
3. 研究論文完整的 DPO 訓練條件與 HMA 中間層操作，不能將這個混合參數示範當作完整 DPA。
4. 對比固定模型與多樣化模型的乾淨影像表現、運算成本，以及不同資料分布的穩健性。
