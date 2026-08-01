# Fawkes 照片保護環境

這個資料夾已經準備好 Fawkes 的獨立 Python venv。

## 使用方式

1. 把要處理的照片放到 `photos/`。
2. 執行 high 模式：

```bash
./run-high.sh
```
/Users/taishen/active/face_modify/adversarial-noise/fawkes/.venv/bin/python run_protection.py --directory ./photos/ --mode --help
或手動執行：

```bash
source .venv/bin/activate
python run_protection.py --directory ./photos/ --mode high
```

完成後同一個資料夾會出現 `*_cloaked.png`。

## 資料夾

- `photos/`: 放原始照片。
- `photos/*_cloaked.*`: Fawkes 產出的保護後照片會出現在同一個資料夾。
- `run_protection.py`: 啟動 wrapper，會在載入 fawkes 之前修補 TF 2.11+ 優化器相容性。
- `run-high.sh`: 一鍵跑 high 模式的腳本。

## 環境資訊

- Python: 3.10
- Package: `fawkes==1.0.3`
- 依賴版本鎖定（見 `requirements.txt`）：
  - `tensorflow==2.15.1`、`keras==2.15.0`：Fawkes 1.0.3 依賴 Keras 2 API；TF 2.16+ 預設 Keras 3 會直接壞。
  - `numpy==1.26.4`：NumPy 2.x 改了 ABI，TF 2.15 / Fawkes 都不相容。
  - `mtcnn==0.1.1`：`mtcnn` 1.x 把 `MTCNN(min_face_size=...)` 介面整個改掉，必須維持舊版。
  - `setuptools<81`：新版 setuptools 拿掉了 `pkg_resources`，會讓 `fawkes/utils.py` 啟動就壞掉。

## 已知相容性修補

- TF 2.11 起 Keras 換成新版優化器，對 Fawkes 手動建立的 `tf.Variable` 直接 `apply_gradients` 會丟 `KeyError: 'The optimizer cannot recognize variable Variable:0'`。`run_protection.py` 在匯入 fawkes 之前把 `tf.keras.optimizers.Adadelta` 別名換成 legacy 版本來解決這個問題（M1/M2 Mac 也會比較快）。

## 重新建立環境

如果 `.venv` 被刪掉，可以在本資料夾重新建立：

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```



--mode MODE, -m MODE  cloak generation mode, select from min, low, mid, high. 