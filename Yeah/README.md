# Adversarial Face Research Platform

本專案目前的正式研究主線是 LFW face-verification adversarial attack：以
ArcFace / InsightFace `buffalo_l` 作為 white-box surrogate，FaceNet VGGFace2
作為不參與梯度的 black-box victim，比較 PGD Full 與 MI-FGSM 的 verification
dodging、transfer 與 perceptual quality。

## 新組員先讀這兩份文件

1. [研究狀態與可引用結果](docs/RESEARCH_STATUS.md)
2. [正式實驗 protocol](docs/EXPERIMENT_PROTOCOL.md)
3. [Reproducibility 與 research asset setup](docs/REPRODUCIBILITY.md)

`docs/RESEARCH_STATUS.md` 是正式研究結論與結果 provenance 的最高權威。
若文件、HTML report 或舊輸出與它不一致，請以它為準。

## 正式結果位置

- calibration：`results/calibration/{arcface_lfw_v1,facenet_lfw_v1}/`
- PGD source robustness：`results/robustness/lfw_verification_pgd_v1/`
- transfer：`results/transfer/{pgd_arcface_to_facenet_verification_v1,mi_fgsm_arcface_to_facenet_verification_v1}/`
- Perceptual Evaluation v2：`results/perceptual/lfw_development_transfer_v2/`

正式分析僅限 LFW、calibration、PGD、MI-FGSM 與 Perceptual Evaluation v2。
`archive/` 與 `results/archive/` 的內容只供歷史追溯，**不得作為正式分析依據**。

## 開發與驗證

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -e .
pytest
```

Perceptual Evaluation v2 只從已持久化的正式 artifact 重建評估，不會重跑 attack：

```bash
.venv/bin/python scripts/evaluate_lfw_perceptual_v2.py
```

探索性單圖、sun / musk、FGSM、smoke、舊 benchmark 與 historical meeting 材料
已退出 active research navigation；其可追溯材料見 [archive/README.md](archive/README.md)。
