================================================================================
FGSM（Fast Gradient Sign Method）— 單步 L∞ 攻擊
================================================================================

原理
----
FGSM 是最基本的對抗攻擊。它的核心概念是：
  1. 把「相似度 loss」對「輸入圖像像素」求一次梯度。
  2. 取梯度的正負號（sign），乘以步長 eps×255。
  3. 沿著讓 cosine 最小化的方向，移動一步。

公式（像素域）：
  x_adv = x - (eps × 255) × sign( d(cosine) / dx )

攻擊目標
--------
  損失函數：InsightFace buffalo_l 的 ArcFace 模型（w600k_r50.onnx）自匹配 cosine。
  目標：讓「對抗圖的 embedding」與「原圖的 embedding」的 cosine 盡量低（< 0.4）。

優點
----
  - 速度快（只算一次梯度）
  - 概念簡單，容易理解

缺點
----
  - 單步更新不夠精準，相同 eps 下效果不如 PGD
  - eps 稍大（約 > 0.07）就可能讓人臉偵測失敗（臉已扭曲）

實驗結果（sun.png）
-------------------
  eps=0.005 → cosine=0.808（幾乎沒影響）
  eps=0.01  → cosine=0.669
  eps=0.02  → cosine=0.381  ← 突破 0.4 門檻（肉眼幾乎看不出差異）
  eps=0.03  → cosine=0.265
  eps=0.05  → cosine=0.144
  eps=0.07  → cosine=0.108
  eps=0.1+  → 人臉偵測失敗（擾動過大）

執行指令
--------
  python scripts/attack_fgsm.py --mode fgsm --run-name sun_v2 --img data/raw/sun.png
  python scripts/attack_fgsm.py --mode fgsm --run-name musk_v2 --img data/raw/musk1.jpg

子資料夾說明
------------
  <run-name>/
    fgsm_cosine_metrics.csv     每個 eps 的 cosine 數值
    fgsm_cosine_line_chart.png  cosine vs eps 折線圖
    fgsm_summary.png            原圖 + 各 eps 對抗圖總覽
    fgsm_<stem>_eps_*.png       各 eps 的對抗圖（單獨一張）
    base_<stem>.png             原圖備份
