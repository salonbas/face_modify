================================================================================
PGD（Projected Gradient Descent）— 多步迭代 L∞ 攻擊
================================================================================

原理
----
PGD 是 FGSM 的多步版本，被認為是目前最強的一階對抗攻擊。
  1. 從原圖出發，重複以下動作 N 步：
     a. 計算當前 cosine 對像素的梯度。
     b. 沿著讓 cosine 最小化的方向，走一小步（步長 = eps×255 / steps）。
     c. 將擾動「投影（Project）」回 L∞ ball：確保每個像素的變動不超過 eps×255。
     d. 把像素值 clamp 到 [0, 255]。
  2. 最終的 x_adv 就是 steps 步後的結果。

FGSM vs PGD 差異
-----------------
  FGSM：1 步、步長大（eps×255）→ 方向較粗糙
  PGD ：N 步、步長小（eps×255/N）→ 每步都重新算梯度，方向更精準
  
  結果：相同 eps 下，PGD 的 cosine 降幅比 FGSM 大很多（通常 5–10 倍）

攻擊目標
--------
  同 FGSM：InsightFace buffalo_l w600k_r50.onnx ArcFace 自匹配 cosine。

實驗結果（sun.png，steps=20）
------------------------------
  eps=0.005 → cosine=0.727
  eps=0.01  → cosine=0.460
  eps=0.02  → cosine=0.035  ← 遠低於 0.4 門檻，且肉眼幾乎看不出差異
  eps=0.03  → cosine=-0.149  ← 負值！embedding 方向已完全相反
  eps=0.05  → cosine=-0.315
  eps=0.07  → cosine=-0.460
  eps=0.1   → cosine=-0.463

  對比 FGSM：eps=0.02 時 FGSM=0.381，PGD=0.035，效果差了 10 倍以上。

執行指令
--------
  python scripts/attack_fgsm.py --mode pgd --steps 20 --run-name sun_v2 --img data/raw/sun.png

子資料夾說明
------------
  <run-name>/
    pgd_cosine_metrics.csv      每個 eps 的 cosine 數值（含 steps 欄）
    pgd_cosine_line_chart.png   cosine vs eps 折線圖
    pgd_<stem>_eps_*.png        各 eps 的對抗圖（單獨一張）
    base_<stem>.png             原圖備份
