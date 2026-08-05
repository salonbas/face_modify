================================================================================
Compare — 跨攻擊方法比較圖
================================================================================

存放由 scripts/compare_attacks.py 產生的跨方法比較折線圖。
每張圖可同時顯示多種攻擊方法的 cosine 曲線，並標出 cosine=0.4 門檻線。

執行範例
--------
  # FGSM vs PGD-20（sun.png）
  python scripts/compare_attacks.py \
    --csv "results/fgsm/sun_v1/fgsm_cosine_metrics.csv:FGSM" \
    --csv "results/pgd/sun_v1/pgd_cosine_metrics.csv:PGD-20" \
    --out results/compare/fgsm_vs_pgd_sun.png \
    --title "sun.png: FGSM vs PGD-20"

  # 加入第三條曲線
  python scripts/compare_attacks.py \
    --csv "results/fgsm/sun_v1/fgsm_cosine_metrics.csv:FGSM" \
    --csv "results/pgd/sun_v1/pgd_cosine_metrics.csv:PGD-20" \
    --csv "results/pgd/sun_pgd40/pgd_cosine_metrics.csv:PGD-40" \
    --out results/compare/fgsm_vs_pgd_comparison.png
