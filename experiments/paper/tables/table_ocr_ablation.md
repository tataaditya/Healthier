# OCR cumulative and leave-one-out ablation (verified 131-image test set)

Cumulative preprocessing and post-processing chain on the verified test set ($n = 131$ images, 544 legible fields).

| Configuration | Field accuracy | 95% CI | Delta vs previous | Exact | Wrong | Missing |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| Raw OCR (no preprocessing) | 29.96% | [23.70%, 36.60%] | - | 163 | 52 | 328 |
| + deskew | 28.68% | [22.99%, 34.67%] | -1.29 | 156 | 58 | 330 |
| + deglare | 28.68% | [22.99%, 34.67%] | 0.00 | 156 | 58 | 330 |
| + CLAHE | 23.35% | [17.95%, 29.13%] | -5.33 | 127 | 52 | 365 |
| + ROI | 23.53% | [17.63%, 29.91%] | +0.18 | 128 | 53 | 363 |
| + semantic filter | 22.24% | [16.85%, 27.93%] | -1.29 | 121 | 60 | 363 |
| + fuzzy matching | 27.21% | [21.42%, 33.27%] | +4.96 | 148 | 78 | 318 |
| + energy/fat fallback | 30.33% | [24.33%, 36.55%] | +3.12 | 165 | 92 | 287 |
| + clinical validation | 29.60% | [23.69%, 35.73%] | -0.74 | 161 | 82 | 301 |
| Full pipeline (grid enabled) | 42.65% | [36.23%, 49.00%] | +13.05 | 232 | 98 | 213 |

Leave-one-out (LOO) post-processing ablation on the full pipeline:

| Configuration | Field accuracy | 95% CI | Delta vs full | McNemar p | Exact | Wrong | Missing |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| Full minus semantic filter | **46.14%** | [38.79%, 53.35%] | +3.49% | 0.000878 | 251 | 93 | 199 |
| Full minus fuzzy matching | 32.35% | [26.13%, 38.62%] | -10.29% | 1.2e-10 | 176 | 98 | 268 |
| Full minus energy/fat fallback | 42.10% | [35.58%, 48.61%] | -0.55% | 0.453 | 229 | 90 | 224 |
| Full minus clinical validation | 43.01% | [36.66%, 49.44%] | +0.37% | 0.500 | 234 | 108 | 201 |
