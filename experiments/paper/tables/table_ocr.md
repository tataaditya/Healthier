# OCR field-level evaluation (verified 131-image test set)

Indonesian packaged-food label photographs across five product categories (`Frozen Food` 20, `Makanan Kaleng` 16, `Makanan Ringan` 53, `Minuman Sachet` 18, `Saos dan Perbumbuan` 24) with human-verified ground truth (`ground_truth_verified.json`). Field accuracy = exact value or within 5% of the per-serving value across 544 legible fields (38 unreadable printed fields excluded from all denominators). CI is a 2,000-iteration cluster bootstrap over images (`seed=42`).

| Pipeline | Images | Fields | Field accuracy | 95% CI | Exact | Wrong | Missing |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: |
| A. Tesseract | 131 | 544 | 14.89% | [10.28%, 20.15%] | 81 | 154 | 309 |
| B. EasyOCR | 131 | 544 | 19.67% | [15.03%, 24.53%] | 107 | 210 | 226 |
| C. PaddleOCR, raw + line parser | 131 | 544 | 35.11% | [29.29%, 40.89%] | 191 | 159 | 193 |
| D. Preprocessing + layout parser, no grid | 131 | 544 | 23.53% | [17.63%, 29.91%] | 128 | 53 | 363 |
| E. Full pipeline (grid reconstruction) | 131 | 544 | **42.65%** | [36.23%, 49.00%] | 232 | 98 | 213 |
