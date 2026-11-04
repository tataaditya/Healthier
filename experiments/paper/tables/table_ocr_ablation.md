# OCR per-stage preprocessing ablation (pilot)

Cumulative preprocessing chain with the baseline (non-grid) parser, six images and 46 fields; each row adds one stage to the row above. The grid parser is measured separately in the field-level table.

| Variant | Field accuracy | 95% CI | delta vs raw | Exact | Wrong | Missing |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| raw | 32.61% | [27.27%, 37.50%] | +0.00% | 15 | 17 | 14 |
| +deskew | 39.13% | [27.27%, 53.85%] | +6.52% | 18 | 14 | 14 |
| +deglare | 39.13% | [27.27%, 53.85%] | +6.52% | 18 | 14 | 14 |
| +clahe | 41.30% | [25.00%, 59.52%] | +8.69% | 19 | 12 | 15 |
| +roi | 41.30% | [25.00%, 59.52%] | +8.69% | 19 | 10 | 17 |
| +semantic | 41.30% | [25.00%, 59.52%] | +8.69% | 19 | 10 | 17 |
| +fuzzy | 50.00% | [30.43%, 68.89%] | +17.39% | 23 | 11 | 12 |
| +fallback | 50.00% | [30.43%, 68.89%] | +17.39% | 23 | 11 | 12 |
| +validation | 47.83% | [29.17%, 66.67%] | +15.22% | 22 | 7 | 17 |
| full | 47.83% | [29.17%, 66.67%] | +15.22% | 22 | 7 | 17 |
