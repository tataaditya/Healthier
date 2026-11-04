# OCR field-level evaluation (pilot)

Indonesian snack-label photos with hand-transcribed ground truth. Field accuracy = exact value or within 5% of the per-serving value. CI is a cluster bootstrap over images.

| Pipeline | Images | Fields | Field accuracy | 95% CI | Exact | Wrong | Missing |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: |
| Baseline parser | 6 | 47 | 14.89% | [4.65%, 22.92%] | 14.89% | 17.02% | 68.09% |
| Grid parser (revised) | 6 | 47 | 44.68% | [24.49%, 66.67%] | 44.68% | 19.15% | 36.17% |

The revised parser adds a row-label x value-column grid and splits tall numeric boxes that PP-OCR merges across adjacent rows (e.g. `1725` = 1, 7, 2, 5), which separates the Per Sajian and %AKG columns. Standalone unit tokens (`mg`, `g`) are kept through the semantic filter.

This is a **pilot (n=6)** and must be reported as exploratory, not as a generalisation result: the CI is wide and the sample is not stratified by brand or store. Rerun on >=100 images before citing a headline accuracy.
