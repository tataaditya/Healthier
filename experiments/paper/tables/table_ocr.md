# OCR field-level evaluation (pilot)

Indonesian snack-label photos with hand-transcribed ground truth. Field accuracy = exact value or within 5% of the per-serving value. CI is a cluster bootstrap over images.

| Pipeline | Images | Fields | Field accuracy | 95% CI | Exact | Wrong | Missing |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: |
| Baseline parser | 6 | 46 | 21.74% | [16.67%, 25.58%] | 21.74% | 13.04% | 65.22% |
| Grid parser (revised) | 6 | 46 | 47.83% | [29.17%, 66.67%] | 47.83% | 17.39% | 34.78% |

The revised parser adds a row-label x value-column grid and splits tall numeric boxes that PP-OCR merges across adjacent rows (e.g. `1725` = 1, 7, 2, 5), which separates the Per Sajian and %AKG columns. Standalone unit tokens (`mg`, `g`) are kept through the semantic filter.

This is a **pilot (n=6)** and must be reported as exploratory, not as a generalisation result: the CI is wide and the sample is not stratified by brand or store. Rerun on >=100 images before citing a headline accuracy.
