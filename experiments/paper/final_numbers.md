# Final numbers (single source of truth)

> Synced with `experiments/results/ocr_eval/REPORT.md`, `experiments/results/ocr_eval/e2e_propagation_verified.md`, `experiments/results/edge/cpu/bench.md`, `experiments/results/segmentation/dataset_audit.md`, and `experiments/paper/manuscript_jri.md`.
> Every number in the manuscript must trace back to this file or to `experiments/results/`.

## Run configuration

| Item | Value |
| :--- | :--- |
| Model | YOLO11s-seg |
| Training images | 11,684 |
| Validation images | 2,017 |
| External test images | 5,013 (UEC 948; FoodSeg 2,096; FruitSeg 1,969) |
| Candidate corpus | 19,087 images; 373 duplicates removed; 18,714 unique |
| Class remapping | 104 FoodSeg103 classes + 102 UEC IDs → 5 *Isi Piringku* groups (24.0% of FoodSeg classes mapped to catch-all `Others`) |
| Train class imbalance | Instance max/min **3.04×** (`Vegetables` 10,689 vs `Fruits` 3,513); image max/min **4.12×** (`Staple Food` 5,708 vs `Fruits` 1,384) |
| Max epochs | 100 |
| Early stopping | patience 20; stopped at epoch 67 |
| Best checkpoint | epoch 48 |
| Optimizer | AdamW (lr0 0.001, lrf 0.01, weight decay 0.0005) |
| Batch / image size | 16 / 640 |
| Seed | 42 (deterministic) |
| Hardware | Kaggle Tesla T4, 15.6 GB VRAM |
| Training time | 4.01 h (14,420.7 s) |
| Ultralytics / Torch | 8.4.199 / 2.10.0+cu128 |
| Python | 3.12.13 |

## Headline numbers

**Internal validation, best checkpoint**

- Mask mAP50: **71.66%**
- Mask mAP50-95: **56.91%**
- Mask precision / recall: **72.95% / 65.50%**
- Box mAP50 / mAP50-95: **72.37% / 60.11%**

**External tests (mask, %) — domain shift is part of the result**

- FoodSeg103 validation: mAP50 **63.66**, mAP50-95 **48.16**, P **67.47**, R **58.37**
- UECFoodPIXCOMPLETE test: mAP50 **69.15**, mAP50-95 **54.30**, P **67.42**, R **67.44**
- FruitSeg30: mAP50 **17.06**, mAP50-95 **13.55**, P **17.01**, R **44.25**

**Per-class mask AP50 (external)**

- FoodSeg103 validation: best **Fruits 74.96**, worst **Others 51.24**
- UECFoodPIXCOMPLETE test: best **Staple Food 85.70**, worst **Vegetables 55.42**
- FruitSeg30: best **Fruits 17.29**, worst **Fruits 17.29** (pixel P 97.17%, R 2.14%, F1 4.19%)

**OCR field-level accuracy (verified test split, n=131 images, 544 legible fields, 5 categories)**

- Candidate pool: 453 images → 15 duplicates removed (MD5 + dHash) → 438 unique → locked test **131** / dev **307** (`seed=42`, SHA-256 `a0c7a4942526120b...`)
- Ground truth: **131/131 human-verified**; 582 printed fields, **544** legible denominator fields, **38** unreadable fields excluded; 60 images have no nutrition table (`absent`); 177 human cell corrections across 36 images during verification.
- A. Tesseract: **14.89%** (95% CI [10.28%, 20.15%], Exact 81, Wrong 154, Missing 309)
- B. EasyOCR: **19.67%** (95% CI [15.03%, 24.53%], Exact 107, Close 1, Wrong 210, Missing 226)
- C. PaddleOCR, raw + line parser: **35.11%** (95% CI [29.29%, 40.89%], Exact 191, Close 1, Wrong 159, Missing 193)
- D. Preprocessing + layout parser, no grid: **23.53%** (95% CI [17.63%, 29.91%], Exact 128, Wrong 53, Missing 363)
- E. Full pipeline (grid reconstruction): **42.65%** (95% CI [36.23%, 49.00%], Exact 232, Close 1, Wrong 98, Missing 213)
- Key paired differences: E vs D **+19.12 pp** ($p = 1.38 \times 10^{-17}$); E vs C **+7.54 pp** ($p = 0.0021$); E vs `cum_plus_validation` (grid contribution) **+13.05 pp** ($p = 1.43 \times 10^{-13}$).

**OCR cumulative preprocessing & post-processing ablation (n=131, 544 fields)**

- `cum_raw` (no preprocessing): **29.96%** (95% CI [23.70%, 36.60%], Exact 163, Wrong 52, Missing 328)
- `+ deskew`: **28.68%** (-1.29 pp vs previous)
- `+ deglare`: **28.68%** (+0.00 pp vs previous)
- `+ CLAHE`: **23.35%** (-5.33 pp vs previous)
- `+ ROI`: **23.53%** (+0.18 pp vs previous)
- `+ semantic filter`: **22.24%** (-1.29 pp vs previous)
- `+ fuzzy matching`: **27.21%** (+4.96 pp vs previous)
- `+ energy/fat fallback`: **30.33%** (+3.12 pp vs previous)
- `+ clinical validation`: **29.60%** (-0.74 pp vs previous)
- `Full pipeline (grid enabled)`: **42.65%** (+13.05 pp vs previous)

**OCR leave-one-out (LOO) post-processing ablation (n=131, 544 fields)**

- `loo_full_minus_semantic`: **46.14%** (**+3.49 pp** vs full pipeline, 95% CI [+1.08%, +6.23%], McNemar $p = 8.8 \times 10^{-4}$) — negative finding reported explicitly
- `loo_full_minus_fuzzy`: **32.35%** (**-10.29 pp** vs full pipeline, $p = 1.2 \times 10^{-10}$)
- `loo_full_minus_fallback`: **42.10%** (-0.55 pp vs full pipeline, $p = 0.453$, NS)
- `loo_full_minus_validation`: **43.01%** (+0.37 pp vs full pipeline, $p = 0.50$, NS)

**HITL quantitative correction workload (E_full_pipeline on 131 verified images)**

- Fields accepted unchanged: **232 / 544** (**42.6%**)
- Total edit actions: **423** (including 98 wrong-value replacements, 213 missing-value entries, and 111 spurious-value clearances on `absent` fields; 64.6% of 655 actionable cells)
- Images requiring $\ge 1$ edit: **100 / 131** (**76.3%**)
- Edits per image: median **3**, mean **3.23**, maximum **10**

**Clinical rules: verification and end-to-end propagation (n=131 verified images × 6 profiles = 786 decisions; 426 on 71 table-bearing images)**

- Implementation vs specification (`frontend/js/clinical-rules.js` vs `verify_clinical_rules.js`): daily limits **36/36**, status decisions **30/30**
- Status unchanged vs GT (all 786 decisions / 426 table-image decisions):
  - C. Raw PaddleOCR: **600/786 (76.3%)** / **284/426 (66.7%)**; GT DANGER → SAFE: **9/25**; GT WARNING → SAFE: **47/117**
  - D. Parser without grid: **644/786 (81.9%)** / **308/426 (72.3%)**; GT DANGER → SAFE: **17/25**; GT WARNING → SAFE: **71/117**
  - E. Full pipeline: **668/786 (85.0%)** / **335/426 (78.6%)**; GT DANGER → SAFE: **18/25 (72.0%)**; GT WARNING → SAFE: **55/117**

**Edge CPU benchmark (`experiments/results/edge/cpu/bench.md`)**

- YOLO11s-seg ($n=30$, 640 px, CPU): median **0.1304 s**, P95 **0.1854 s**, peak RSS **770.0 MB**
- Full OCR pipeline ($n=12$, CPU): median **20.5777 s**, P95 **45.2932 s**, peak RSS **3,760.8 MB**
- Clinical Rules Engine (CPU): median **0.000039 s** (0.039 ms) per call
- Sequential end-to-end sum (CPU): **20.7081 s**
- Artifact sizes (MB): `yolo_last_pt` **20.5**, `yolo_best_onnx` **40.6**, `ppocr_det` **4.9**, `ppocr_rec` **7.8**, `slanet` **8.0** (combined `.pt` + OCR = **41.2 MB**)
- Local GPU not benchmarked because local PyTorch is CPU-only (`torch 2.15.0+cpu`, `torch.cuda.is_available() = False`).

## Manuscript-ready tables

- `table_dataset.md`
- `table_segmentation_internal.md`
- `table_segmentation_external.md`
- `table_segmentation_pixel.md`
- `table_segmentation_per_class.md`
- `table_ocr.md`
- `table_ocr_ablation.md`
- `table_clinical.md`
