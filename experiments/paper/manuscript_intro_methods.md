# Healthier — Introduction & Research Methods (fact-checked draft)

> **Companion files.** Abstract, Results & Discussion, and Conclusions are in
> `experiments/paper/manuscript_results_conclusions.md`; references are in
> `experiments/paper/references.bib`.
>
> **Purpose.** Publish-style Introduction and Research Methods for the Healthier
> manuscript, aligned with what the repository and experiments actually contain.
> Every number traces to `experiments/paper/final_numbers.md` or `experiments/results/`.
> Inline citations are APA-style (author–year) so Mendeley can match them to
> `references.bib` in the same folder.
>
> **Markers used below**
> - `[TO COMPLETE]` = experiment not finished yet (do not invent a number).
> - `[FIGURE n — take from …]` = ready-made asset and its exact path.
> - `[VERIFY]` = source cannot be checked from the repository; confirm in Mendeley.

---

# INTRODUCTION

Non-communicable diseases (NCDs) are strongly associated with dietary patterns, and
diabetes and hypertension remain among the largest contributors to the disease burden
in Indonesia. The 2023 Indonesian Health Survey (SKI) reported a diabetes prevalence of
11.7%, an increase from 10.9% in the 2018 Basic Health Research (Riskesdas), while
hypertension prevalence decreased from 34.1% to 30.8% over the same period
(Kementerian Kesehatan Republik Indonesia, 2024). Projections indicate that diabetes
morbidity and mortality in Indonesia will continue to rise until at least 2045
(Wahidin et al., 2024), and analyses of the 2023 SKI link the increase to lifestyle and
metabolic factors such as central obesity and infrequent blood-pressure screening
(Mivtahurrahimah & Sevtiyani, 2025). Treatment-seeking behaviour also remains
inadequate, since a substantial share of people diagnosed with hypertension or diabetes
do not receive regular treatment or follow-up (Kementerian Kesehatan Republik
Indonesia, 2024). These figures point to a need for practical, everyday self-management
tools rather than clinical intervention alone.

At the individual level, the ability to interpret nutrition information is limited. In
a survey of 1,029 adults in Semarang, 96.02% of respondents had problematic
nutrition-fact literacy (Aprianti et al., 2023). A systematic review of traffic-light
labelling in Indonesia likewise reported that consumer understanding of nutrition
labels remains low (Shanti, 2025). Interpretive front-of-pack labels can improve
consumers' understanding of nutritional quality, although effects on purchasing and
intake vary across settings (Bonaccio et al., 2023; Ganderats-Fuentes & Morgan, 2023;
Shrestha et al., 2023), and Indonesia has begun introducing label formats such as the
Nutri-Level label with the same objective (Kahendra et al., 2026). In practice, however,
conventional nutrition tables remain difficult to process automatically because the text
can be small, layouts vary across products, and reflections or curved packaging degrade
image quality (Nagayi et al., 2025). For meals served on a plate there is no printed
table at all, so dietary assessment must begin from visual food recognition (Dalakleidi
et al., 2022; Mohanty et al., 2022).

Image-based dietary assessment is usually described as a pipeline of food image
analysis, volume or portion estimation, and nutrient derivation (Wang et al., 2022).
Reviews of the field note that segmentation, recognition, and nutrient estimation are
commonly developed as separate tasks, and that dataset diversity, portion estimation,
and real-world image conditions remain persistent challenges (Chotwanvirat et al.,
2024; Shonkoff et al., 2023; Sultana et al., 2023). For people living with diabetes
specifically, a scoping review of calorie-counting applications shows that food
segmentation, classification, and volume estimation are the main computer-vision
components (Rouhafzay et al., 2025). Indonesian studies have evaluated YOLO-based
detection of local foods and nutritional estimation (Hananta & Cahyono, 2026; Utami et
al., 2023), but the literature remains fragmented across pipeline stages.

A recurring limitation is cultural coverage. Food-recognition research is heavily
biased toward Western foods, and contributions covering Indonesian cuisine remain
limited (Fauzan & Pamungkas, 2025). Indonesian food datasets do exist, but they are
mainly classification collections; for example, Rasyidi et al. (2024) introduced 24,427
images across 160 Indonesian food types for classification. To the best of our
knowledge, there is still no public instance-segmentation benchmark for Indonesian
plated meals. Generalization across sources is a known concern in food segmentation as
well: recent benchmarks show that current methods suffer from limited data diversity
and degrade under new viewpoints and domains (AlMughrabi et al., 2026). Because no
Indonesian plate-segmentation benchmark is available, domain shift must be measured
explicitly rather than assumed away.

The label-reading problem presents a parallel challenge. A benchmark on real-world food
packaging compared Tesseract, EasyOCR, PaddleOCR, and TrOCR, and found that dense
tabular structures such as nutrition-facts panels cause recognition errors (Nagayi et
al., 2025). Preprocessing, layout reconstruction, and domain-specific correction are
therefore commonly used (Alireza & Kurniawardhani, 2026; Rosyadi et al., 2024; Saputra
et al., 2024, 2026). PaddleOCR provides an Apache-licensed OCR toolkit that includes the
PP-OCR model family and is practical for local deployment (Cui et al., 2025).

Automated recognition still makes mistakes, so human verification matters. Manual
correction remains valuable when recognition encounters ambiguous foods or portions
(Folson et al., 2023; Moyen et al., 2022; Oei et al., 2024). In a longitudinal
food-logging deployment, image recognition still required a 20.7% log edit rate (Bakar
et al., 2025), which motivates the Human-in-the-Loop (HITL) design adopted here
(Mosqueira-Rey et al., 2023; Yuan et al., 2024). Local inference is also attractive for
health applications because on-device processing can reduce privacy exposure and
dependence on continuous cloud connectivity (Heydari & Mahmoud, 2025; Wang et al.,
2025).

This study therefore develops and evaluates **Healthier**, a nutrition-assistance
application that combines (1) YOLO11s-seg instance segmentation of plate images into
five Isi Piringku groups, (2) PaddleOCR-based extraction of nutrition-table values from
packaging, and (3) a deterministic Clinical Rules Engine with explicit user
verification. The segmentation model is trained on public datasets that are not
specific to Indonesian cuisine, so domain shift is treated as a research question rather
than hidden. The contributions of this study are:

1. A leakage-controlled data protocol for food segmentation (perceptual-hash
   deduplication, a fixed validation split, and three external test sets that were never
   used for training or model selection), together with a per-class analysis of
   generalization across sources.
2. A reproducible OCR pipeline for nutrition tables, with a grid-based layout parser,
   field-level evaluation using image-level cluster bootstrap, and [**ablation of each
   preprocessing stage — TO COMPLETE**].
3. An integrated decision layer that links segmentation and label values to
   guideline-based SAFE/WARNING/DANGER classes, evaluated with [**end-to-end error
   propagation and HITL correction effect — TO COMPLETE**].

The research question is: how well do segmentation and label extraction that are
trained or configured on non-Indonesian sources generalize to new sources, and how can
their outputs be translated into transparent, guideline-aligned feedback with explicit
user verification?

---

# RESEARCH METHODS

This research adopts an iterative experimental methodology informed by CRISP-ML(Q), with
separate evaluation of the segmentation, OCR, clinical-rule, and usability components.
The design emphasizes repeated validation and strict separation of data because
variability in food images, datasets, ground truth, and reporting practices is a major
barrier to reliable comparison between dietary-assessment systems (Chotwanvirat et al.,
2024; Dalakleidi et al., 2022; Shonkoff et al., 2023). Segmentation training was run on
a Kaggle Tesla T4 GPU (15.6 GB VRAM), whereas OCR experiments and deployment tests were
run on a laptop with a 12th-generation Intel Core i5 processor, 8 GB RAM, and an NVIDIA
RTX 3050 GPU (6 GB). Software versions were Ultralytics 8.4.199, PyTorch 2.10.0+cu128,
and Python 3.12.13 for segmentation, and PaddleOCR 3.7.0 (with the PP-OCRv4 recognition
model) for OCR; the full environment is recorded in the project repository.

`[FIGURE 1 — System architecture: two input paths (plate image → segmentation;
packaging image → OCR), a shared HITL verification layer, and the deterministic
Clinical Rules Engine. Redraw from the Mermaid diagram in `docs/architecture.md` (there
is no PNG yet).]`

## Segmentation Datasets and Splits

Three public datasets were used: UECFoodPIXCOMPLETE (Okamoto & Yanai, 2021),
FoodSeg103 (Wu et al., 2021), and FruitSeg30 (Shamrat et al., 2024). FruitSeg30
contains 1,969 images in 30 fruit classes, collected in natural and uncontrolled
environments across Malaysia, Bangladesh, and Australia (Shamrat et al., 2024). The
candidate corpus contained 19,087 images. Duplicates were detected with a difference
hash (dHash) computed on an 8×8 grayscale reduction of each image
(`prepare_yolo_seg.py`); 373 duplicate images were removed, leaving 18,714 unique
images. `[VERIFY: regenerate `experiments/dataset/dataset_summary.json` to reproduce the
373 count; that file is not tracked in the repository.]` Class labels from all sources
were remapped into five Isi Piringku groups — Staple Food, Side Dishes, Vegetables,
Fruits, and Others — using a fixed mapping table.

The training pool consisted of the official training portions of UEC and FoodSeg103 and
was split 85:15 into training (11,684) and validation (2,017) sets (stratified, seed
42, no overlap; validation images are a held-out 15% of the training pool). No
oversampling and no class-weighted loss were applied, so that the external
generalization results are not confounded by resampling. The official UEC test set (948
images), the FoodSeg103 validation set (2,096 images), and the whole of FruitSeg30
(1,969 images) were kept as external test sets (5,013 images total). They were not used
for training, early stopping, or checkpoint selection. FruitSeg30 therefore acts as a
deliberate domain-shift test, because the Fruits class in the training data comes only
from FoodSeg103.

`[TABLE 1 — Dataset and split composition. Take from `experiments/paper/tables/table_dataset.md`.]`

`[FIGURE 2 — Dataset composition / split diagram (optional). Build from Table 1.]`

## Segmentation Model Development

The YOLO11s-seg model (Ultralytics, 2024; Khanam & Hussain, 2024) was trained with
Ultralytics 8.4.199 for at most 100 epochs with early-stopping patience 20; training
stopped at epoch 67 and the best checkpoint was epoch 48. Optimization used AdamW with
an initial learning rate of 0.001, a final learning-rate factor of 0.01, weight decay
0.0005, and momentum 0.9, with batch size 16, image size 640, seed 42, and mixed
precision enabled. Spatial and photometric augmentation was applied to improve
robustness (HSV jitter, translation, scaling, horizontal flipping, mosaic, and random
erasing). Training required 4.01 h on the Tesla T4. `[TABLE 2 — Run configuration. Take
from `final_numbers.md` §"Run configuration".]`

For plate images, the region for each food group is obtained as a segmentation polygon,
and the nutritional proportion is computed as the ratio of the food-polygon area to the
analyzed plate area. Detections of non-food objects (cutlery, the plate itself, hands)
are filtered out so that proportions are based on food pixels only.

`[FIGURE 3 — Training curves (loss and validation metrics). Take from
`experiments/runs/seg_main/results.png`.]`
`[FIGURE 4 — Internal validation confusion matrix (normalized). Take from
`experiments/runs/seg_main/confusion_matrix_normalized.png`.]`
`[FIGURE 5 — Internal qualitative results: ground truth vs prediction. Take from
`experiments/runs/seg_main/val_batch0_labels.jpg` and
`experiments/runs/seg_main/val_batch0_pred.jpg`.]`

## OCR Extraction Pipeline

Packaging images pass through four preprocessing stages before recognition: deskewing
using the Hough transform, deglare using Telea inpainting (Telea, 2004), CLAHE-based
contrast normalization, and ROI cropping around the nutrition table. The processed
region is recognized with PaddleOCR (Cui et al., 2025) configured with the **PP-OCRv4**
recognition model. Layout reconstruction then (i) groups text boxes into logical rows
using Y-band grouping, (ii) detects the value columns (Per Sajian / Serving and %AKG /
%DV) against the row-label column, and (iii) splits tall numeric boxes that PP-OCR
merges across adjacent rows (for example, `1725` is split into 1, 7, 2, 5), which
separates the per-serving and %AKG columns. A semantic filter removes non-nutritional
text (promotional slogans, BPOM registration numbers, ingredients, cooking
instructions), and fuzzy matching based on edit distance maps OCR tokens to standardized
nutrition terms using a similarity threshold of ≥ 0.75:

`lev(a, b) = max(|a|, |b|) if min(|a|, |b|) = 0`  (1)

`Similarity(a, b) = 1 − lev(a, b) / max(|a|, |b|)`  (2)

Field-level accuracy is used because the downstream system consumes specific nutrition
fields rather than full transcriptions: a field is counted as correct when it is exact
or within 5% of the ground-truth per-serving value. Exact, wrong, and missing fields are
reported separately, and confidence intervals are obtained by a cluster bootstrap over
images (2,000 iterations, seed 42) rather than a sample-size-based interval.

At the time of writing, OCR evaluation is an **exploratory pilot on 6 manually
transcribed Indonesian snack labels (46 fields)**; the field-level result (baseline
parser 21.74% → grid parser 47.83%) must not be reported as a generalization result
because the confidence interval is wide and the sample is not stratified by brand or
store. The protocol reserves ≥ 100 images split **by product/brand** (not randomly by
image) for the headline evaluation. `[TO COMPLETE: ≥100-image evaluation and the
per-stage ablation.]`

`[TABLE 6 — OCR field-level pilot. Take from `experiments/paper/tables/table_ocr.md`.]`

`[FIGURE 9 — OCR pipeline diagram (four preprocessing stages → PP-OCRv4 → grid parser →
semantic filter → fuzzy matching). Redraw from `docs/architecture.md`; there is no PNG
yet.]`
`[FIGURE 10 — OCR qualitative example (input, ROI crop, reconstructed table). Example
crops already available: `experiments/results/ocr_vlm/gt_audit/134337_kolom_nilai.png`,
`.../134337_lemak_total.png`, `.../132308_gula.png`.]`

## Clinical Rules Engine

Nutrition values that have passed HITL verification are evaluated by a deterministic
rules engine that runs in the browser (no cloud inference). The engine computes Basal
Metabolic Rate (BMR) with the Mifflin–St Jeor equations and Total Daily Energy
Expenditure (TDEE) as BMR multiplied by an activity factor (1.2 for sedentary, used in
the worked example):

`BMR_male = 10W + 6.25H − 5A + 5`  (3)

`BMR_female = 10W + 6.25H − 5A − 161`  (4)

where W is body weight (kg), H is height (cm), and A is age (years). Personal limits are
then derived for each user profile. For type-2 diabetes, the daily sugar limit follows
PERKENI 2024 and is set dynamically at 5% of TDEE (`0.05 × TDEE / 4` g); for
non-diabetic profiles the default is 10% of TDEE. Sodium limits are absolute values —
1,500 mg/day for hypertension and 2,300 mg/day otherwise. Saturated-fat limits are 10%
of TDEE (`0.10 × TDEE / 9` g), reduced to 7% for dyslipidemia; cholesterol is limited
to 200 mg for dyslipidemia; and the protein limit for diabetic nephropathy is
`0.8 g × body weight`.

The engine classifies accumulated intake into SAFE, WARNING, and DANGER. A WARNING is
raised when sugar **or** saturated fat exceeds **50%** of the daily limit, when actual
sodium exceeds **40%** of the sodium limit, when protein exceeds **30%** of the
nephropathy quota, or when a product's sodium density exceeds **600 mg/100 g** for
hypertension (800 mg/100 g otherwise). DANGER is raised **only** when sugar exceeds
**100%** of the daily limit; sodium and saturated fat saturate at WARNING. These bands
are the thresholds implemented in `frontend/js/clinical-rules.js` and are reported here
as a **rule verification**, not as clinical validation, because no independent
registered-dietitian review was conducted. `[TABLE 7 — 10–20 profile × food scenarios
with expected SAFE/WARNING/DANGER status. TO COMPLETE.]`

`[FIGURE 11 — Clinical decision flow (limits → per-meal evaluation → SAFE / WARNING /
DANGER). Redraw from `frontend/js/clinical-rules.js`; there is no PNG yet.]`

## Human-in-the-Loop and Usability

Because automated recognition can fail on inputs outside its distribution, Healthier
provides two interactive controls: a **Segmentation Editor** that lets the user correct
predicted polygons before analysis, and a **Nutrient Value Verification Form** that lets
the user edit OCR-extracted values before the clinical rules are applied. This design
follows the HITL literature, which emphasizes human participation for handling uncertain
predictions and preserving meaningful human control (Mosqueira-Rey et al., 2023; Yuan et
al., 2024).

System usability was assessed with Nielsen's heuristic evaluation using four evaluators:
two independent machine-learning engineers/data scientists (E1, E2), one lead data
scientist performing an asynchronous architecture review (E3), and one self-evaluator
(E4) verifying documentation–implementation consistency; E4's findings were excluded
from the severity weighting. `[TO COMPLETE: this section requires the author's own
usability notes — the number of findings per severity (0–4), per evaluator, is not
recorded anywhere in the repository. Replace the placeholder below, or report usability
as future work.]`

`[TABLE 8 — Evaluator profile. Take from the current draft (Table 2).]`

## Metrics and Statistical Analysis

Segmentation is evaluated at the instance level (box and mask precision, recall, mAP50,
and mAP50-95) and at the pixel level (per-class accuracy, precision, recall, F1), with
normalized confusion matrices for the internal validation set and for each external test
set. Per-class average precision is reported to expose source-specific failure modes.
OCR is evaluated with field-level accuracy and exact/wrong/missing rates with a
cluster-bootstrap confidence interval over images. Clinical-rule behaviour is reported
as a deterministic scenario comparison (rule verification). A full-budget comparison
against an alternative segmentation architecture was not completed and is reported as a
screening-only baseline `[TO COMPLETE: baseline or label it as screening]`.

`[TABLE 3 — Internal validation metrics. Take from
`experiments/paper/tables/table_segmentation_internal.md` (fix training time to 4.01 h).]`
`[TABLE 4 — External test metrics. Take from
`experiments/paper/tables/table_segmentation_external.md`.]`
`[TABLE 5 — Per-class external mask AP. Take from
`experiments/paper/tables/table_segmentation_per_class.md`.]`

`[FIGURE 6 — External-set normalized confusion matrices (UEC / FoodSeg103 /
FruitSeg30). Take from
`experiments/results/segmentation/uec_test_instance/confusion_matrix_normalized.png`,
`.../foodseg_val_instance/confusion_matrix_normalized.png`,
`.../fruitseg_instance/confusion_matrix_normalized.png`.]`
`[FIGURE 7 — Mask precision–recall / F1 curves per external set. Take from each set's
`MaskPR_curve.png` and `MaskF1_curve.png` under `experiments/results/segmentation/`.]`
`[FIGURE 8 — Per-class external mask AP50 bar chart (optional). Build from
`experiments/results/segmentation/per_class_all.csv`.]`

### Note on the OCR engine version

The manuscript draft states "PP-OCRv5", but the code pins the recognition model to
**PP-OCRv4** (`backend/app.py`, `NUTRILABEL_OCR_VERSION`), and PP-OCRv4 is a supported
model in PaddleOCR 3.7.0. The library version (3.7.0) is not the model version. Either
report **PP-OCRv4** (matches the code and all current results) or re-run the OCR
evaluation with PP-OCRv5 and update both the text and the tables. `[VERIFY: after one
real run, read `status.model` and the PaddleOCR init log to confirm which model is
actually loaded.]`

---

# Figure and table checklist

| # | Content | Source path in repo | Status |
| :--- | :--- | :--- | :--- |
| Fig. 1 | System architecture (two paths + HITL + rules) | redraw from `docs/architecture.md` | **create PNG** |
| Fig. 2 | Dataset/split composition | from `tables/table_dataset.md` | optional |
| Fig. 3 | Training curves | `experiments/runs/seg_main/results.png` | ready |
| Fig. 4 | Internal confusion matrix | `experiments/runs/seg_main/confusion_matrix_normalized.png` | ready |
| Fig. 5 | Internal GT vs prediction | `experiments/runs/seg_main/val_batch0_labels.jpg` + `val_batch0_pred.jpg` | ready |
| Fig. 6 | External confusion matrices | `experiments/results/segmentation/{uec_test_instance,foodseg_val_instance,fruitseg_instance}/confusion_matrix_normalized.png` | ready |
| Fig. 7 | External PR/F1 curves | same folders, `MaskPR_curve.png` / `MaskF1_curve.png` | ready |
| Fig. 8 | Per-class external AP50 | `experiments/results/segmentation/per_class_all.csv` | build |
| Fig. 9 | OCR pipeline diagram | redraw from `docs/architecture.md` | **create PNG** |
| Fig. 10 | OCR qualitative example | `experiments/results/ocr_vlm/gt_audit/*.png` | ready (examples) |
| Fig. 11 | Clinical decision flow | redraw from `frontend/js/clinical-rules.js` | **create PNG** |
| Fig. 12 | HITL UI screenshots (editor + verification form) | run the app / `frontend/` | **screenshot** |
| Table 1 | Dataset and splits | `tables/table_dataset.md` | ready (373 caveat) |
| Table 2 | Run configuration | `final_numbers.md` | ready |
| Table 3 | Internal metrics | `tables/table_segmentation_internal.md` | fix time → 4.01 h |
| Table 4 | External metrics | `tables/table_segmentation_external.md` | ready |
| Table 5 | Per-class external AP | `tables/table_segmentation_per_class.md` | ready |
| Table 6 | OCR pilot | `tables/table_ocr.md` | exploratory |
| Table 7 | Clinical scenarios | build from `clinical-rules.js` | **TO DO** |
| Table 8 | Evaluator profile | current draft Table 2 | needs user data |

> **Optional / future work only:** the cloud-VLM comparison (`vlm_analysis.md`,
> `experiments/results/ocr_vlm/`) is a future-work exploration (n=6). It is not part of
> the core claim (edge OCR without cloud LLM) and should not be presented as a main
> result.

---

# Open items before submission (honest list)

1. **OCR headline** — expand the pilot to ≥ 100 images split by brand; run the
   preprocessing ablation. Until then, describe OCR as exploratory.
2. **373 duplicates** — regenerate `experiments/dataset/dataset_summary.json` to
   reproduce the count; otherwise soften the wording.
3. **Latency** — re-measure inference time on the deployment device; training time is
   not inference time, and the old "168 ms/image" number is not verified.
4. **Baseline** — train a full-budget alternative segmentation model or label the
   existing comparison as screening-only.
5. **Usability** — provide the Nielsen findings per severity from your own notes, or
   move usability entirely to future work.
6. **Clinical sources** — verify the 5%-TDEE sugar value in the PERKENI 2024 document
   and the sodium values/pages in the cited guidance. The code values are
   1,500 mg (hypertension) and 2,300 mg (general).
7. **PP-OCR version** — confirm the model actually loaded (see note above).
