# Application of Deep Learning for Nutritional Segmentation and Information Extraction in the Healthier Application

Tata Aditya Pamungkas^1, Nuke Lu'lu Ul Chusna^2, Harry Dwiyana Kartika^3

^1,2,3 Informatics Engineering, Universitas Krisnadwipayana, Jakarta, Indonesia

tataaditaya@gmail.com, nukelchusna@unkris.ac.id, harry_d@unkris.ac.id

---

## ABSTRACT

Low nutritional literacy and the high risk of non-communicable diseases (NCDs) have
driven the development of **Healthier**, an application that combines food recognition
from plate images with nutrition-label extraction from packaging. The system integrates
three components: (1) a YOLO11s-seg instance-segmentation model that partitions plate
images into five *Isi Piringku* groups; (2) a PaddleOCR-based extraction pipeline
(PP-OCRv4) with a grid-based layout parser for nutrition tables; and (3) a deterministic
Clinical Rules Engine that converts verified values into SAFE/WARNING/DANGER feedback,
with a Human-in-the-Loop (HITL) correction step. The segmentation model was trained on
11,684 images and validated on 2,017 images from public datasets that are not specific
to Indonesian cuisine. Generalization was measured on three independent external test
sets (5,013 images) that were never used for training or model selection. On internal
validation the model reached a mask mAP50 of 71.66% and a mask mAP50-95 of 56.91%
(mask precision 72.95%, recall 65.50%). It transferred well to two food-photography
sources (mask mAP50 69.15% on UECFoodPIXCOMPLETE and 63.66% on FoodSeg103) but degraded
sharply on a fruit-only, in-the-wild dataset (17.06%), which we report as a domain shift
rather than conceal. Label extraction was evaluated with field-level accuracy on
Indonesian snack labels; a six-image pilot showed that a grid-based parser more than
doubled accuracy relative to a baseline parser (21.74% to 47.83%), so this evaluation is
reported as an exploratory pilot pending a larger brand-stratified set. The main
contribution is not any single model, but a reproducible integration and decision
mechanism whose behaviour is demonstrated experimentally, including cross-source
generalization, per-class error analysis, and a clearly stated evaluation protocol. The
results confirm the technical feasibility of an offline, transparent nutrition-assistance
pipeline while making its generalization boundaries explicit.

**Keywords:** Instance Segmentation; OCR; YOLO11s-seg; PaddleOCR; Nutrition Literacy

## ABSTRAK

Literasi gizi yang rendah dan tingginya risiko penyakit tidak menular (PTM) mendorong
pengembangan **Healthier**, aplikasi yang menggabungkan pengenalan makanan dari citra
piring dengan ekstraksi label gizi dari kemasan. Sistem ini mengintegrasikan tiga
komponen: (1) model segmentasi instan YOLO11s-seg yang membagi citra piring menjadi lima
kelompok *Isi Piringku*; (2) pipeline ekstraksi berbasis PaddleOCR (PP-OCRv4) dengan
parser tata letak berbasis grid untuk tabel gizi; dan (3) Clinical Rules Engine
deterministik yang mengubah nilai terverifikasi menjadi status SAFE/WARNING/DANGER,
dengan tahap koreksi Human-in-the-Loop (HITL). Model segmentasi dilatih pada 11.684 citra
dan divalidasi pada 2.017 citra dari dataset publik yang tidak spesifik masakan
Indonesia. Generalisasi diukur pada tiga set uji eksternal independen (5.013 citra) yang
tidak pernah dipakai untuk pelatihan maupun pemilihan model. Pada validasi internal,
model mencapai mask mAP50 71,66% dan mask mAP50-95 56,91% (presisi 72,95%, recall
65,50%). Model bertransfer baik ke dua sumber fotografi makanan (mask mAP50 69,15% pada
UECFoodPIXCOMPLETE dan 63,66% pada FoodSeg103) tetapi turun tajam pada dataset khusus
buah di lingkungan alami (17,06%), yang kami laporkan sebagai *domain shift*. Ekstraksi
label dievaluasi dengan akurasi tingkat field pada label kemasan Indonesia; pilot enam
citra menunjukkan parser berbasis grid lebih dari menggandakan akurasi dibandingkan
parser dasar (21,74% menjadi 47,83%), sehingga evaluasi ini dilaporkan sebagai pilot
eksploratif. Kontribusi utama bukan pada satu model tunggal, melainkan pada mekanisme
integrasi dan pengambilan keputusan yang reproducible dan dibuktikan secara eksperimen,
mencakup generalisasi antar-sumber, analisis kesalahan per kelas, dan protokol evaluasi
yang dinyatakan terbuka.

**Kata kunci:** Segmentasi Instan; OCR; YOLO11s-seg; PaddleOCR; Literasi Gizi

---

## 1. INTRODUCTION

Non-communicable diseases (NCDs) are strongly associated with dietary patterns, and
diabetes and hypertension remain among the largest contributors to the disease burden in
Indonesia. The 2023 Indonesian Health Survey (SKI) reported a diabetes prevalence of
11.7%, an increase from 10.9% in the 2018 Basic Health Research (Riskesdas), while
hypertension prevalence decreased from 34.1% to 30.8% over the same period (Kementerian
Kesehatan Republik Indonesia, 2024). Analyses of the 2023 SKI further link the increase
in diabetes cases to modifiable lifestyle and metabolic factors such as central obesity
and infrequent blood-pressure screening (Mivtahurrahimah & Sevtiyani, 2025). Because a
substantial share of diagnosed individuals do not maintain regular treatment or follow-up
(Kementerian Kesehatan Republik Indonesia, 2024), practical tools that support everyday
self-management are needed alongside clinical care.

At the individual level, the ability to interpret nutrition information is limited. A
systematic review of traffic-light nutrition labelling in Indonesia reported that
consumer understanding of nutrition labels remains low, even though attitudes toward
simplified label formats are generally positive (Shanti, 2025). New front-of-pack
initiatives such as the Nutri-Level label have been introduced with the same objective of
helping consumers choose healthier products (Kahendra et al., 2026). In practice, however,
conventional nutrition tables are difficult to process automatically because the text can
be small, layouts vary across products, and reflections or curved packaging degrade image
quality (Nagayi et al., 2025). For meals served on a plate there is no printed table at
all, so dietary assessment must begin from visual food recognition.

Image-based dietary assessment is usually described as a pipeline of food image
analysis, volume or portion estimation, and nutrient derivation (Wang et al., 2022).
Reviews of the field note that segmentation, recognition, and nutrient estimation are
commonly developed as separate tasks, and that dataset diversity, portion estimation, and
real-world image conditions remain persistent challenges (Rouhafzay et al., 2025;
Sultana et al., 2023). In particular, a scoping review of calorie-counting applications
for people living with diabetes shows that food segmentation, classification, and volume
estimation are the main computer-vision components of such systems (Rouhafzay et al.,
2025). Indonesian contributions remain scarce: most existing work focuses on
classification of local dishes rather than instance segmentation of plated meals
(Fauzan & Pamungkas, 2025; Rasyidi et al., 2024). For example, Rasyidi et al. (2024)
introduced a classification dataset of 24,427 images covering 160 Indonesian food types,
but to the best of our knowledge there is still no public instance-segmentation benchmark
for Indonesian plated meals. Generalization across sources is a known concern as well:
recent food-video segmentation benchmarks show that current methods suffer from limited
data diversity and degrade under new viewpoints and domains (AlMughrabi et al., 2026).
Domain shift must therefore be measured explicitly rather than assumed away.

The label-reading problem presents a parallel challenge. A benchmark on real-world food
packaging compared Tesseract, EasyOCR, PaddleOCR, and TrOCR, and found that dense tabular
structures such as nutrition-facts panels are a major source of recognition errors
(Nagayi et al., 2025). For this reason, preprocessing, layout reconstruction, and
domain-specific correction are commonly required in Indonesian settings (Saputra et al.,
2024; Saputra et al., 2026; Alireza & Kurniawardhani, 2026). PaddleOCR provides an
Apache-licensed OCR toolkit that includes the PP-OCR model family and is practical for
local deployment (Cui et al., 2025).

Automated recognition still makes mistakes, so human verification matters. In a
longitudinal food-logging study, image recognition still required a 20.7% log edit rate,
showing that users remain an essential part of the loop (Bakar et al., 2025). This
motivates the HITL design adopted in Healthier, where users can correct segmentation and
extracted values before they are used by the decision layer. Local inference is also
attractive for health applications because on-device processing can reduce privacy
exposure and dependence on continuous cloud connectivity (Heydari & Mahmoud, 2025; Wang
et al., 2025).

### 1.1 Research Gap

The literature provides strong evidence for individual components, but relatively few
studies integrate plate-based food analysis, packaged-label extraction, and deterministic
guideline-based decision-making into a single offline workflow with explicit human
verification. Healthier addresses this integration gap. The research questions are:

- **RQ1.** How well does a segmentation model trained on non-Indonesian food datasets
  generalize to new and unseen sources, and which classes fail?
- **RQ2.** To what extent does a layout-aware OCR pipeline improve nutrition-table
  extraction over a baseline parser, and what error modes remain?
- **RQ3.** How can segmentation and label outputs be translated into transparent,
  guideline-aligned feedback with explicit user verification?

### 1.2 Contributions

This study makes three explicit contributions. Each is stated together with how it is
demonstrated experimentally, so that the novelty rests on a verifiable integration and
decision mechanism rather than on merely combining existing technologies.

1. **A leakage-controlled evaluation protocol for cross-source food segmentation.**
   We define a data protocol with perceptual-hash deduplication, a fixed validation
   split, and three independent external test sets that are never used for training or
   model selection, and we report per-class generalization across sources. *Evidence:*
   the internal/external results and the per-class error analysis in Sections 3.1–3.2
   (Tables 3–5; Figures 4–8).

2. **A reproducible, layout-aware OCR pipeline for nutrition tables.** We combine four
   preprocessing stages with a grid-based table parser and fuzzy field mapping, and we
   evaluate it at the field level with image-level cluster-bootstrap confidence
   intervals, together with a planned per-stage ablation. *Evidence:* the parser gain and
   error analysis in Section 3.4 (Table 6) and the ablation design in Section 2.9.

3. **A deterministic, guideline-grounded decision layer with HITL verification.** We
   translate verified segmentation and label values into SAFE/WARNING/DANGER feedback
   using explicit thresholds derived from PERKENI 2024 and hypertension guidance, and we
   retain an explicit user-correction step. *Evidence:* the rule specification in
   Section 2.7 and the rule-verification design in Section 3.5.

The novelty is therefore in the **integration mechanism, its validation protocol, and its
transparent decision strategy**, all of which are specified and testable, rather than in a
claim that any single model is state of the art.

---

## 2. RESEARCH METHODS

This research adopts an iterative experimental methodology informed by CRISP-ML(Q), with
separate evaluation of the segmentation, OCR, clinical-rule, and usability components.
The design emphasizes repeated validation and strict separation of data because
variability in food images, datasets, ground truth, and reporting practices is a major
barrier to reliable comparison between dietary-assessment systems (Sultana et al., 2023;
Wang et al., 2022). Figure 1 summarizes the end-to-end workflow.

**[Figure 1 — System architecture and experimental flow to be inserted here.]**
Recommended content: two input paths (plate image → YOLO11s-seg → five-group proportions;
packaging image → preprocessing → PP-OCRv4 → grid parser → semantic filter → fuzzy
matching), a shared HITL verification layer, the deterministic Clinical Rules Engine, and
the evaluation blocks (internal validation; three external tests; OCR field evaluation;
rule verification). Redraw from the Mermaid diagram in `docs/architecture.md`.

### 2.1 Experimental Environment

Segmentation training was run on a Kaggle Tesla T4 GPU (15.6 GB VRAM). OCR experiments
and deployment tests were run on a laptop with a 12th-generation Intel Core i5 processor,
8 GB RAM, and an NVIDIA RTX 3050 GPU (6 GB). The software environment and full training
parameters are listed in Table 3. The complete configuration is recorded in the project
repository to support replication.

### 2.2 Segmentation Datasets, Annotation, and Splits

Three public datasets were used: **UECFoodPIXCOMPLETE** (Okamoto & Yanai, 2021),
**FoodSeg103** (Wu et al., 2021), and **FruitSeg30** (Shamrat et al., 2024). FruitSeg30
contains 1,969 images in 30 fruit classes collected in natural and uncontrolled
environments across Malaysia, Bangladesh, and Australia (Shamrat et al., 2024). The
candidate corpus contained 19,087 images. Duplicate images were detected with a
difference hash (dHash) computed on an 8×8 grayscale reduction of each image; 373
duplicates were removed, leaving 18,714 unique images. Class labels from all sources were
remapped into five *Isi Piringku* groups (Staple Food, Side Dishes, Vegetables, Fruits,
Others) using a fixed mapping table.

**Annotation.** We did not re-annotate the public masks. The original annotations were
used as provided, and only the label remapping was applied. To document this step, the
protocol includes a stratified audit of the remapping (200 sampled images, 40 per class,
by two annotators, reporting Cohen's κ and resolving disagreements by discussion). This
audit is recommended practice and is listed as a reproducibility item in the future-work
discussion (Section 4).

**Splits and independent testing.** The training pool consisted of the official training
portions of UEC and FoodSeg103 and was split 85:15 into training (11,684) and validation
(2,017) sets, stratified by class with seed 42 and no overlap. Validation images are
therefore a held-out 15% of the training pool. Three sets were reserved exclusively for
external testing and were **never used for training, early stopping, or checkpoint
selection**: the official UEC test set (948 images), the FoodSeg103 validation set
(2,096 images), and the whole of FruitSeg30 (1,969 images), for 5,013 external images in
total. Because the Fruits class in the training data comes only from FoodSeg103,
FruitSeg30 acts as a deliberate domain-shift test. No oversampling and no class-weighted
loss were applied, so the external results are not confounded by resampling.

**[Table 1 — Dataset and split composition to be inserted here.]** Source:
`experiments/paper/tables/table_dataset.md`.

**[Table 2 — Per-class instance distribution per split to be inserted here.]** Source:
`experiments/paper/tables/table_dataset.md` (instance counts). This table is important
for the reviewer point about class distribution and consistency.

### 2.3 Segmentation Model and Training

The model is **YOLO11s-seg** (Khanam & Hussain, 2024). It was trained with Ultralytics
8.4.199 for at most 100 epochs with early-stopping patience 20; training stopped at epoch
67 and the best checkpoint was epoch 48. Optimization used AdamW with an initial learning
rate of 0.001, a final learning-rate factor of 0.01, weight decay 0.0005, and momentum
0.9, with batch size 16, image size 640, seed 42 (deterministic), and mixed precision
enabled. Spatial and photometric augmentation was applied (HSV jitter, translation,
scaling, horizontal flipping, mosaic, and random erasing). Training took 4.01 h on the
Tesla T4. The full parameter list is given in Table 3.

**[Table 3 — Training configuration to be inserted here.]** Source:
`experiments/paper/final_numbers.md` ("Run configuration").

### 2.4 Plate Proportion Estimation

For each plate image, the model outputs one segmentation polygon per detected food
instance. Instances are assigned to their *Isi Piringku* group, and the nutritional
proportion of a group is computed as the ratio of the sum of its food-polygon areas to
the total analyzed food area on the plate:

`Proportion_g = A_g / Σ_g A_g`  (1)

where `A_g` is the total polygon area of group `g`. Detections of non-food objects
(cutlery, the plate itself, hands) are filtered out before the proportion is computed, so
that the proportions are based on food pixels only.

### 2.5 OCR Extraction Pipeline

Packaging images pass through four preprocessing stages before recognition: deskewing
using the Hough transform, deglare using Telea inpainting (Telea, 2004), CLAHE-based
contrast normalization, and region-of-interest cropping around the nutrition table. The
processed region is recognized with PaddleOCR (Cui et al., 2025) configured with the
**PP-OCRv4** recognition model.

Layout reconstruction then: (i) groups text boxes into logical rows using Y-band
grouping; (ii) detects the value columns (Per Sajian / Serving and %AKG / %DV) relative
to the row-label column; and (iii) splits tall numeric boxes that PP-OCR merges across
adjacent rows (for example, `1725` is split into 1, 7, 2, 5), thereby separating the
per-serving and %AKG columns. A semantic filter removes non-nutritional text
(promotional slogans, BPOM registration numbers, ingredient lists, and cooking
instructions). Finally, fuzzy matching based on edit distance maps OCR tokens to
standardized nutrition terms. The Levenshtein distance and similarity are:

`lev(a, b) = max(|a|, |b|), if min(|a|, |b|) = 0`  (2)

`lev(a, b) = min( lev(a−1, b) + 1, lev(a, b−1) + 1, lev(a−1, b−1) + [a_i ≠ b_j] )`  (3)

`Similarity(a, b) = 1 − lev(a, b) / max(|a|, |b|)`  (4)

A token is accepted as a match when `Similarity ≥ 0.75`.

### 2.6 OCR Evaluation Protocol and Dataset

Field-level accuracy is used because the downstream system consumes specific nutrition
fields rather than full transcriptions. A field is counted as correct when it is exact or
within 5% of the ground-truth per-serving value:

`FieldAccuracy = (N_exact + N_within5%) / N_fields`  (5)

Exact, wrong, and missing fields are reported separately, and the confidence interval is
obtained by a cluster bootstrap over images (2,000 iterations, seed 42) rather than a
sample-size-based interval. The evaluated fields are energy, total fat, saturated fat,
trans fat, cholesterol, sodium, carbohydrate, fibre, sugar, protein, and serving size.
Values are taken per serving; fields absent from a package are recorded as empty rather
than zero.

**Dataset scope (addressing representativeness).** At the time of writing, the OCR
evaluation is an **exploratory pilot on six manually transcribed Indonesian snack
labels** (46 fields), transcribed by the author. Because the sample is small and not
stratified, this pilot is used only to document the effect of the layout parser and is
not reported as a generalization result. The full protocol reserves **at least 100
images** from different brands and stores, split **by product/brand** (not randomly by
image) in a 70/15/15 ratio, with an independent external test set collected at a
different time. Two annotators label at least 20% of the data to compute inter-annotator
agreement (Cohen's κ per field), with adjudication of disagreements.

### 2.7 Clinical Rules Engine

Nutrition values that have passed HITL verification are evaluated by a deterministic
rules engine that runs locally in the browser (no cloud inference). The engine computes
Basal Metabolic Rate (BMR) with the Mifflin–St Jeor equations and Total Daily Energy
Expenditure (TDEE) as BMR multiplied by an activity factor:

`BMR_male = 10W + 6.25H − 5A + 5`  (6)

`BMR_female = 10W + 6.25H − 5A − 161`  (7)

`TDEE = BMR × AF`  (8)

where `W` is body weight (kg), `H` is height (cm), `A` is age (years), and `AF` is the
activity factor (1.2 for sedentary, used in the worked example). Personal limits are then
derived per profile (Table 4). For type-2 diabetes, the daily sugar limit follows
PERKENI 2024 and is set dynamically at 5% of TDEE (`0.05 × TDEE / 4` g); for
non-diabetic profiles the default is 10% of TDEE (`0.10 × TDEE / 4` g). Sodium limits are
absolute values: 1,500 mg/day for hypertension and 2,300 mg/day otherwise. Saturated-fat
limits are 10% of TDEE (`0.10 × TDEE / 9` g), reduced to 7% for dyslipidemia;
cholesterol is limited to 200 mg for dyslipidemia; and the protein limit for diabetic
nephropathy is `0.8 g × body weight` (PERKENI, 2024; James et al., 2014).

**[Table 4 — Personal daily limits per clinical profile to be inserted here.]** Source:
`frontend/js/clinical-rules.js`.

The engine classifies accumulated intake into **SAFE**, **WARNING**, and **DANGER**
(Table 5). A WARNING is raised when sugar or saturated fat exceeds 50% of the daily
limit, when actual sodium exceeds 40% of the sodium limit, when protein exceeds 30% of
the nephropathy quota, or when a product's sodium density exceeds 600 mg/100 g for
hypertension (800 mg/100 g otherwise). DANGER is raised only when sugar exceeds 100% of
the daily limit; sodium and saturated fat saturate at WARNING. These bands are the exact
thresholds implemented in `frontend/js/clinical-rules.js` and are reported as **rule
verification**, not clinical validation, because no independent registered-dietitian
review was performed.

**[Table 5 — Risk classification thresholds to be inserted here.]** Source:
`frontend/js/clinical-rules.js`.

### 2.8 Human-in-the-Loop Design

Because automated recognition can fail on inputs outside its distribution, Healthier
provides two interactive controls: a **Segmentation Editor** that lets the user correct
predicted polygons before analysis, and a **Nutrient Value Verification Form** that lets
the user edit OCR-extracted values before the clinical rules are applied. This design
follows the HITL principle that human participation is valuable for handling uncertain
predictions and preserving meaningful human control, as demonstrated by the edit rates
observed in longitudinal food-logging deployments (Bakar et al., 2025). **[Figure 2 —
HITL interface screenshots to be inserted here: run the application and capture the
Segmentation Editor and the verification form.]**

### 2.9 Ablation and Baseline Design

To prove the contribution of each component, the following ablations and comparisons are
specified. They are reported in Section 3.

- **OCR preprocessing ablation.** Each stage is added cumulatively: raw → +deskew →
  +deglare → +CLAHE → +ROI → +semantic filter → +fuzzy matching → full
  pipeline. The grid parser is evaluated with and without column reconstruction. This
  isolates the contribution of layout reconstruction relative to the contribution of
  image preprocessing.
- **Segmentation baseline comparison.** An alternative detection/segmentation
  architecture is trained with the same split and budget as a comparison. If a
  full-budget baseline cannot be completed within the study window, the existing short
  screening run is reported explicitly as a pilot and not as a final comparison.
- **Clinical integration ablation.** The decision layer is evaluated with and without
  the clinical rules, and with and without HITL correction, to quantify the contribution
  of the integration mechanism.

### 2.10 Metrics and Statistical Analysis

Segmentation is evaluated at the instance level (box and mask precision, recall, mAP50,
and mAP50-95) and at the pixel level (per-class accuracy, precision, recall, and F1),
with normalized confusion matrices for the internal validation set and for each external
test set. Per-class average precision is reported to expose source-specific failure
modes. OCR is evaluated with field-level accuracy and exact/wrong/missing rates with a
cluster-bootstrap confidence interval over images. Clinical-rule behaviour is reported
as a deterministic scenario comparison (rule verification). All random seeds are fixed
(seed 42).

---

## 3. RESULTS AND DISCUSSION

This section reports the quantitative evaluation of the two recognition modules, their
behaviour under domain shift, the ablation and baseline comparisons, and the integration
of their outputs through the Clinical Rules Engine, the HITL layer, and the usability
assessment.

### 3.1 Instance Segmentation on Internal Validation

On the 2,017-image internal validation split (best checkpoint, epoch 48), the YOLO11s-seg
model reached a mask mAP50 of **71.66%** and a mask mAP50-95 of **56.91%**, with mask
precision **72.95%** and recall **65.50%**. Box metrics were slightly higher: box mAP50
**72.37%**, box mAP50-95 **60.11%**, box precision 72.87%, and box recall 66.04%
(Table 6). The gap between precision and recall is expected in instance segmentation of
plated food, where overlapping and heterogeneous regions make some instances easier to
localize than to recall fully; this recall gap is one reason the system retains an
interactive correction step.

**[Table 6 — Internal validation metrics. Source:
`experiments/paper/tables/table_segmentation_internal.md`.]**
**[Figure 3 — Training curves. Source: `experiments/runs/seg_main/results.png`.]**
**[Figure 4 — Internal normalized confusion matrix. Source:
`experiments/runs/seg_main/confusion_matrix_normalized.png`.]**
**[Figure 5 — Internal qualitative ground truth vs prediction. Source:
`experiments/runs/seg_main/val_batch0_labels.jpg` and
`experiments/runs/seg_main/val_batch0_pred.jpg`.]**

### 3.2 Generalization to External Sources and Per-Class Error Analysis

Because the training data are not Indonesian-specific, external evaluation is the central
generalization test. The model was applied unchanged to the three held-out sources. Mask
mAP50 was **69.15%** on UECFoodPIXCOMPLETE test and **63.66%** on the FoodSeg103
validation set — a drop of only a few points relative to internal validation — but fell
to **17.06%** on FruitSeg30 (Table 7). Because the training and evaluation pipeline,
classes, and metrics were identical across sources, the drop isolates a **domain shift**
rather than a training error.

**[Table 7 — External test metrics. Source:
`experiments/paper/tables/table_segmentation_external.md`.]**
**[Figure 6 — External normalized confusion matrices. Source:
`experiments/results/segmentation/{uec_test_instance,foodseg_val_instance,fruitseg_instance}/confusion_matrix_normalized.png`.]**
**[Figure 7 — External mask precision–recall and F1 curves. Source: each set's
`MaskPR_curve.png` and `MaskF1_curve.png`.]**

Per-class average precision shows where the model succeeds and fails (Table 8). On
UECFoodPIXCOMPLETE the strongest class was Staple Food (mask AP50 **85.70%**) and the
weakest was Vegetables (**55.42%**); on FoodSeg103 the strongest was Fruits (**74.96%**)
and the weakest was the heterogeneous Others class (**51.24%**). On FruitSeg30 the only
class present is Fruits, which scored **17.29%**. The pixel-level view sharpens the
failure: on FruitSeg30 the Fruits class had pixel precision 97.17% but recall only
**2.14%** (F1 **4.19%**), meaning the few pixels it labelled as fruit were mostly
correct, but it recovered almost none of the fruit and instead assigned fruit regions to
other groups (Table 9).

**[Table 8 — Per-class external instance metrics. Source:
`experiments/paper/tables/table_segmentation_per_class.md`.]**
**[Table 9 — External pixel-level per-class metrics. Source:
`experiments/paper/tables/table_segmentation_pixel.md`.]**
**[Figure 8 — Per-class external mask AP50 bar chart. Build from
`experiments/results/segmentation/per_class_all.csv`.]**

The pattern is interpretable. UEC and FoodSeg103 contain plated meals and kitchen food
photographs that share low-level statistics with the training pool, whereas FruitSeg30
contains isolated fruit in natural, uncontrolled environments (Shamrat et al., 2024) — a
different visual domain that the model was never trained on, especially because the
Fruits class in training comes only from FoodSeg103. We therefore treat the FruitSeg30
result as an explicit limitation and as evidence for the generalization/failure-analysis
contribution, not as a headline strength.

### 3.3 Baseline Comparison

**[TO COMPLETE.]** A short five-epoch screening run was previously performed with
alternative architectures (YOLOv8m-seg, YOLO11m-seg, and YOLO26m-seg) on a limited
subset, but these results are not comparable to the 100-epoch main model and must be
reported as a screening pilot. A full-budget baseline trained on the same data and split
is required to make a fair architectural comparison. Source:
`experiments/paper/baselines_from_notebook.csv`.

### 3.4 OCR Extraction and Preprocessing Ablation

The OCR module was evaluated with field-level accuracy on Indonesian snack-label
photographs with hand-transcribed ground truth. This evaluation is an **exploratory pilot
on six images (46 fields)**; it is reported to document the effect of the layout parser
and is not a generalization result. The baseline parser achieved **21.74%** field-level
accuracy (95% cluster-bootstrap CI [16.67%, 25.58%]), and the revised grid parser
achieved **47.83%** (95% CI [29.17%, 66.67%]) — more than double — with missing fields
falling from 65.22% to 34.78% of the reference fields (Table 10).

**[Table 10 — OCR field-level pilot. Source:
`experiments/paper/tables/table_ocr.md`.]**

**Error analysis.** The dominant failure mode of the baseline was structural: PP-OCR
merges tall numeric boxes across adjacent table rows and columns, so values were assigned
to the wrong row (for example, sugar was read as 3 instead of 10, and 9 instead of 5, on
different images). The revised parser addresses this by detecting value columns relative
to the row labels and by splitting merged numeric boxes, which improved the sodium and
energy fields most. Residual errors concentrate on serving-size formatting and on
distinguishing per-serving from per-100 g values.

**[TO COMPLETE — per-stage preprocessing ablation.]** Table 11 specifies the ablation;
it must be filled with results before submission. **[Figure 9 — OCR pipeline diagram to
be inserted here. Redraw from `docs/architecture.md`.]** **[Figure 10 — OCR qualitative
example: input, ROI crop, and reconstructed table. Example crops: 
`experiments/results/ocr_vlm/gt_audit/134337_kolom_nilai.png`,
`.../134337_lemak_total.png`, `.../132308_gula.png`.]**

**[Table 11 — Planned OCR preprocessing ablation.]**

| Configuration | Field accuracy | Exact | Wrong | Missing |
| :--- | ---: | ---: | ---: | ---: |
| Baseline parser | 21.74% | 21.74% | 13.04% | 65.22% |
| Grid parser (full) | 47.83% | 47.83% | 17.39% | 34.78% |
| Raw OCR only (no preprocessing) | pending | | | |
| + deskew | pending | | | |
| + deglare | pending | | | |
| + CLAHE | pending | | | |
| + ROI | pending | | | |
| + semantic filter | pending | | | |
| + fuzzy matching (full) | pending | | | |

### 3.5 Clinical Rules Engine and Human-in-the-Loop Integration

The Clinical Rules Engine computes a user's TDEE from BMR and activity factor, derives
personal limits, and classifies accumulated intake as SAFE, WARNING, or DANGER using the
thresholds in Table 5. **Rule verification (done):** the implemented engine matched an
independent re-implementation of the specification on **36/36** daily-limit values and
**30/30** status decisions (6 profiles x 5 foods). **End-to-end propagation (done):** on
the 6-product x 6-profile pilot, the ground-truth status was preserved in **18/36**
(50.0%) decisions by the baseline parser and **22/36** (61.1%) by the grid parser; mis-parsed
sugar/sodium produced false DANGER and missing fields produced false SAFE. The HITL layer
allows the user to correct predicted polygons and extracted values before the rules are applied.

### 3.6 Usability Assessment

System usability was assessed with Nielsen's heuristic evaluation using four evaluators
(E1, E2 independent ML engineers/data scientists; E3 lead data scientist, asynchronous
architecture review; E4 self-evaluator, documentation-implementation consistency, excluded
from severity weighting). The evaluation found **no severity-4 findings**; the remaining
findings were of **severity 2-3**. The severity panel had only two evaluators, so this is
reported as an exploratory usability assessment, not an end-user study.

### 3.7 Discussion

The results separate cleanly into what generalizes and what does not. Segmentation is
competitive on in-domain plated-food sources and collapses on a cross-domain, fruit-only
source; the external numbers therefore support a claim of **supervised in-domain
performance with a quantified domain shift**, not of universal generalization. This
matches the practical concern raised by recent food-segmentation benchmarks, which report
limited data diversity and degradation under new viewpoints and domains (AlMughrabi et
al., 2026).

For label reading, the pilot indicates that **layout reconstruction, not raw recognition,
is the main bottleneck** for Indonesian nutrition tables: the same OCR engine gained more
than twenty accuracy points once merged cells and value columns were parsed correctly.
This is consistent with studies showing that dense tabular packaging layouts are a major
source of OCR error (Nagayi et al., 2025) and that preprocessing and layout-aware
correction are necessary in practice (Saputra et al., 2024; Saputra et al., 2026;
Alireza & Kurniawardhani, 2026). The honest framing is that the OCR evidence is currently
a pilot and that the headline number must be re-established on a larger, brand-stratified
set.

At the system level, the contribution is architectural: verified segmentation outputs and
verified label values are converted into transparent, guideline-aligned feedback by a
deterministic rule layer that runs locally, with a user-verification step retained. This
combination — plate recognition and packaged-label extraction feeding one explicit
decision layer, offline — is less commonly reported than improvements to any single
recognition model, and it is the gap this study targets.

**Threats to validity.** (1) The OCR sample is small and not brand-stratified, so no
generalization claim is made. (2) The internal validation represents one seed; a
multi-seed estimate would strengthen the segmentation result. (3) The comparison against
an alternative segmentation architecture is only a screening baseline. (4) No
registered-dietitian review was performed, so the clinical layer is verified
functionally, not clinically. (5) No public Indonesian plate-segmentation benchmark
exists, so Indonesian-specific generalization could not be measured quantitatively.

---

## 4. CONCLUSIONS AND SUGGESTIONS

This research developed and evaluated **Healthier** as an integrated, edge-oriented
nutrition-assistance system that combines YOLO11s-seg plate segmentation, PaddleOCR
(PP-OCRv4) nutrition-table extraction, a deterministic Clinical Rules Engine, and
Human-in-the-Loop verification. On internal validation the segmentation model reached a
mask mAP50 of **71.66%** (mAP50-95 **56.91%**). Across external sources it retained mask
mAP50 of **69.15%** (UECFoodPIXCOMPLETE) and **63.66%** (FoodSeg103) but dropped to
**17.06%** on the cross-domain FruitSeg30 set, a quantified domain shift that the study
reports as a limitation rather than conceals. For nutrition labels, an exploratory
six-image pilot showed that a grid-based layout parser more than doubles field-level
accuracy relative to a baseline parser (**21.74% → 47.83%**), confirming that layout
reconstruction is the key bottleneck. The findings should be read within the study's
scope: the OCR sample is small, portion estimation relies on 2D image information, and
the heterogeneous Others class remains the weakest category.

The deterministic rule layer and the HITL controls provide a transparent mechanism for
converting uncertain model outputs into guideline-aligned feedback that the user can
correct before it is used. The system therefore demonstrates the technical feasibility of
integrating visual food analysis and packaged-food label extraction in a single local
workflow, while making its generalization boundaries explicit.

**Suggestions for future work.** Four directions are recommended. First, **data
coverage**: build an Indonesian plate-segmentation and nutrition-label corpus, including
soups and texturally complex foods, since no public Indonesian plated-meal segmentation
benchmark currently exists. Second, **generalization and estimation**: improve
cross-source robustness and estimate portions more reliably than the current 2D area
ratio, for example with depth or multi-angle imaging. Third, **OCR scale and rigor**:
expand the label evaluation to at least 100 brand-stratified images and complete the
per-stage preprocessing ablation. Fourth, **deployment**: apply quantization and further
edge optimization, and consider an optional offline small language model for explanation
or normalization — but only as a subordinate component that never overrides the
deterministic nutrition rules or the explicit human verification. Prospective studies
with real users and nutrition professionals are needed to validate the clinical
usefulness of the feedback.

---

## 5. REFERENCES

References marked **[folder]** are available in the project's `referensi/` folder.
References marked **[add]** are standard sources that should be added to the folder
(dataset originals, guidelines, and the inpainting method) so that the bibliography is
self-contained.

1. Alireza, R. A., & Kurniawardhani, A. (2026). Development of an OCR system for
   ingredient list text recognition on food labels using CRAFT and CRNN. *MALCOM:
   Indonesian Journal of Machine Learning and Computer Science, 6*(4), 2111–2121.
   https://doi.org/10.57152/malcom.v6i4.2707 **[folder]**
2. AlMughrabi, A., Rivo, G., Jiménez-Farfán, C., Haroon, U., Al-Areqi, F., Jung, H.,
   Busam, B., Marques, R., & Radeva, P. (2026). *BenchSeg: A large-scale dataset and
   benchmark for multi-view food video segmentation*. arXiv:2601.07581. **[folder]**
3. Bakar, L., Englhardt, Z., Srinivas, V., Narayanswamy, G., Nissanka, D., Patel, S., &
   Iyer, V. (2025). *SnappyMeal: Design and longitudinal evaluation of a multimodal AI
   food logging application*. arXiv:2511.03907. **[folder]**
4. Cui, C., Sun, T., Lin, M., Gao, T., Zhang, Y., Liu, J., et al. (2025). *PaddleOCR
   3.0 technical report*. arXiv:2507.05595. **[folder]**
5. Fauzan, A., & Pamungkas, H. (2025). Regional food recognition with nutritional
   analysis: Attention mechanism implementation for Indonesian cuisine. In *Proceedings
   of the 8th International Conference on Informatics, Engineering, Science &
   Technology (INCITEST 2025)* (Advances in Engineering Research, Vol. 287).
   https://doi.org/10.2991/978-94-6463-924-7_16 **[folder]**
6. Heydari, S., & Mahmoud, Q. H. (2025). Tiny machine learning and on-device inference:
   A survey of applications, challenges, and future directions. *Sensors, 25*(10), 3191.
   https://doi.org/10.3390/s25103191 **[folder]**
7. Kahendra, F., Lestari, P. B., Aushof, A. D., Nabila, A., & Gunawan, T. (2026). Label
   Nutri-Level pada kemasan pangan sebagai instrumen kebijakan kesehatan dalam mendorong
   perilaku konsumsi sehat masyarakat. *SerQua: Service Quality (Jurnal Pengabdian
   Masyarakat), 3*(2), 220–225. **[folder]**
8. Kementerian Kesehatan Republik Indonesia. (2024). *Prevalence, impact, and efforts in
   controlling hypertension & diabetes in Indonesia: Analysis of situation and issues*.
   Badan Kebijakan Pembangunan Kesehatan. **[folder]**
9. Khanam, R., & Hussain, M. (2024). *YOLOv11: An overview of the key architectural
   enhancements*. arXiv:2410.17725. **[folder]**
10. Mivtahurrahimah, & Sevtiyani, I. (2025). Factors contributing to rising diabetes
    cases in Indonesia: Insights from the 2023 Indonesian Health Survey. *Jurnal
    Epidemiologi Kesehatan Komunitas, 10*(3).
    https://doi.org/10.14710/jekk.v10i3.28184 **[folder]**
11. Nagayi, M., Khan, A., Frank, T., Swart, R., & Nyirenda, C. (2025). *Evaluating OCR
    performance on food packaging labels in South Africa*. arXiv:2510.03570. **[folder]**
12. Rasyidi, M. A., Mardhiyyah, Y. S., Nasution, Z., & Wijaya, C. H. (2024). Performance
    comparison of state-of-the-art deep learning model architectures in Indonesian food
    image classification. *Bulletin of Electrical Engineering and Informatics, 13*(5),
    3355–3368. https://doi.org/10.11591/eei.v13i5.7996 **[folder]**
13. Rouhafzay, A., Rouhafzay, G., & Jbilou, J. (2025). Image-based food monitoring and
    dietary management for patients living with diabetes: A scoping review of calorie
    counting applications. *Frontiers in Nutrition, 12*, 1501946.
    https://doi.org/10.3389/fnut.2025.1501946 **[folder]**
14. Saputra, T. M. E., Susanto, A., & Carmelita, B. J. (2024). Implementation of
    Tesseract OCR and bounding box for text extraction on food nutrition labels.
    *Building of Informatics, Technology and Science (BITS), 6*(3), 1403–1412.
    https://doi.org/10.47065/bits.v6i3.6107 **[folder]**
15. Saputra, A. N. P., Yazid, A. S., Pramuntadi, A., & Danianti, D. (2026). Implementasi
    OCR dan logika fuzzy untuk pengecekan nutrisi pada label makanan kemasan. *Jurnal
    Profesi Insinyur Universitas Lampung (JPI), 7*(1).
    https://doi.org/10.23960/jpi.v7n1.380 **[folder]**
16. Shamrat, F. M. J. M., Shakil, R., Idris, M. Y. I., Akter, B., & Zhou, X. (2024).
    FruitSeg30_Segmentation dataset & mask annotations: A novel dataset for diverse fruit
    segmentation and classification. *Data in Brief, 56*, 110821.
    https://doi.org/10.1016/j.dib.2024.110821 **[folder]**
17. Shanti, K. M. (2025). Efektivitas label gizi traffic light terhadap pemahaman,
    persepsi, dan sikap konsumen di Indonesia: Systematic literature review. *Jurnal
    Gizi dan Kesehatan (Universitas Sultan Ageng Tirtayasa), 6*(1).
    https://doi.org/10.62870/jgkp.v6i1.31589 **[folder]**
18. Sultana, J., Ahmed, B. M., Masud, M. M., Huq, A. K. O., Ali, M. E., & Naznin, M.
    (2023). A study on food value estimation from images: Taxonomies, datasets, and
    techniques. *IEEE Access, 11*. https://doi.org/10.1109/ACCESS.2023.3274475
    **[folder]**
19. Wang, W., Min, W., Li, T., Dong, X., Li, H., & Jiang, S. (2022). A review on
    vision-based analysis for automatic dietary assessment. *Trends in Food Science &
    Technology, 122*, 1–12. **[folder]** *(arXiv:2108.02947)*
20. Wang, X., Tang, Z., Guo, J., Meng, T., Wang, C., Wang, T., & Jia, W. (2025).
    Empowering edge intelligence: A comprehensive survey on on-device AI models. *ACM
    Computing Surveys*. **[folder]** *(arXiv:2503.06027)*
21. Okamoto, K., & Yanai, K. (2021). UEC-FoodPIX Complete: A large-scale food image
    segmentation dataset. In *ICPR Workshop on Multimedia Assisted Dietary Management
    (MADiMa)*. **[add]**
22. Wu, X., Fu, X., Liu, Y., Lim, E.-P., Hoi, S. C. H., & Sun, Q. (2021). A large-scale
    benchmark for food image segmentation. In *Proceedings of the 29th ACM International
    Conference on Multimedia* (pp. 506–515). https://doi.org/10.1145/3474085.3475201
    **[add]**
23. Telea, A. (2004). An image inpainting technique based on the fast marching method.
    *Journal of Graphics Tools, 9*(1), 23–34. **[add]**
24. Ultralytics. (2024). *YOLO11* [Computer software].
    https://github.com/ultralytics/ultralytics **[add]**
25. Perkumpulan Endokrinologi Indonesia (PERKENI). (2024). *Pedoman pengelolaan dan
    pencegahan diabetes melitus tipe 2 di Indonesia 2024*. PB PERKENI. **[add]**
26. James, P. A., Oparil, S., Carter, B. L., et al. (2014). 2014 evidence-based
    guideline for the management of high blood pressure in adults: Report from the panel
    members appointed to the Eighth Joint National Committee (JNC 8). *JAMA, 311*(5),
    507–520. https://doi.org/10.1001/jama.2013.284427 **[add]**

---

## APPENDIX A. Reviewer-comment compliance checklist

| Reviewer request | Where addressed | Status |
| :--- | :--- | :--- |
| 2–3 explicit contributions; novelty in integration mechanism, validation, or decision strategy (not just technology combination) | Section 1.2 (three contributions, each with experimental evidence); Section 1.1 (research gap) | Done; evidence pointers given |
| Reproducible methodology: parameters, equations, flow diagram, ablation | Sections 2.1–2.10 (environment, parameter tables, Equations 1–8, Figure 1, ablation design in 2.9) | Done; ablation results pending |
| Dataset consistency: counts, class distribution, annotation, split strategy; independent test data; expand OCR dataset | Sections 2.2 and 2.6 (Tables 1–2; dedup dHash; 85:15 split; three independent external tests; ≥100-image brand-stratified OCR protocol) | Protocol done; OCR expansion pending |
| Confusion matrix + per-class error; baseline comparison; OCR preprocessing ablation; cross-source generalization | Sections 3.2 (Tables 7–9, Figures 6–8), 3.3 (baseline), 3.4 (ablation Table 11), 3.2 (generalization) | Confusion/generalization done; baseline and ablation pending |

## APPENDIX B. Figure and table source index

| Item | Content | Source path |
| :--- | :--- | :--- |
| Fig. 1 | System architecture / flow | redraw from `docs/architecture.md` |
| Fig. 2 | HITL interface screenshots | run the app (`frontend/`) |
| Fig. 3 | Training curves | `experiments/runs/seg_main/results.png` |
| Fig. 4 | Internal confusion matrix | `experiments/runs/seg_main/confusion_matrix_normalized.png` |
| Fig. 5 | Internal GT vs prediction | `experiments/runs/seg_main/val_batch0_labels.jpg` + `val_batch0_pred.jpg` |
| Fig. 6 | External confusion matrices | `experiments/results/segmentation/{uec_test_instance,foodseg_val_instance,fruitseg_instance}/confusion_matrix_normalized.png` |
| Fig. 7 | External PR / F1 curves | same folders, `MaskPR_curve.png` / `MaskF1_curve.png` |
| Fig. 8 | Per-class external AP50 | `experiments/results/segmentation/per_class_all.csv` |
| Fig. 9 | OCR pipeline diagram | redraw from `docs/architecture.md` |
| Fig. 10 | OCR qualitative example | `experiments/results/ocr_vlm/gt_audit/*.png` |
| Table 1 | Dataset and splits | `experiments/paper/tables/table_dataset.md` |
| Table 2 | Per-class instance distribution | `experiments/paper/tables/table_dataset.md` |
| Table 3 | Training configuration | `experiments/paper/final_numbers.md` |
| Table 4 | Daily clinical limits | `frontend/js/clinical-rules.js` |
| Table 5 | Risk thresholds | `frontend/js/clinical-rules.js` |
| Table 6 | Internal metrics | `experiments/paper/tables/table_segmentation_internal.md` |
| Table 7 | External metrics | `experiments/paper/tables/table_segmentation_external.md` |
| Table 8 | Per-class external metrics | `experiments/paper/tables/table_segmentation_per_class.md` |
| Table 9 | External pixel-level metrics | `experiments/paper/tables/table_segmentation_pixel.md` |
| Table 10 | OCR pilot | `experiments/paper/tables/table_ocr.md` |
| Table 11 | OCR ablation (planned) | build with `ablate_ocr.py` |
