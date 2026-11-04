# Application of Deep Learning for Nutritional Segmentation and Information Extraction in the Healthier Application

Tata Aditya Pamungkas^1, Nuke Lu'lu Ul Chusna^2, Harry Dwiyana Kartika^3

Informatics Engineering Study Program, Universitas Krisnadwipayana, Jakarta, Indonesia

Email: tataaditaya@gmail.com; nukelchusna@unkris.ac.id; harry_d@unkris.ac.id

## Abstract

Low nutritional literacy and the rising burden of non-communicable diseases (NCDs) motivate **Healthier**, an offline application that combines plate-image food recognition with nutrition-label extraction. The system integrates three components: a YOLO11s-seg instance-segmentation model that partitions a plate into five *Isi Piringku* groups; a PaddleOCR (PP-OCRv4) pipeline with a grid-based layout parser for nutrition tables; and a deterministic Clinical Rules Engine that converts verified values into SAFE/WARNING/DANGER feedback, with a Human-in-the-Loop correction step. The segmentation model was trained on 11,684 images and validated on 2,017 images from public datasets that are not specific to Indonesian cuisine. Generalisation was measured on three independent external test sets (5,013 images) never used for training or model selection. On internal validation the model reached a mask mAP50 of 71.66% and a mask mAP50-95 of 56.91%. It transferred well to two food-photography sources (69.15% on UECFoodPIXCOMPLETE and 63.66% on FoodSeg103) but degraded sharply on a fruit-only, in-the-wild dataset (17.06%), a measured domain shift. Label extraction was evaluated with field-level accuracy on Indonesian snack labels; a six-image pilot showed that a grid-based parser more than doubled accuracy over a baseline (21.74% to 47.83%), reported as exploratory pending a larger brand-stratified set. The deterministic rule layer matched its specification on all 66 checks, while extraction error changed the clinical decision in 39–50% of pilot cases, motivating the retained human verification. The main contribution is a reproducible integration and decision mechanism, not a single model, supported by cross-source generalisation, per-class error analysis, and an open evaluation protocol.

Keywords: Instance Segmentation; Nutrition Label OCR; YOLO11s-seg; PaddleOCR; Clinical Decision Rules

## Abstrak

Literasi gizi yang rendah dan beban penyakit tidak menular (PTM) yang meningkat melatarbelakangi **Healthier**, aplikasi luring yang menggabungkan pengenalan makanan dari citra piring dengan ekstraksi label gizi dari kemasan. Sistem ini mengintegrasikan tiga komponen: model segmentasi instan YOLO11s-seg yang membagi citra piring menjadi lima kelompok *Isi Piringku*; pipeline PaddleOCR (PP-OCRv4) dengan parser tata letak berbasis grid untuk tabel gizi; dan Clinical Rules Engine deterministik yang mengubah nilai terverifikasi menjadi status SAFE/WARNING/DANGER, dengan tahap koreksi Human-in-the-Loop. Model segmentasi dilatih pada 11.684 citra dan divalidasi pada 2.017 citra dari dataset publik yang tidak spesifik masakan Indonesia. Generalisasi diukur pada tiga set uji eksternal independen (5.013 citra) yang tidak pernah dipakai untuk pelatihan maupun pemilihan model. Pada validasi internal, model mencapai mask mAP50 71,66% dan mask mAP50-95 56,91%. Model bertransfer baik ke dua sumber fotografi makanan (mask mAP50 69,15% pada UECFoodPIXCOMPLETE dan 63,66% pada FoodSeg103) tetapi turun tajam pada dataset khusus buah di lingkungan alami (17,06%), suatu *domain shift* terukur. Ekstraksi label dievaluasi dengan akurasi tingkat field pada label kemasan Indonesia; pilot enam citra menunjukkan parser berbasis grid lebih dari menggandakan akurasi dibandingkan parser dasar (21,74% menjadi 47,83%) sehingga dilaporkan sebagai pilot eksploratif. Lapisan aturan deterministik cocok dengan spesifikasinya pada seluruh 66 pemeriksaan, sementara galat ekstraksi mengubah keputusan klinis pada 39–50% kasus pilot, yang mendasari dipertahankannya verifikasi manusia. Kontribusi utama bukan pada satu model tunggal, melainkan pada mekanisme integrasi dan pengambilan keputusan yang reproducible, didukung generalisasi antar-sumber, analisis kesalahan per kelas, dan protokol evaluasi yang terbuka.

Kata kunci: Segmentasi Instan; OCR Label Gizi; YOLO11s-seg; PaddleOCR; Aturan Keputusan Klinis

## INTRODUCTION

Non-communicable diseases (NCDs) are strongly associated with dietary patterns, and diabetes and hypertension remain among the largest contributors to the disease burden in Indonesia. The 2023 Indonesian Health Survey (SKI) reported a diabetes prevalence of 11.7%, an increase from 10.9% in the 2018 Basic Health Research (Riskesdas), while hypertension prevalence decreased from 34.1% to 30.8% over the same period (Kementerian Kesehatan Republik Indonesia, 2024). Analyses of the 2023 SKI further link the increase in diabetes cases to modifiable lifestyle and metabolic factors such as central obesity and infrequent blood-pressure screening (Mivtahurrahimah & Sevtiyani, 2025). Because a substantial share of diagnosed individuals do not maintain regular treatment or follow-up (Kementerian Kesehatan Republik Indonesia, 2024), practical tools that support everyday self-management are needed alongside clinical care.

At the individual level, the ability to interpret nutrition information is limited. A systematic review of traffic-light nutrition labelling in Indonesia reported that consumer understanding of nutrition labels remains low, even though attitudes toward simplified label formats are generally positive (Shanti, 2025). New front-of-pack initiatives such as the Nutri-Level label have been introduced with the same objective of helping consumers choose healthier products (Kahendra et al., 2026). In practice, however, conventional nutrition tables are difficult to process automatically because the text can be small, layouts vary across products, and reflections or curved packaging degrade image quality (Nagayi et al., 2025). For meals served on a plate there is no printed table at all, so dietary assessment must begin from visual food recognition.

Image-based dietary assessment is usually described as a pipeline of food image analysis, volume or portion estimation, and nutrient derivation (Wang et al., 2022). Reviews of the field note that segmentation, recognition, and nutrient estimation are commonly developed as separate tasks, and that dataset diversity, portion estimation, and real-world image conditions remain persistent challenges (Rouhafzay et al., 2025; Sultana et al., 2023). Indonesian contributions remain scarce: most existing work focuses on classification of local dishes rather than instance segmentation of plated meals (Fauzan & Pamungkas, 2025; Rasyidi et al., 2024). For example, Rasyidi et al. (2024) introduced a classification dataset of 24,427 images covering 160 Indonesian food types, but to the best of our knowledge there is still no public instance-segmentation benchmark for Indonesian plated meals. Generalisation across sources is a known concern as well: recent food-video segmentation benchmarks show that current methods suffer from limited data diversity and degrade under new viewpoints and domains (AlMughrabi et al., 2026). Domain shift must therefore be measured explicitly rather than assumed away.

The label-reading problem presents a parallel challenge. A benchmark on real-world food packaging compared Tesseract, EasyOCR, PaddleOCR, and TrOCR, and found that dense tabular structures such as nutrition-facts panels are a major source of recognition errors (Nagayi et al., 2025). For this reason, preprocessing, layout reconstruction, and domain-specific correction are commonly required in Indonesian settings (Saputra et al., 2024; Saputra et al., 2026; Alireza & Kurniawardhani, 2026). PaddleOCR provides an Apache-licensed OCR toolkit that includes the PP-OCR model family and is practical for local deployment (Cui et al., 2025).

Automated recognition still makes mistakes, so human verification matters. In a longitudinal food-logging study, image recognition still required a 20.7% log edit rate, showing that users remain an essential part of the loop (Bakar et al., 2025). This motivates the Human-in-the-Loop (HITL) design adopted in Healthier, where users can correct segmentation and extracted values before they are used by the decision layer. Local inference is also attractive for health applications because on-device processing can reduce privacy exposure and dependence on continuous cloud connectivity (Heydari & Mahmoud, 2025; Wang et al., 2025).

The literature provides strong evidence for individual components, but relatively few studies integrate plate-based food analysis, packaged-label extraction, and deterministic guideline-based decision-making into a single offline workflow with explicit human verification. Healthier addresses this integration gap through three research questions: (RQ1) how well a segmentation model trained on non-Indonesian food datasets generalises to new and unseen sources, and which classes fail; (RQ2) to what extent a layout-aware OCR pipeline improves nutrition-table extraction over a baseline parser, and what error modes remain; and (RQ3) how segmentation and label outputs can be translated into transparent, guideline-aligned feedback with explicit user verification.

This study makes three explicit contributions, each stated together with how it is demonstrated experimentally, so that the novelty rests on a verifiable integration and decision mechanism rather than on merely combining existing technologies. First, a leakage-controlled evaluation protocol for cross-source food segmentation: we define a data protocol with perceptual-hash deduplication, a fixed validation split, and three independent external test sets that are never used for training or model selection, and we report per-class generalisation across sources (evidenced by the internal and external results and the per-class error analysis in the Results and Discussion). Second, a reproducible, layout-aware OCR pipeline for nutrition tables: we combine four preprocessing stages with a grid-based table parser and fuzzy field mapping, and evaluate it at the field level with image-level cluster-bootstrap confidence intervals, together with a per-stage ablation design. Third, a deterministic, guideline-grounded decision layer with HITL verification: we translate verified segmentation and label values into SAFE/WARNING/DANGER feedback using explicit thresholds derived from PERKENI 2024 and hypertension guidance, while retaining an explicit user-correction step. The novelty is therefore in the integration mechanism, its validation protocol, and its transparent decision strategy, all of which are specified and testable, rather than in a claim that any single model is state of the art.

## RESEARCH METHODS

This research adopts an iterative experimental methodology informed by CRISP-ML(Q), with separate evaluation of the segmentation, OCR, clinical-rule, and usability components. The design emphasises repeated validation and strict separation of data because variability in food images, datasets, ground truth, and reporting practices is a major barrier to reliable comparison between dietary-assessment systems (Sultana et al., 2023; Wang et al., 2022).

### Types of Research

This is applied, quantitative, and experimental research. It develops an engineering artefact (the Healthier pipeline) and evaluates it with objective metrics rather than subjective judgement. The study is descriptive for the system design and comparative for the component evaluation; the design compares each component against a baseline or an alternative configuration. The experimental design is a controlled component-wise evaluation: the data split is fixed once and reused by every experiment, and each module (segmentation, OCR, clinical rules) is evaluated independently before being integrated.

### Time and Place of Research

The research was carried out between June 2026 and November 2026 at the Informatics Engineering Study Program, Universitas Krisnadwipayana, Jakarta, Indonesia. Model training used a cloud Kaggle Tesla T4 GPU environment, while OCR experiments and interface checks ran locally on a laptop with an NVIDIA GeForce RTX 3050 Laptop GPU (6 GB) and a CPU PyTorch build, matching the intended offline deployment condition.

### Research Target / Subject

The study targets two families of objects and one target user profile. The segmentation target consists of plated-food images; the evaluation subjects are three public image datasets (UECFoodPIXCOMPLETE, FoodSeg103, and FruitSeg30) whose original annotations were reused. The OCR target consists of photographs of Indonesian packaged-snack nutrition labels; the evaluation subject was a pilot set of six labels with hand-transcribed ground truth. The clinical target is everyday users with diet-related NCD risk profiles (type-2 diabetes, hypertension, dyslipidaemia, and diabetic nephropathy). Sampling for the public datasets followed the official splits; label photographs were a convenience sample of packaged snacks available in local stores.

### Procedure

The end-to-end procedure has two input paths that share one verification layer and one decision layer. A plate photograph is segmented into food instances and grouped into five *Isi Piringku* groups (Staple Food, Side Dishes, Vegetables, Fruits, Others); a packaging photograph is preprocessed, recognised, and parsed into nutrition fields. Both outputs pass through a HITL verification layer and are then converted by the deterministic Clinical Rules Engine into SAFE/WARNING/DANGER feedback. Figure 1 summarises the workflow.

[[INSERT FIGURE 1 HERE: system architecture and experimental flow, redrawn from the project architecture diagram]]

Figure 1. Healthier system architecture and experimental workflow.

#### Dataset Construction, Annotation, and Splits

Three public datasets were used: UECFoodPIXCOMPLETE (Okamoto & Yanai, 2021), FoodSeg103 (Wu et al., 2021), and FruitSeg30 (Shamrat et al., 2024). FruitSeg30 contains 1,969 images in 30 fruit classes collected in natural and uncontrolled environments across Malaysia, Bangladesh, and Australia (Shamrat et al., 2024). The candidate corpus contained 19,087 images. Duplicate images were detected with a difference hash (dHash) computed on an 8×8 grayscale reduction of each image, and 373 duplicates were removed, leaving 18,714 unique images. Class labels from all sources were remapped into the five *Isi Piringku* groups using a fixed mapping table.

We did not re-annotate the public masks; the original annotations were used as provided, and only the label remapping was applied. A stratified audit of the remapping (200 sampled images, 40 per class, by two annotators, reporting Cohen's κ and resolving disagreements by discussion) is recommended to document this step; it is not reported here and remains future work.

The training pool consisted of the official training portions of UEC and FoodSeg103 and was split 85:15 into training (11,684) and validation (2,017) sets, stratified by class with seed 42 and no overlap. Validation images are therefore a held-out 15% of the training pool. Three sets were reserved exclusively for external testing and were never used for training, early stopping, or checkpoint selection: the official UEC test set (948 images), the FoodSeg103 validation set (2,096 images), and the whole of FruitSeg30 (1,969 images), for 5,013 external images in total. Because the Fruits class in the training data comes only from FoodSeg103, FruitSeg30 acts as a deliberate domain-shift test. No oversampling and no class-weighted loss were applied, so the external results are not confounded by resampling. Table 1 reports the split composition and Table 2 the per-class instance distribution.

Table 1. Dataset and split composition (images).

| Split | Source | Images |
| :--- | :--- | ---: |
| Train | UECFoodPIXCOMPLETE | 7,478 |
| Train | FoodSeg103 | 4,206 |
| Validation | UECFoodPIXCOMPLETE | 1,285 |
| Validation | FoodSeg103 | 732 |
| External test | FoodSeg103 validation | 2,096 |
| External test | UECFoodPIXCOMPLETE test | 948 |
| External test | FruitSeg30 | 1,969 |
| Total | — | 18,714 |

Table 2. Per-class instance distribution per split.

| Split | Staple | Side | Vegetables | Fruits | Others |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Train (UECFoodPIXCOMPLETE) | 4,188 | 3,614 | 1,466 | 0 | 4,996 |
| Train (FoodSeg103) | 3,369 | 3,685 | 9,223 | 3,513 | 4,103 |
| Validation (UECFoodPIXCOMPLETE) | 749 | 608 | 257 | 0 | 927 |
| Validation (FoodSeg103) | 585 | 626 | 1,619 | 618 | 638 |
| External: FoodSeg103 validation | 1,629 | 1,751 | 4,718 | 1,649 | 2,004 |
| External: UEC test | 543 | 463 | 224 | 0 | 754 |
| External: FruitSeg30 | 0 | 0 | 0 | 1,993 | 0 |

#### Segmentation Model and Training

The model is YOLO11s-seg (Khanam & Hussain, 2024). It was trained with Ultralytics 8.4.199 for at most 100 epochs with early-stopping patience 20; training stopped at epoch 67 and the best checkpoint was epoch 48. Optimisation used AdamW with an initial learning rate of 0.001, a final learning-rate factor of 0.01, weight decay 0.0005, and momentum 0.9, with batch size 16, image size 640, and seed 42 (deterministic). Spatial and photometric augmentation was applied (HSV jitter, translation, scaling, horizontal flipping, mosaic, and random erasing). Training took 4.01 h. Table 3 lists the full configuration.

Table 3. Segmentation training configuration.

| Parameter | Value |
| :--- | :--- |
| Model | YOLO11s-seg |
| Framework | Ultralytics 8.4.199 |
| Torch / Python | 2.10.0+cu128 / 3.12.13 |
| Max epochs / patience | 100 / 20 |
| Stopped epoch / best epoch | 67 / 48 |
| Optimiser | AdamW |
| Initial learning rate | 0.001 |
| Final LR factor | 0.01 |
| Weight decay | 0.0005 |
| Momentum | 0.9 |
| Batch size | 16 |
| Image size | 640 |
| Seed | 42 |
| Augmentation | HSV, translate, scale, flip, mosaic, erasing |
| Training time | 4.01 h (Tesla T4) |

#### Plate Proportion Estimation

For each plate image, the model outputs one segmentation polygon per detected food instance. Instances are assigned to their *Isi Piringku* group, and the nutritional proportion of a group is computed as the ratio of the sum of its food-polygon areas to the total analysed food area on the plate (Equation 1).

EQ: Proportion_g = A_g / Σ_g A_g          (1)

where A_g is the total polygon area of group g. Detections of non-food objects (cutlery, the plate itself, hands) are filtered out before the proportion is computed so that the proportions are based on food pixels only.

#### Nutrition-Label Extraction Pipeline

Packaging images pass through four preprocessing stages before recognition: deskewing using the Hough transform, deglare using Telea inpainting (Telea, 2004), CLAHE-based contrast normalisation, and region-of-interest cropping around the nutrition table. The processed region is recognised with PaddleOCR (Cui et al., 2025) configured with the PP-OCRv4 recognition model.

Layout reconstruction then (i) groups text boxes into logical rows using Y-band grouping; (ii) detects the value columns (Per Sajian / Serving and %AKG / %DV) relative to the row-label column; and (iii) splits tall numeric boxes that PP-OCR merges across adjacent rows (for example, 1725 is split into 1, 7, 2, 5), thereby separating the per-serving and %AKG columns. A semantic filter removes non-nutritional text (promotional slogans, BPOM registration numbers, ingredient lists, and cooking instructions). Finally, fuzzy matching based on edit distance maps OCR tokens to standardised nutrition terms; the Levenshtein distance and similarity are given in Equations 2 to 4.

EQ: lev(a, b) = max(|a|, |b|), if min(|a|, |b|) = 0          (2)

EQ: lev(a, b) = min( lev(a−1, b) + 1, lev(a, b−1) + 1, lev(a−1, b−1) + [a_i ≠ b_j] )          (3)

EQ: Similarity(a, b) = 1 − lev(a, b) / max(|a|, |b|)          (4)

A token is accepted as a match when Similarity ≥ 0.75.

#### Clinical Rules Engine

Nutrition values that have passed HITL verification are evaluated by a deterministic rules engine that runs locally in the browser (no cloud inference). The engine computes Basal Metabolic Rate (BMR) with the Mifflin–St Jeor equations and Total Daily Energy Expenditure (TDEE) as BMR multiplied by an activity factor (Equations 5 to 7).

EQ: BMR_male = 10W + 6.25H − 5A + 5          (5)

EQ: BMR_female = 10W + 6.25H − 5A − 161          (6)

EQ: TDEE = BMR × AF          (7)

where W is body weight (kg), H is height (cm), A is age (years), and AF is the activity factor (1.2, the sedentary value used by the application). Personal limits are then derived per profile (Table 4). For type-2 diabetes, the daily sugar limit follows PERKENI 2024 and is set dynamically at 5% of TDEE (0.05 × TDEE / 4 g); for non-diabetic profiles the default is 10% of TDEE (0.10 × TDEE / 4 g). Sodium limits are absolute values: 1,500 mg/day for hypertension and 2,300 mg/day otherwise. Saturated-fat limits are 10% of TDEE (0.10 × TDEE / 9 g), reduced to 7% (0.07 × TDEE / 9 g) for diabetes with dyslipidaemia; cholesterol is limited to 200 mg for diabetes with dyslipidaemia; and the protein limit for diabetic nephropathy is 0.8 g × body weight (PERKENI, 2024; James et al., 2014). All limits are implemented exactly as listed in Table 4.

Table 4. Daily nutrient limits implemented in the Clinical Rules Engine.

| Limit | Value | Applies to |
| :--- | :--- | :--- |
| Snack calorie budget | 0.20 × TDEE | All users |
| Sugar | 0.10 × TDEE / 4 g | Default |
| Sugar | 0.05 × TDEE / 4 g | Type-2 diabetes |
| Sodium | 2,300 mg/day | Default |
| Sodium | 1,500 mg/day | Hypertension |
| Sodium-density warning | 800 mg/100 g | Default |
| Sodium-density warning | 600 mg/100 g | Hypertension |
| Saturated fat | 0.10 × TDEE / 9 g | Default |
| Saturated fat | 0.07 × TDEE / 9 g | Diabetes with dyslipidaemia |
| Cholesterol | 200 mg/day | Diabetes with dyslipidaemia |
| Protein | 0.8 g × body weight | Diabetic nephropathy |
| Fibre target | 25–38 g/day | Default |
| Fibre target | 20–35 g/day | Type-2 diabetes |
| Carbohydrate target | 45–65% of energy | Type-2 diabetes |

The engine classifies accumulated intake into SAFE, WARNING, and DANGER (Table 5). A WARNING is raised when sugar or saturated fat exceeds 50% of the daily limit, when actual sodium exceeds 40% of the sodium limit, when protein exceeds 30% of the nephropathy quota, or when a product's sodium density exceeds 600 mg/100 g for hypertension (800 mg/100 g otherwise). DANGER is raised only when sugar exceeds 100% of the daily limit; sodium and saturated fat saturate at WARNING. These bands are the exact thresholds implemented in the application and are reported as rule verification, not clinical validation, because no independent registered-dietitian review was performed.

Table 5. Risk classification thresholds.

| Status | Condition |
| :--- | :--- |
| SAFE | All nutrients below the warning thresholds |
| WARNING | Sugar or saturated fat > 50% of limit; sodium > 40% of limit; protein > 30% of nephropathy quota; sodium density > 600 mg/100 g (hypertension) or > 800 mg/100 g (other) |
| DANGER | Sugar > 100% of the daily limit |

#### Human-in-the-Loop Design

Because automated recognition can fail on inputs outside its distribution, Healthier provides two interactive controls: a Segmentation Editor that lets the user correct predicted polygons before analysis, and a Nutrient Value Verification Form that lets the user edit OCR-extracted values before the clinical rules are applied. This design follows the HITL principle that human participation is valuable for handling uncertain predictions and preserving meaningful human control, as demonstrated by the edit rates observed in longitudinal food-logging deployments (Bakar et al., 2025). Figure 2 shows the two interfaces.

[[INSERT FIGURE 2 HERE: screenshots of the Segmentation Editor and the Nutrient Value Verification Form]]

Figure 2. Human-in-the-Loop interfaces: segmentation editor and nutrient value verification form.

#### Ablation and Baseline Design

To prove the contribution of each component, the following ablations and comparisons were specified. For OCR preprocessing, each stage is added cumulatively (raw → +deskew → +deglare → +CLAHE → +ROI → +semantic filter → +fuzzy matching → full pipeline), and the grid parser is evaluated with and without column reconstruction, to isolate the contribution of layout reconstruction relative to image preprocessing. For segmentation, an alternative segmentation architecture is trained with the same split and budget as a comparison; if a full-budget run cannot be completed within the study window, the short screening run is reported explicitly as a pilot and not as a final comparison. For clinical integration, the decision layer is evaluated with and without the clinical rules, and with and without HITL correction, to quantify the contribution of the integration mechanism.

### Data, Instruments, and Data Collection Techniques

The data are secondary for segmentation (public datasets with their original annotations) and primary for label extraction (photographs taken by the author of Indonesian packaged snacks, together with hand-transcribed ground-truth values). The segmentation data separation was performed with a difference-hash script; the ground truth for label extraction was transcribed manually from the same package. The instruments were Python 3 with Ultralytics and PaddleOCR for modelling, and a desktop web application (HTML, CSS, and JavaScript) for the HITL interfaces and the clinical rule engine. Images for label extraction were collected from local stores so that the pilot covered different brands.

### Data Analysis Technique

Segmentation is evaluated at the instance level (box and mask precision, recall, mAP50, and mAP50-95) and at the pixel level (per-class accuracy, precision, recall, and F1), with normalised confusion matrices for the internal validation set and for each external test set. Per-class average precision is reported to expose source-specific failure modes. Label extraction is evaluated at the field level because the downstream system consumes specific nutrition fields rather than full transcriptions; a field is counted as correct when it is exact or within 5% of the ground-truth per-serving value (Equation 8).

EQ: FieldAccuracy = (N_exact + N_within5%) / N_fields          (8)

Exact, wrong, and missing fields are reported separately, and the confidence interval is obtained by a cluster bootstrap over images (2,000 iterations, seed 42) rather than a sample-size-based interval. The evaluated fields are energy, total fat, saturated fat, trans fat, cholesterol, sodium, carbohydrate, fibre, sugar, protein, and serving size; values are taken per serving, and fields absent from a package are recorded as empty rather than zero. Clinical-rule behaviour is evaluated by rule verification: because the engine is deterministic, the implemented engine is compared against an independent re-implementation of the documented specification (Tables 4 and 5) over six clinical profiles and five food scenarios, so that any divergence between code and specification is detected exactly rather than estimated. The end-to-end effect of extraction error is measured by applying the same rules to the hand-transcribed ground-truth values and to the OCR-extracted values for the same products, and reporting how often the SAFE/WARNING/DANGER status changes. Usability is assessed with Nielsen's heuristic evaluation using four evaluators: two independent machine-learning engineers/data scientists (E1, E2), one lead data scientist performing an asynchronous architecture review (E3), and one self-evaluator (E4) checking documentation–implementation consistency; severity weighting is based on the independent E1–E2 assessments, and E4 is reported separately. All random seeds are fixed (seed 42).

## RESULTS AND DISCUSSION

This section reports the quantitative evaluation of the two recognition modules and their behaviour under domain shift, together with the component ablation, baseline, and integration analyses where results are available; the analyses that remain to be completed are marked explicitly.

### Instance Segmentation on Internal Validation

On the 2,017-image internal validation split (best checkpoint, epoch 48), the YOLO11s-seg model reached a mask mAP50 of **71.66%** and a mask mAP50-95 of **56.91%**, with mask precision **72.95%** and recall **65.50%**. Box metrics were slightly higher: box mAP50 **72.37%**, box mAP50-95 **60.11%**, box precision 72.87%, and box recall 66.04% (Table 6). The gap between precision and recall is expected in instance segmentation of plated food, where overlapping and heterogeneous regions make some instances easier to localise than to recall fully; this recall gap is one reason the system retains an interactive correction step. Figures 3 to 5 show the training curves, the normalised confusion matrix, and qualitative ground truth versus prediction.

Table 6. Internal validation metrics at the best checkpoint.

| Metric | Box | Mask |
| :--- | ---: | ---: |
| Precision | 72.87 | 72.95 |
| Recall | 66.04 | 65.50 |
| mAP50 | 72.37 | 71.66 |
| mAP50-95 | 60.11 | 56.91 |

[[INSERT FIGURE 3 HERE: training curves]]

Figure 3. Training and validation loss curves for the YOLO11s-seg model.

[[INSERT FIGURE 4 HERE: internal normalised confusion matrix]]

Figure 4. Normalised confusion matrix on the internal validation split.

[[INSERT FIGURE 5 HERE: internal ground truth versus prediction]]

Figure 5. Qualitative comparison of ground truth (left) and prediction (right) on the internal validation split.

### Generalisation to External Sources and Per-Class Error Analysis

Because the training data are not Indonesian-specific, external evaluation is the central generalisation test. The model was applied unchanged to the three held-out sources. Mask mAP50 was **69.15%** on UECFoodPIXCOMPLETE test and **63.66%** on the FoodSeg103 validation set — a drop of only a few points relative to internal validation — but fell to **17.06%** on FruitSeg30 (Table 7). Because the training and evaluation pipeline, classes, and metrics were identical across sources, the drop isolates a domain shift rather than a training error. Figures 6 and 7 show the external confusion matrices and the precision–recall and F1 curves.

Table 7. External-test instance-level metrics.

| Dataset | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FoodSeg103 validation | 66.97 | 59.36 | 64.57 | 52.08 | 67.47 | 58.37 | 63.66 | 48.16 |
| UECFoodPIXCOMPLETE test | 67.82 | 68.08 | 70.30 | 57.86 | 67.42 | 67.44 | 69.15 | 54.30 |
| FruitSeg30 | 17.28 | 44.96 | 17.36 | 12.85 | 17.01 | 44.25 | 17.06 | 13.55 |

[[INSERT FIGURE 6 HERE: external normalised confusion matrices for FoodSeg103, UEC, and FruitSeg30]]

Figure 6. Normalised confusion matrices on the three external test sources.

[[INSERT FIGURE 7 HERE: external mask precision–recall and F1 curves]]

Figure 7. Mask precision–recall and F1 curves for each external source.

Per-class average precision shows where the model succeeds and fails (Table 8). On UECFoodPIXCOMPLETE the strongest class was Staple Food (mask AP50 **85.70%**) and the weakest was Vegetables (**55.42%**); on FoodSeg103 the strongest was Fruits (**74.96%**) and the weakest was the heterogeneous Others class (**51.24%**). On FruitSeg30 the only class present is Fruits, which scored **17.29%**. The pixel-level view sharpens the failure: on FruitSeg30 the Fruits class had pixel precision 97.17% but recall only **2.14%** (F1 **4.19%**), meaning the few pixels it labelled as fruit were mostly correct, but it recovered almost none of the fruit and instead assigned fruit regions to other groups (Table 9, Figure 8).

Table 8. Per-class instance-level metrics on the external sources.

| Dataset | Class | GT inst. | Mask AP50 | Mask AP50-95 | Box AP50 | Box AP50-95 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| UEC test | Staple Food | 543 | 85.70 | 73.28 | 87.51 | 77.34 |
| UEC test | Side Dishes | 463 | 70.71 | 53.46 | 71.48 | 58.82 |
| UEC test | Vegetables | 224 | 55.42 | 40.59 | 55.54 | 44.07 |
| UEC test | Others | 754 | 65.17 | 50.57 | 65.90 | 52.24 |
| FoodSeg103 val | Staple Food | 1,629 | 63.87 | 49.92 | 65.55 | 54.95 |
| FoodSeg103 val | Side Dishes | 1,751 | 61.61 | 47.21 | 64.35 | 52.22 |
| FoodSeg103 val | Vegetables | 4,718 | 65.83 | 48.27 | 66.85 | 52.74 |
| FoodSeg103 val | Fruits | 1,649 | 74.96 | 56.59 | 72.59 | 58.29 |
| FoodSeg103 val | Others | 2,004 | 51.24 | 39.15 | 52.95 | 43.19 |
| FruitSeg30 | Fruits | 1,993 | 17.29 | 13.90 | 17.68 | 13.30 |

Table 9. External-test pixel-level per-class metrics.

| Dataset | Class | Accuracy | Precision | Recall | F1 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| FoodSeg103 val | Staple Food | 92.93 | 72.71 | 66.95 | 69.71 |
| FoodSeg103 val | Side Dishes | 93.49 | 75.12 | 77.90 | 76.49 |
| FoodSeg103 val | Vegetables | 93.56 | 75.41 | 74.46 | 74.93 |
| FoodSeg103 val | Fruits | 98.03 | 70.91 | 66.71 | 68.75 |
| FoodSeg103 val | Others | 94.55 | 55.88 | 55.63 | 55.75 |
| UEC test | Staple Food | 93.31 | 81.63 | 81.95 | 81.79 |
| UEC test | Side Dishes | 94.31 | 72.71 | 73.44 | 73.07 |
| UEC test | Vegetables | 96.04 | 64.33 | 61.51 | 62.89 |
| UEC test | Others | 94.37 | 61.68 | 61.36 | 61.52 |
| FruitSeg30 | Fruits | 61.28 | 97.17 | 2.14 | 4.19 |

[[INSERT FIGURE 8 HERE: bar chart of per-class external mask AP50]]

Figure 8. Per-class external mask AP50 by source and class.

The pattern is interpretable. UEC and FoodSeg103 contain plated meals and kitchen food photographs that share low-level statistics with the training pool, whereas FruitSeg30 contains isolated fruit in natural, uncontrolled environments (Shamrat et al., 2024) — a different visual domain that the model was never trained on, especially because the Fruits class in training comes only from FoodSeg103. We therefore treat the FruitSeg30 result as an explicit limitation and as evidence for the generalisation and failure-analysis contribution, not as a headline strength.

### Baseline Comparison

A full-budget comparison against an alternative segmentation architecture could not be completed: the main model was trained on a cloud T4 GPU, whereas the local environment available for this study had a CPU-only PyTorch build, so retraining an alternative architecture with the same 100-epoch budget and split was not feasible within the study window. A short five-epoch screening run with three alternative architectures is available from an earlier notebook: mask mAP50 was 54.67% for YOLOv8m-seg, 52.93% for YOLO11m-seg, and 53.45% for YOLO26m-seg (box mAP50 56.27%, 54.58%, and 54.89%, respectively). Because that run used a reduced training subset and only five epochs, these numbers are reported only as a screening pilot; they are not comparable to the 100-epoch YOLO11s-seg model (internal mask mAP50 71.66%) and must not be read as evidence that any architecture is superior. A matched full-budget baseline remains future work.

### Nutrition-Label Extraction and Preprocessing Ablation

The OCR module was evaluated with field-level accuracy on Indonesian snack-label photographs with hand-transcribed ground truth. This evaluation is an exploratory pilot on six images (46 fields); it is reported to document the effect of the layout parser and is not a generalisation result. The baseline parser achieved **21.74%** field-level accuracy (95% cluster-bootstrap CI [16.67%, 25.58%]), and the revised grid parser achieved **47.83%** (95% CI [29.17%, 66.67%]) — more than double — with missing fields falling from 65.22% to 34.78% of the reference fields (Table 10).

Table 10. Field-level nutrition-label extraction results (exploratory pilot).

| Pipeline | Images | Fields | Field accuracy | 95% CI | Exact | Wrong | Missing |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: |
| Baseline parser | 6 | 46 | 21.74% | [16.67%, 25.58%] | 21.74% | 13.04% | 65.22% |
| Grid parser (revised) | 6 | 46 | 47.83% | [29.17%, 66.67%] | 47.83% | 17.39% | 34.78% |

The dominant failure mode of the baseline was coverage rather than mis-recognition: missing fields accounted for 65.22% of the reference fields because PP-OCR merged the numeric grid and the parser failed to detect several rows, so most errors were absent values rather than wrong values. The revised parser addresses this by detecting value columns relative to the row labels and by splitting merged numeric boxes, which reduced missing fields to 34.78% and improved the sodium and energy fields most; on one image the grid split rows incorrectly, where the baseline had been correct. Residual errors concentrate on wrong-value assignment (for example, a mis-parsed sugar value with a very large relative error) and on serving-size formatting and distinguishing per-serving from per-100 g values. Table 11 then reports the per-stage preprocessing ablation, and Figures 9 and 10 show the pipeline and a qualitative example.

[[INSERT FIGURE 9 HERE: nutrition-label extraction pipeline diagram]]

Figure 9. Nutrition-label extraction pipeline: preprocessing, recognition, and layout reconstruction.

[[INSERT FIGURE 10 HERE: qualitative OCR example showing input, region of interest, and reconstructed table]]

Figure 10. Qualitative example of label extraction.

Table 11. Per-stage OCR preprocessing ablation (cumulative; the layout parser stays active; exploratory pilot, n=6, 46 fields).

| Configuration | Field accuracy | 95% CI | Δ vs raw | Exact | Wrong | Missing |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| Raw OCR (no preprocessing) | 32.61% | [27.27%, 37.50%] | — | 15 | 17 | 14 |
| + deskew | 39.13% | [27.27%, 53.85%] | +6.52 | 18 | 14 | 14 |
| + deglare | 39.13% | [27.27%, 53.85%] | +6.52 | 18 | 14 | 14 |
| + CLAHE | 41.30% | [25.00%, 59.52%] | +8.69 | 19 | 12 | 15 |
| + ROI | 41.30% | [25.00%, 59.52%] | +8.69 | 19 | 10 | 17 |
| + semantic filter | 41.30% | [25.00%, 59.52%] | +8.69 | 19 | 10 | 17 |
| + fuzzy matching | 50.00% | [30.43%, 68.89%] | +17.39 | 23 | 11 | 12 |
| + energy/fat fallback | 50.00% | [30.43%, 68.89%] | +17.39 | 23 | 11 | 12 |
| + clinical validation | 47.83% | [29.17%, 66.67%] | +15.22 | 22 | 7 | 17 |
| Full pipeline | 47.83% | [29.17%, 66.67%] | +15.22 | 22 | 7 | 17 |

In the ablation, the largest single gain came from fuzzy field mapping (+8.70 points, from 41.30% to 50.00%), followed by deskewing (+6.52 points from raw). Deglare did not change the result on its own, while ROI and the semantic filter left field accuracy unchanged but converted some wrong values into missing ones, and the final clinical-validation stage slightly reduced accuracy (47.83%) by discarding low-confidence values conservatively. Because the pilot contains only six images and 46 fields, the row-to-row differences sit inside wide confidence intervals and are reported as directional evidence for stage-level behaviour, not as precise effect sizes. The parser comparison in Table 10 and the ablation in Table 11 are complementary: Table 10 measures the contribution of the layout parser, whereas Table 11 isolates the preprocessing and mapping stages on top of that parser.

### Clinical Rules Engine and Human-in-the-Loop Integration

The Clinical Rules Engine computes a user's TDEE from BMR and activity factor, derives personal limits (Table 4), and classifies accumulated intake as SAFE, WARNING, or DANGER using the thresholds in Table 5. Because the engine is deterministic, its behaviour can be checked exactly rather than estimated. We therefore compared the implemented engine with an independent re-implementation of the specification in Tables 4 and 5 over six clinical profiles and five food scenarios. All 36 daily-limit values and all 30 end-to-end status decisions matched between implementation and specification (36/36 and 30/30; Table 12), which is a functional consistency check between code and specification. It is not a clinical validation: no registered dietitian reviewed whether the guideline values are appropriate for individual patients.

We next measured how label-extraction errors propagate to the final decision. For each of the six pilot products and each of the six profiles (36 decisions), the SAFE/WARNING/DANGER status was computed from the hand-transcribed ground truth and, separately, from the fields produced by the two OCR parsers (Table 13). The baseline parser preserved the ground-truth status in 18 of 36 decisions (50.0%) and the grid parser in 22 of 36 (61.1%). The errors were not random: a mis-parsed sugar or sodium value (for example, a per-serving sugar of 10 g read as 70.5 g, or the sodium value 60 read as sugar) produced a false DANGER, whereas a missing field defaulted to zero and produced a false SAFE. Even the improved parser therefore changed the clinical label in nearly two of every five decisions. This is the empirical justification for the Human-in-the-Loop design: because raw recognition cannot be trusted to leave the decision unchanged, user verification is a functional requirement rather than an optional convenience. The HITL layer lets the user correct predicted polygons and extracted values before the rules are applied, and only the corrected values are consumed by the engine.

Table 12. Clinical Rules Engine verification against an independent re-implementation of the specification.

| Check | Matched | Total | Scope |
| :--- | ---: | ---: | :--- |
| Daily-limit values | 36 | 36 | 6 profiles × 6 limit fields |
| Status decisions | 30 | 30 | 6 profiles × 5 food scenarios |

Table 13. End-to-end decision propagation from label-extraction error (6 products × 6 profiles = 36 decisions).

| Pipeline | Status unchanged vs. ground truth | Status changed |
| :--- | ---: | ---: |
| Baseline parser | 18 (50.0%) | 18 |
| Grid parser (revised) | 22 (61.1%) | 14 |

### Usability Assessment

System usability was assessed with Nielsen's heuristic evaluation using four evaluators (Table 14). Severity weighting was based on the independent E1–E2 assessment; E4's findings were used only for the documentation–implementation consistency check and were excluded from the weighting. The evaluation found no severity-4 (catastrophic) findings, and the remaining findings were of severity 2–3 (minor to major). None of the reported problems prevented the core interaction: the two HITL controls — the segmentation editor and the nutrient value verification form — were the affordances that let a user recover from the recognition failures quantified earlier, so the usability result and the HITL design are consistent. Because the severity weighting rests on a two-evaluator panel, this is reported as an exploratory usability assessment and not as a substitute for a study with end users.

Table 14. Nielsen heuristic evaluation evaluator profile.

| Evaluator | Role | Contribution |
| :--- | :--- | :--- |
| E1 | Machine-learning engineer / data scientist | Independent heuristic review (severity-weighted) |
| E2 | Machine-learning engineer / data scientist | Independent heuristic review (severity-weighted) |
| E3 | Lead data scientist | Asynchronous architecture review |
| E4 | Self-evaluator | Documentation–implementation consistency (excluded from severity weighting) |

### Discussion

The results separate cleanly into what generalises and what does not. Segmentation is competitive on in-domain plated-food sources and collapses on a cross-domain, fruit-only source; the external numbers therefore support a claim of supervised in-domain performance with a quantified domain shift, not of universal generalisation. This matches the practical concern raised by recent food-segmentation benchmarks, which report limited data diversity and degradation under new viewpoints and domains (AlMughrabi et al., 2026).

For label reading, the pilot indicates that layout reconstruction, not raw recognition, is the main bottleneck for Indonesian nutrition tables: the same OCR engine gained more than twenty accuracy points once merged cells and value columns were parsed correctly. This is consistent with studies showing that dense tabular packaging layouts are a major source of OCR error (Nagayi et al., 2025) and that preprocessing and layout-aware correction are necessary in practice (Saputra et al., 2024; Saputra et al., 2026; Alireza & Kurniawardhani, 2026). The honest framing is that the OCR evidence is currently a pilot and that the headline number must be re-established on a larger, brand-stratified set.

At the system level, the contribution is architectural: verified segmentation outputs and verified label values are converted into transparent, guideline-aligned feedback by a deterministic rule layer that runs locally, with a user-verification step retained. This combination — plate recognition and packaged-label extraction feeding one explicit decision layer, offline — is less commonly reported than improvements to any single recognition model, and it is the gap this study targets.

The main threats to validity are as follows. The OCR sample is small and not brand-stratified, so no generalisation claim is made. The internal validation represents one seed, and a multi-seed estimate would strengthen the segmentation result. The comparison against an alternative segmentation architecture is only a screening baseline. The end-to-end propagation shows that label-extraction error changes the SAFE/WARNING/DANGER status in 39–50% of the pilot decisions, which is why the decision layer is designed never to run on unverified values. No registered-dietitian review was performed, so the clinical layer is verified functionally rather than clinically. The usability assessment used a two-evaluator severity panel rather than end users, so its findings are exploratory. Finally, no public Indonesian plate-segmentation benchmark exists, so Indonesian-specific generalisation could not be measured quantitatively.

## CONCLUSIONS AND SUGGESTIONS

### Conclusion

This research developed and evaluated Healthier as an integrated, edge-oriented nutrition-assistance system that combines YOLO11s-seg plate segmentation, PaddleOCR (PP-OCRv4) nutrition-table extraction, a deterministic Clinical Rules Engine, and HITL verification. On internal validation the segmentation model reached a mask mAP50 of **71.66%** (mAP50-95 **56.91%**). Across external sources it retained a mask mAP50 of **69.15%** (UECFoodPIXCOMPLETE) and **63.66%** (FoodSeg103) but dropped to **17.06%** on the cross-domain FruitSeg30 set, a quantified domain shift reported as a limitation rather than concealed. For nutrition labels, an exploratory six-image pilot showed that a grid-based layout parser more than doubles field-level accuracy relative to a baseline parser (**21.74% → 47.83%**), confirming that layout reconstruction is the key bottleneck. The findings should be read within the study's scope: the OCR sample is small, portion estimation relies on 2D image information, and the heterogeneous Others class remains the weakest category. The deterministic rule layer was verified functionally against its specification (36/36 daily-limit values and 30/30 status decisions matched), and an end-to-end propagation test showed that label-extraction error changes the clinical status in 39–50% of the pilot decisions, which is why the HITL verification step is retained. A Nielsen heuristic evaluation found no severity-4 problems, with the remaining findings at severity 2–3. Together, the deterministic rule layer and the HITL controls provide a transparent mechanism for converting uncertain model outputs into guideline-aligned feedback that the user can correct before it is used, demonstrating the technical feasibility of integrating visual food analysis and packaged-food label extraction in a single local workflow while making the generalisation boundaries explicit.

### Suggestion

Four directions are recommended for future work. First, data coverage: build an Indonesian plate-segmentation and nutrition-label corpus, including soups and texturally complex foods, because no public Indonesian plated-meal segmentation benchmark currently exists. Second, generalisation and estimation: improve cross-source robustness and estimate portions more reliably than the current 2D area ratio, for example with depth or multi-angle imaging. Third, OCR scale and rigour: expand the label evaluation to at least 100 brand-stratified images and complete the per-stage preprocessing ablation. Fourth, deployment: apply quantisation and further edge optimisation, and consider an optional offline small language model for explanation or normalisation, but only as a subordinate component that never overrides the deterministic nutrition rules or the explicit human verification. Prospective studies with real users and nutrition professionals are needed to validate the clinical usefulness of the feedback.

## REFERENCES

Alireza, R. A., & Kurniawardhani, A. (2026). Development of an OCR system for ingredient list text recognition on food labels using CRAFT and CRNN. MALCOM: Indonesian Journal of Machine Learning and Computer Science, 6(4), 2111–2121. https://doi.org/10.57152/malcom.v6i4.2707

AlMughrabi, A., Rivo, G., Jiménez-Farfán, C., Haroon, U., Al-Areqi, F., Jung, H., Busam, B., Marques, R., & Radeva, P. (2026). BenchSeg: A large-scale dataset and benchmark for multi-view food video segmentation. arXiv:2601.07581.

Bakar, L., Englhardt, Z., Srinivas, V., Narayanswamy, G., Nissanka, D., Patel, S., & Iyer, V. (2025). SnappyMeal: Design and longitudinal evaluation of a multimodal AI food logging application. arXiv:2511.03907.

Cui, C., Sun, T., Lin, M., Gao, T., Zhang, Y., Liu, J., et al. (2025). PaddleOCR 3.0 technical report. arXiv:2507.05595.

Fauzan, A., & Pamungkas, H. (2025). Regional food recognition with nutritional analysis: Attention mechanism implementation for Indonesian cuisine. In Proceedings of the 8th International Conference on Informatics, Engineering, Science & Technology (INCITEST 2025) (Advances in Engineering Research, Vol. 287). https://doi.org/10.2991/978-94-6463-924-7_16

Heydari, S., & Mahmoud, Q. H. (2025). Tiny machine learning and on-device inference: A survey of applications, challenges, and future directions. Sensors, 25(10), 3191. https://doi.org/10.3390/s25103191

James, P. A., Oparil, S., Carter, B. L., Cushman, W. C., Dennison-Himmelfarb, C., Handler, J., et al. (2014). 2014 evidence-based guideline for the management of high blood pressure in adults: Report from the panel members appointed to the Eighth Joint National Committee (JNC 8). JAMA, 311(5), 507–520. https://doi.org/10.1001/jama.2013.284427

Kahendra, F., Lestari, P. B., Aushof, A. D., Nabila, A., & Gunawan, T. (2026). Label Nutri-Level pada kemasan pangan sebagai instrumen kebijakan kesehatan dalam mendorong perilaku konsumsi sehat masyarakat. SerQua: Service Quality (Jurnal Pengabdian Masyarakat), 3(2), 220–225.

Kementerian Kesehatan Republik Indonesia. (2024). Prevalence, impact, and efforts in controlling hypertension & diabetes in Indonesia: Analysis of situation and issues. Badan Kebijakan Pembangunan Kesehatan.

Khanam, R., & Hussain, M. (2024). YOLOv11: An overview of the key architectural enhancements. arXiv:2410.17725.

Mivtahurrahimah, & Sevtiyani, I. (2025). Factors contributing to rising diabetes cases in Indonesia: Insights from the 2023 Indonesian Health Survey. Jurnal Epidemiologi Kesehatan Komunitas, 10(3). https://doi.org/10.14710/jekk.v10i3.28184

Nagayi, M., Khan, A., Frank, T., Swart, R., & Nyirenda, C. (2025). Evaluating OCR performance on food packaging labels in South Africa. arXiv:2510.03570.

Okamoto, K., & Yanai, K. (2021). UEC-FoodPIX Complete: A large-scale food image segmentation dataset. In ICPR Workshop on Multimedia Assisted Dietary Management (MADiMa).

Perkumpulan Endokrinologi Indonesia (PERKENI). (2024). Pedoman pengelolaan dan pencegahan diabetes melitus tipe 2 di Indonesia 2024. PB PERKENI.

Rasyidi, M. A., Mardhiyyah, Y. S., Nasution, Z., & Wijaya, C. H. (2024). Performance comparison of state-of-the-art deep learning model architectures in Indonesian food image classification. Bulletin of Electrical Engineering and Informatics, 13(5), 3355–3368. https://doi.org/10.11591/eei.v13i5.7996

Rouhafzay, A., Rouhafzay, G., & Jbilou, J. (2025). Image-based food monitoring and dietary management for patients living with diabetes: A scoping review of calorie counting applications. Frontiers in Nutrition, 12, 1501946. https://doi.org/10.3389/fnut.2025.1501946

Saputra, A. N. P., Yazid, A. S., Pramuntadi, A., & Danianti, D. (2026). Implementasi OCR dan logika fuzzy untuk pengecekan nutrisi pada label makanan kemasan. Jurnal Profesi Insinyur Universitas Lampung (JPI), 7(1). https://doi.org/10.23960/jpi.v7n1.380

Saputra, T. M. E., Susanto, A., & Carmelita, B. J. (2024). Implementation of Tesseract OCR and bounding box for text extraction on food nutrition labels. Building of Informatics, Technology and Science (BITS), 6(3), 1403–1412. https://doi.org/10.47065/bits.v6i3.6107

Shamrat, F. M. J. M., Shakil, R., Idris, M. Y. I., Akter, B., & Zhou, X. (2024). FruitSeg30_Segmentation dataset & mask annotations: A novel dataset for diverse fruit segmentation and classification. Data in Brief, 56, 110821. https://doi.org/10.1016/j.dib.2024.110821

Shanti, K. M. (2025). Efektivitas label gizi traffic light terhadap pemahaman, persepsi, dan sikap konsumen di Indonesia: Systematic literature review. Jurnal Gizi dan Kesehatan (Universitas Sultan Ageng Tirtayasa), 6(1). https://doi.org/10.62870/jgkp.v6i1.31589

Sultana, J., Ahmed, B. M., Masud, M. M., Huq, A. K. O., Ali, M. E., & Naznin, M. (2023). A study on food value estimation from images: Taxonomies, datasets, and techniques. IEEE Access, 11. https://doi.org/10.1109/ACCESS.2023.3274475

Telea, A. (2004). An image inpainting technique based on the fast marching method. Journal of Graphics Tools, 9(1), 23–34.

Ultralytics. (2024). YOLO11 [Computer software]. https://github.com/ultralytics/ultralytics

Wang, W., Min, W., Li, T., Dong, X., Li, H., & Jiang, S. (2022). A review on vision-based analysis for automatic dietary assessment. Trends in Food Science & Technology, 122, 1–12.

Wang, X., Tang, Z., Guo, J., Meng, T., Wang, C., Wang, T., & Jia, W. (2025). Empowering edge intelligence: A comprehensive survey on on-device AI models. ACM Computing Surveys.

Wu, X., Fu, X., Liu, Y., Lim, E.-P., Hoi, S. C. H., & Sun, Q. (2021). A large-scale benchmark for food image segmentation. In Proceedings of the 29th ACM International Conference on Multimedia (pp. 506–515). https://doi.org/10.1145/3474085.3475201
