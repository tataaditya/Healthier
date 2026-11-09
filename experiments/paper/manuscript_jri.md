# Application of Deep Learning for Nutritional Segmentation and Information Extraction in the Healthier Application

Tata Aditya Pamungkas^1, Nuke Lu'lu Ul Chusna^2, Harry Dwiyana Kartika^3

Informatics Engineering Study Program, Universitas Krisnadwipayana, Jakarta, Indonesia

Email: tataaditaya@gmail.com; nukelchusna@unkris.ac.id; harry_d@unkris.ac.id

## Abstract

Low nutritional literacy and the rising burden of non-communicable diseases (NCDs) motivate **Healthier**, an offline application that combines plate-image food recognition with nutrition-label extraction. The system integrates three components: a YOLO11s-seg instance-segmentation model that partitions a plate into five *Isi Piringku* groups; a PaddleOCR (PP-OCRv4) pipeline with a grid-based layout parser for nutrition tables; and a deterministic Clinical Rules Engine that converts verified values into SAFE/WARNING/DANGER feedback, with a Human-in-the-Loop correction step. The segmentation model was trained on 11,684 images and validated on 2,017 images from public datasets that are not specific to Indonesian cuisine. Generalisation was measured on three independent external test sets (5,013 images) never used for training or model selection. On internal validation the model reached a mask mAP50 of 71.66% and a mask mAP50-95 of 56.91%. It transferred well to two food-photography sources (69.15% on UECFoodPIXCOMPLETE and 63.66% on FoodSeg103) but degraded sharply on a fruit-only, in-the-wild dataset (17.06%), a measured domain shift. Label extraction was evaluated on 131 manually verified Indonesian label photographs across five product categories: the full pipeline reached 42.65% field-level accuracy, against 35.11% for raw PaddleOCR and 14.89% for Tesseract, with grid reconstruction contributing +13.05 points to an otherwise identical configuration; the semantic filter proved harmful and is reported as a negative finding rather than tuned away. The deterministic rule layer matched its specification on all 66 checks, while extraction error changed the clinical status in 15.0% of 786 decisions across 131 verified images (21.4% on images that contain a nutrition table), and a ground-truth DANGER became SAFE in 18 of 25 DANGER decisions because missing fields default to zero, motivating the retained human verification. The main contribution is a reproducible integration and decision mechanism, not a single model, supported by cross-source generalisation, per-class error analysis, and an open evaluation protocol.

Keywords: Instance Segmentation; Nutrition Label OCR; YOLO11s-seg; PaddleOCR; Clinical Decision Rules

## Abstrak

Literasi gizi yang rendah dan beban penyakit tidak menular (PTM) yang meningkat melatarbelakangi **Healthier**, aplikasi luring yang menggabungkan pengenalan makanan dari citra piring dengan ekstraksi label gizi dari kemasan. Sistem ini mengintegrasikan tiga komponen: model segmentasi instan YOLO11s-seg yang membagi citra piring menjadi lima kelompok *Isi Piringku*; pipeline PaddleOCR (PP-OCRv4) dengan parser tata letak berbasis grid untuk tabel gizi; dan Clinical Rules Engine deterministik yang mengubah nilai terverifikasi menjadi status SAFE/WARNING/DANGER, dengan tahap koreksi Human-in-the-Loop. Model segmentasi dilatih pada 11.684 citra dan divalidasi pada 2.017 citra dari dataset publik yang tidak spesifik masakan Indonesia. Generalisasi diukur pada tiga set uji eksternal independen (5.013 citra) yang tidak pernah dipakai untuk pelatihan maupun pemilihan model. Pada validasi internal, model mencapai mask mAP50 71,66% dan mask mAP50-95 56,91%. Model bertransfer baik ke dua sumber fotografi makanan (mask mAP50 69,15% pada UECFoodPIXCOMPLETE dan 63,66% pada FoodSeg103) tetapi turun tajam pada dataset khusus buah di lingkungan alami (17,06%), suatu *domain shift* terukur. Ekstraksi label dievaluasi pada 131 citra label kemasan Indonesia yang diverifikasi manual di lima kategori produk: pipeline penuh mencapai akurasi tingkat field 42,65%, dibandingkan 35,11% untuk PaddleOCR mentah dan 14,89% untuk Tesseract, dengan rekonstruksi grid menyumbang +13,05 poin pada konfigurasi yang sama; filter semantik terbukti merugikan dan dilaporkan sebagai temuan negatif. Lapisan aturan deterministik cocok dengan spesifikasinya pada seluruh 66 pemeriksaan, sementara galat ekstraksi mengubah keputusan klinis pada 15,0% dari 786 keputusan di 131 citra terverifikasi (21,4% pada citra yang memuat tabel gizi), dan status DANGER pada ground truth berubah menjadi SAFE pada 18 dari 25 keputusan DANGER karena field yang hilang bernilai bawaan nol, yang mendasari dipertahankannya verifikasi manusia. Kontribusi utama bukan pada satu model tunggal, melainkan pada mekanisme integrasi dan pengambilan keputusan yang reproducible, didukung generalisasi antar-sumber, analisis kesalahan per kelas, dan protokol evaluasi yang terbuka.

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

The study targets two families of objects and one target user profile. The segmentation target consists of plated-food images; the evaluation subjects are three public image datasets (UECFoodPIXCOMPLETE, FoodSeg103, and FruitSeg30) whose original annotations were reused. The OCR target consists of photographs of Indonesian packaged-food nutrition labels across five product categories (Frozen Food, Makanan Kaleng, Makanan Ringan, Minuman Sachet, and Saos dan Perbumbuan); from a deduplicated pool of 438 photographs (453 raw candidates minus 15 MD5/dHash duplicates), a locked test split of 131 photographs with human-verified ground truth was used for evaluation, while the remaining 307 photographs were set aside as an unannotated development split. The clinical target is everyday users with diet-related NCD risk profiles (type-2 diabetes, hypertension, dyslipidaemia, and diabetic nephropathy). Sampling for the public datasets followed the official splits; label photographs were collected across five product categories available in local stores.

### Procedure

The end-to-end procedure has two input paths that share one verification layer and one decision layer. A plate photograph is segmented into food instances and grouped into five *Isi Piringku* groups (Staple Food, Side Dishes, Vegetables, Fruits, Others); a packaging photograph is preprocessed, recognised, and parsed into nutrition fields. Both outputs pass through a HITL verification layer and are then converted by the deterministic Clinical Rules Engine into SAFE/WARNING/DANGER feedback. Figure 1 summarises the workflow.

[[INSERT FIGURE 1 HERE: system architecture and experimental flow, redrawn from the project architecture diagram]]

Figure 1. Healthier system architecture and experimental workflow.

#### Dataset Construction, Annotation, and Splits

Three public datasets were used: UECFoodPIXCOMPLETE (Okamoto & Yanai, 2021), FoodSeg103 (Wu et al., 2021), and FruitSeg30 (Shamrat et al., 2024). FruitSeg30 contains 1,969 images in 30 fruit classes collected in natural and uncontrolled environments across Malaysia, Bangladesh, and Australia (Shamrat et al., 2024). The candidate corpus contained 19,087 images. Duplicate images were detected with a difference hash (dHash) computed on an 8×8 grayscale reduction of each image, and 373 duplicates were removed, leaving 18,714 unique images. Class labels from all sources were remapped into the five *Isi Piringku* groups using a fixed mapping table.

We did not re-annotate the public masks; the original annotations were used as provided, and only the label remapping was applied. Specifically, 104 source classes in FoodSeg103 (12 Staple Food, 12 Side Dishes, 34 Vegetables, 21 Fruits, and 25 Others) and 102 category IDs in UECFoodPIXCOMPLETE (38 Staple Food, 42 Side Dishes, 11 Vegetables, 0 Fruits, and 11 Others) were mapped deterministically into the five *Isi Piringku* groups using a frozen lookup table (`experiments/configs/foodseg_class_map.csv` and `experiments/results/segmentation/dataset_audit.md`). Notably, 24.0% of FoodSeg103 classes (25 of 104, including candy, chocolate, cream, biscuit, jam, honey, and sauces) map to `Others`, making `Others` a semantically heterogeneous catch-all category rather than a uniform visual class; this structural heterogeneity is documented explicitly to explain its lower segmentation precision. Because the remapping was defined by a single author without a second independent annotator, inter-annotator agreement (Cohen's κ) is not available and is reported as an explicit limitation.

The training pool consisted of the official training portions of UEC and FoodSeg103 and was split 85:15 into training (11,684) and validation (2,017) sets, stratified by class with seed 42 and no overlap. Validation images are therefore a held-out 15% of the training pool. Three sets were reserved exclusively for external testing and were never used for training, early stopping, or checkpoint selection: the official UEC test set (948 images), the FoodSeg103 validation set (2,096 images), and the whole of FruitSeg30 (1,969 images), for 5,013 external images in total. Because the Fruits class in the training data comes only from FoodSeg103, FruitSeg30 acts as a deliberate domain-shift test. A class-imbalance audit across splits shows that in the training set, the max/min instance ratio is 3.04× (`Vegetables` 10,689 instances [28.0%] vs `Fruits` 3,513 instances [9.2%]) and the image-level presence ratio is 4.12× (`Staple Food` 5,708 images [26.6%] vs `Fruits` 1,384 images [6.4%]); in internal validation the ratios are 3.04× (instance) and 4.30× (image). Across external test splits, instance ratios are 3.37× on UEC test (where `Fruits` is absent), 2.90× on FoodSeg103 validation, and 1.00× on FruitSeg30 (which contains 100% `Fruits`), meaning unweighted macro-averages across external sources are not directly comparable without per-class breakdown. Plain Ultralytics training was used without oversampling or class-weighted loss so that external generalisation is not confounded by artificial resampling priors; class imbalance is therefore reported as measured and analysed alongside per-class AP. Table 1 reports the split composition and Table 2 the per-class instance distribution.

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

For each plate image, the model outputs one segmentation polygon per detected food instance. Instances are assigned to their *Isi Piringku* group, and the 2D visual area proportion of a group is computed as the ratio of the sum of its food-polygon areas to the total analysed food area on the plate (Equation 1).

EQ: Proportion_g = A_g / Σ_g A_g          (1)

where A_g is the total polygon area of group g. Detections of non-food objects (cutlery, the plate itself, hands) are filtered out before the proportion is computed so that the proportions are based on food pixels only. Crucially, Equation 1 measures a 2D visual area proportion for qualitative *Isi Piringku* plate-balance feedback; it is not a quantitative mass (gram) or caloric estimate. Without a depth sensor, fiducial scale marker, or food-specific density table, projecting 3D food volume onto a 2D image plane introduces geometric uncertainty from camera tilt (perspective foreshortening), inter-item occlusion, and varying pile height (for example, mounded staple rice versus flat leafy vegetables). Consequently, polygon areas are never automatically converted into grams or kilocalories, and quantitative thresholds in the Clinical Rules Engine operate exclusively on verified nutrient values rather than 2D mask areas.

#### Nutrition-Label Extraction Pipeline

Packaging images pass through four preprocessing stages before recognition: EXIF auto-orientation, deskewing using the Hough transform (Heydari & Mahmoud, 2025), deglare using Telea inpainting (Zhou et al., 2023), CLAHE-based contrast normalisation, and region-of-interest cropping around the nutrition table. The processed region is recognised with PaddleOCR (Cui et al., 2025) configured with the PP-OCRv4 recognition model. An earlier draft of this manuscript referred to PP-OCRv5; that was a documentation error. The implementation pins the model version explicitly (`NUTRILABEL_OCR_VERSION`, default `PP-OCRv4`), and every result reported here was produced with PP-OCRv4; the table-structure stage uses the SLANet_plus table-recognition pipeline.

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

The engine classifies accumulated intake into SAFE, WARNING, and DANGER (Table 5). A WARNING is raised when sugar or saturated fat exceeds 50% of the daily limit, when actual sodium exceeds 40% of the sodium limit, when protein exceeds 30% of the nephropathy quota, or when a product's sodium density exceeds 600 mg/100 g for hypertension (800 mg/100 g otherwise). DANGER is raised only when sugar exceeds 100% of the daily limit; sodium and saturated fat saturate at WARNING. These bands are the exact thresholds implemented in the application (`frontend/js/clinical-rules.js`) and verified by an independent specification check (`experiments/scripts/verify_clinical_rules.js`); they are reported as functional rule verification, not clinical validation, because no independent registered-dietitian review was performed.

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

The data are secondary for segmentation (public datasets with their original annotations) and primary for label extraction (photographs of Indonesian packaged food labels across five categories, together with human-verified ground-truth values). The segmentation data separation was performed with a difference-hash script; for label extraction, 453 candidate photographs were deduplicated via MD5 and dHash (removing 15 duplicates) into 438 unique images, which were split deterministically (`seed=42`, SHA-256 manifest lock) into 131 test images and 307 development images (`experiments/results/ocr_eval/split_lock.json`). Following a written annotation protocol (`experiments/results/ocr_eval/annotation_guideline.md`), an initial visual draft was reviewed and corrected image-by-image by the human annotator in a browser verification interface (`viewer.html`) without inspecting pipeline predictions: values were recorded strictly from the absolute *Per Sajian* (per-serving) column, unprinted nutrient rows were marked `absent`, and printed rows whose numeric values were illegible due to blur, crease, or glare were marked `unreadable` (38 fields across 19 images) and excluded from denominators rather than imputed. Human verification modified 177 cells across 36 of the 131 images (leaving 95 images unchanged) before locking `ground_truth_verified.json`. The instruments were Python 3 with Ultralytics and PaddleOCR for modelling, and a desktop web application (HTML, CSS, and JavaScript) for the HITL interfaces and the clinical rule engine. Images for label extraction were collected from local stores and cover five product categories; brand was not annotated as a field, so the evaluation is stratified by category rather than by brand.

### Data Analysis Technique

Segmentation is evaluated at the instance level (box and mask precision, recall, mAP50, and mAP50-95) and at the pixel level (per-class accuracy, precision, recall, and F1), with normalised confusion matrices for the internal validation set and for each external test set. Per-class average precision is reported to expose source-specific failure modes. Label extraction is evaluated at the field level because the downstream system consumes specific nutrition fields rather than full transcriptions; a field is counted as correct when it is exact or within 5% of the ground-truth per-serving value (Equation 8).

EQ: FieldAccuracy = (N_exact + N_within5%) / N_fields          (8)

Exact, wrong, and missing fields are reported separately, and the confidence interval is obtained by a cluster bootstrap over images (2,000 iterations, seed 42) rather than a sample-size-based interval; paired pipeline differences are tested with paired image-level bootstrap intervals and McNemar's test. The evaluated fields are energy, total fat, saturated fat, trans fat, cholesterol, sodium, carbohydrate, fibre, sugar, protein, and serving size; values are taken per serving, fields absent from a package are recorded as `absent` (excluded from accuracy denominators, while pipeline values emitted on `absent` fields are counted as spurious edits in HITL workload analysis), and printed fields illegible in the photograph are recorded as `unreadable` and excluded from all denominators. Clinical-rule behaviour is evaluated by rule verification: because the engine is deterministic, the implemented engine is compared against an independent re-implementation of the documented specification (Tables 4 and 5) over six clinical profiles and five food scenarios, so that any divergence between code and specification is detected exactly rather than estimated. The end-to-end effect of extraction error is measured across all 131 verified images and six clinical profiles (786 decisions, including 426 decisions on the 71 table-bearing images) by applying the same rules to the human-verified ground-truth values and to the OCR-extracted values, and reporting how often the SAFE/WARNING/DANGER status changes. On-device deployment characteristics are measured on the local target CPU (`experiments/scripts/bench_edge.py`) by recording median and 95th-percentile (P95) wall-clock latency, peak resident set size (RSS) memory, and on-disk model artifact sizes. Usability is assessed with Nielsen's heuristic evaluation using four evaluators: two independent machine-learning engineers/data scientists (E1, E2), one lead data scientist performing an asynchronous architecture review (E3), and one self-evaluator (E4) checking documentation–implementation consistency; severity weighting is based on the independent E1–E2 assessments, and E4 is reported separately. All random seeds are fixed (seed 42).

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

The OCR module was evaluated on a held-out test set of 131 Indonesian packaged-food label photographs spanning five product categories (Frozen Food 20, Makanan Kaleng 16, Makanan Ringan 53, Minuman Sachet 18, Saos dan Perbumbuan 24). The test split was locked before evaluation (seed 42) and the ground truth was transcribed and verified manually; the assistant's output was used only as a pre-annotation draft. Of the 582 field annotations that are printed on the labels, 544 carry a legible numeric value and form the denominator for every configuration; 38 printed values were unreadable in the photographs (mainly trans fat, cholesterol, and fibre) and were excluded from all denominators rather than imputed. Sixty of the 131 photographs contain no nutrition table at all (composition-only or single-ingredient labels); they are retained to test whether a pipeline suppresses rather than hallucinates values.

Table 10. Field-level nutrition-label extraction on the verified 131-image test set.

| Pipeline | Images | Fields | Field accuracy | 95% CI | Exact | Wrong | Missing |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: |
| A. Tesseract | 131 | 544 | 14.89% | [10.28%, 20.15%] | 81 | 154 | 309 |
| B. EasyOCR | 131 | 544 | 19.67% | [15.03%, 24.53%] | 107 | 210 | 226 |
| C. PaddleOCR, raw + line parser | 131 | 544 | 35.11% | [29.29%, 40.89%] | 191 | 159 | 193 |
| D. Preprocessing + layout parser, no grid | 131 | 544 | 23.53% | [17.63%, 29.91%] | 128 | 53 | 363 |
| E. Full pipeline (grid reconstruction) | 131 | 544 | **42.65%** | [36.23%, 49.00%] | 232 | 98 | 213 |

The full pipeline (E) reached **42.65%** field-level accuracy (Table 10), 7.54 points above raw PaddleOCR (C; paired bootstrap 95% CI [+2.54, +12.80], McNemar p = 0.0021) and 27.76 points above Tesseract (A; p < 0.001); EasyOCR (B) reached 19.67%. The bare layout parser without grid reconstruction (D) scored 23.53%, i.e. **below** raw PaddleOCR, because on unprocessed photographs it converts many values into missing fields. The parser only becomes beneficial once preprocessing and grid reconstruction are active, which is why the contribution is reported for the complete pipeline rather than for the parser in isolation.

Table 11. Cumulative preprocessing and post-processing ablation (n = 131, 544 fields; the layout parser stays active).

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

The cumulative ablation does **not** support a monotonic "each stage helps" narrative, and the manuscript reports it as measured (Table 11). Deskewing and deglare left accuracy essentially unchanged (-1.29 and 0.00 points), CLAHE and the semantic filter reduced it (-5.33 and -1.29 points), and ROI recovered only 0.18 points. Fuzzy matching (+4.96) and the energy/fat fallback (+3.12) produced the largest positive step; clinical validation cost 0.74 points. The single largest effect is grid reconstruction: with an identical stage set, enabling the grid raises accuracy from 29.60% to 42.65% (**+13.05 points**, 95% CI [+8.73, +17.36], p = 1.4e-13). Leave-one-out runs on the full pipeline confirm the same asymmetry: removing fuzzy matching drops accuracy to 32.35% (-10.29 points, p = 1.2e-10), whereas removing **the semantic filter raises it to 46.14%** (+3.49 points, 95% CI [+1.08, +6.23], p = 8.8e-4). Removing the fallback or the validation stage does not change accuracy significantly (+0.55 points, p = 0.45; -0.37 points, p = 0.50). The semantic filter is therefore a net loss on this dataset, most likely because it discards numeric tokens whose surrounding text was mis-recognised, and this is reported as a negative finding rather than tuned away on the test set. Because leave-one-out was run only for the four post-processing stages, the image-stage effects are read from the cumulative table and are not claimed as independent leave-one-out estimates.

To quantify the human-in-the-loop correction burden, the verified ground truth was compared with the full pipeline's output used as the annotator's starting point. Of 544 fields, 232 (42.6%) were accepted unchanged, 423 field edits were required in total (including 111 spurious values that had to be cleared), 100 of 131 images (76.3%) needed at least one correction, and the median was 3 edits per image (mean 3.23, maximum 10). As a separate measure of pre-annotation quality, the human corrected 177 fields across 36 of the 131 draft images and left 95 images unchanged, so the draft reduced but did not eliminate transcription work. This is the empirical basis for treating verification as a required step rather than an optional convenience. Figures 9 and 10 illustrate the extraction pipeline and a qualitative extraction example.

[[INSERT FIGURE 9 HERE: nutrition-label extraction pipeline diagram]]

Figure 9. Nutrition-label extraction pipeline: preprocessing, recognition, and layout reconstruction.

[[INSERT FIGURE 10 HERE: qualitative OCR example showing input, region of interest, and reconstructed table]]

Figure 10. Qualitative example of label extraction.

### Clinical Rules Engine and Human-in-the-Loop Integration

The Clinical Rules Engine computes a user's TDEE from BMR and activity factor, derives personal limits (Table 4), and classifies accumulated intake as SAFE, WARNING, or DANGER using the thresholds in Table 5. Because the engine is deterministic, its behaviour can be checked exactly rather than estimated. We therefore compared the implemented engine with an independent re-implementation of the specification in Tables 4 and 5 over six clinical profiles and five food scenarios. All 36 daily-limit values and all 30 end-to-end status decisions matched between implementation and specification (36/36 and 30/30; Table 12), which is a functional consistency check between code and specification. It is not a clinical validation: no registered dietitian reviewed whether the guideline values are appropriate for individual patients.

We next measured how label-extraction errors propagate to the final decision on the verified test set. For each of the 131 images and each of the six clinical profiles (786 decisions, 426 of them on the 71 images that contain a nutrition table), the SAFE/WARNING/DANGER status was computed from the verified ground truth and, separately, from the fields extracted by raw PaddleOCR (C), the parser without grid (D), and the full pipeline (E). Raw PaddleOCR preserved the ground-truth status in 600 of 786 decisions (76.3%), D in 644 (81.9%), and E in 668 (85.0%); restricted to images that actually contain a table, the values are 66.7%, 72.3%, and 78.6% respectively. Agreement alone overstates quality, however, because 644 of the 786 ground-truth decisions are SAFE: a pipeline that outputs nothing defaults missing fields to zero and therefore lands on SAFE. The safety-relevant direction is the opposite one. The full pipeline changed a ground-truth DANGER into SAFE in 18 of 25 DANGER decisions (72%), and a WARNING into SAFE in 55 of 117; raw PaddleOCR did so in 9 of 25 and 47 of 117. In other words, the most accurate field-extraction configuration is also the most conservative about emitting values, and its conservatism becomes a false-SAFE risk once the rule layer substitutes zero for missing fields (Table 13). This is the strongest argument for the Human-in-the-Loop design: verification is not a convenience for improving accuracy, it is required to prevent under-warning, and the rule layer should not silently impute zero for absent values.

Table 12. Clinical Rules Engine verification against an independent re-implementation of the specification.

| Check | Matched | Total | Scope |
| :--- | ---: | ---: | :--- |
| Daily-limit values | 36 | 36 | 6 profiles × 6 limit fields |
| Status decisions | 30 | 30 | 6 profiles × 5 food scenarios |

Table 13. End-to-end decision propagation from label-extraction error (131 images x 6 profiles = 786 decisions; 426 decisions on the 71 images with a nutrition table).

| Pipeline | Status unchanged (all) | Status unchanged (table images) | GT DANGER changed to SAFE | GT WARNING changed to SAFE |
| :--- | ---: | ---: | ---: | ---: |
| C. Raw PaddleOCR | 600/786 (76.3%) | 284/426 (66.7%) | 9/25 | 47/117 |
| D. Parser without grid | 644/786 (81.9%) | 308/426 (72.3%) | 17/25 | 71/117 |
| E. Full pipeline | 668/786 (85.0%) | 335/426 (78.6%) | 18/25 | 55/117 |

### Usability Assessment

System usability was assessed with Nielsen's heuristic evaluation using four evaluators (Table 14). Severity weighting was based on the independent E1–E2 assessment; E4's findings were used only for the documentation–implementation consistency check and were excluded from the weighting. The evaluation found no severity-4 (catastrophic) findings, and the remaining findings were of severity 2–3 (minor to major). None of the reported problems prevented the core interaction: the two HITL controls — the segmentation editor and the nutrient value verification form — were the affordances that let a user recover from the recognition failures quantified earlier, so the usability result and the HITL design are consistent. Because the severity weighting rests on a two-evaluator panel, this is reported as an exploratory usability assessment and not as a substitute for a study with end users.

Table 14. Nielsen heuristic evaluation evaluator profile.

| Evaluator | Role | Contribution |
| :--- | :--- | :--- |
| E1 | Machine-learning engineer / data scientist | Independent heuristic review (severity-weighted) |
| E2 | Machine-learning engineer / data scientist | Independent heuristic review (severity-weighted) |
| E3 | Lead data scientist | Asynchronous architecture review |
| E4 | Self-evaluator | Documentation–implementation consistency (excluded from severity weighting) |

### Edge Deployment Benchmark: Latency, Memory, and Artifact Size

To evaluate local deployment feasibility without cloud dependence, component latency, peak memory consumption (resident set size, RSS), and model artifact sizes were measured on the local development machine using CPU execution (`experiments/results/edge/cpu/bench.md`, Table 15). Because the local PyTorch installation is a CPU-only build (`torch 2.15.0+cpu`, `torch.cuda.is_available() = False`), Ultralytics segmentation could not be benchmarked on the local GPU without replacing the environment; therefore, only verified CPU measurements are reported, and no GPU latency claims are made.

Table 15. Local CPU inference latency, peak memory footprint, and on-disk artifact sizes.

| Component / Artifact | Sample / Format | Median Latency (s) | P95 Latency (s) | Peak RSS (MB) | Artifact Size (MB) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| YOLO11s-seg inference | n = 30 images (640 px, 3 warm-up) | 0.1304 | 0.1854 | 770.0 | 20.5 (.pt) / 40.6 (.onnx) |
| Full OCR pipeline (PP-OCRv4 + SLANet_plus + grid) | n = 12 label images | 20.5777 | 45.2932 | 3,760.8 | 20.7 (total OCR) |
| — PP-OCRv4 detection (`mobile_det`) | Model weight | - | - | - | 4.9 |
| — PP-OCRv4 recognition (`en_mobile_rec`) | Model weight | - | - | - | 7.8 |
| — Table structure (`SLANet_plus`) | Model weight | - | - | - | 8.0 |
| Clinical Rules Engine | Per evaluation call | 0.000039 | - | - | <0.1 (JS) |
| Sequential end-to-end sum | Segmentation + OCR + Rules | 20.7081 | - | 3,760.8 | 41.2 (.pt + OCR) |

On CPU, plate segmentation is lightweight (median **0.1304 s**, P95 **0.1854 s**, peak RSS **770.0 MB**) and the deterministic Clinical Rules Engine is instantaneous (**0.039 ms** per call), with compact disk footprints (20.5 MB for YOLO11s-seg `.pt` weights and 20.7 MB across the three OCR models). However, the full nutrition-label OCR pipeline requires a median of **20.58 s** per image (P95 **45.29 s**) and **3,760.8 MB** peak RSS on CPU because it executes multi-stage image preprocessing, two-pass text detection/recognition, and SLANet_plus table-structure inference on high-resolution packaging photographs. This result demonstrates that while plate segmentation and rule evaluation are immediately practical on CPU, unaccelerated multi-stage OCR is the primary latency and memory bottleneck and is not yet real-time on CPU-only edge devices.

### Discussion

The results separate cleanly into what generalises and what does not. Segmentation is competitive on in-domain plated-food sources and collapses on a cross-domain, fruit-only source; the external numbers therefore support a claim of supervised in-domain performance with a quantified domain shift, not of universal generalisation. This matches the practical concern raised by recent food-segmentation benchmarks, which report limited data diversity and degradation under new viewpoints and domains (AlMughrabi et al., 2026).

For label reading, the verified 131-image evaluation indicates that layout reconstruction is the largest single contributor, not raw recognition: with an identical stage set, enabling grid reconstruction raised field accuracy from 29.60% to 42.65% (+13.05 points), and the complete pipeline reached 42.65% against 14.89% for Tesseract and 35.11% for raw PaddleOCR. The bare layout parser without grid reconstruction scored below raw OCR, so the benefit belongs to the combined preprocessing-plus-reconstruction design rather than to the parser in isolation. This is consistent with studies showing that dense tabular packaging layouts are a major source of OCR error (Nagayi et al., 2025) and that preprocessing and layout-aware correction are necessary in practice (Saputra et al., 2024; Saputra et al., 2026; Alireza & Kurniawardhani, 2026). The evaluation also produced a negative finding that is reported rather than tuned away: the semantic filter reduced accuracy, and removing it improved the full pipeline to 46.14% (+3.49 points, p = 8.8e-4).

At the system level, the contribution is architectural: verified segmentation outputs and verified label values are converted into transparent, guideline-aligned feedback by a deterministic rule layer that runs locally, with a user-verification step retained. This combination — plate recognition and packaged-label extraction feeding one explicit decision layer, offline — is less commonly reported than improvements to any single recognition model, and it is the gap this study targets.

The main threats to validity and scope limitations are as follows. First, the OCR evaluation covers 131 manually verified images across five product categories, but no brand field was annotated, so the results are stratified by product category rather than by brand and no brand-level generalisation claim is made; 38 printed values were unreadable in the photographs and were excluded from denominators rather than imputed, the fuzzy-matching threshold (0.75) was frozen without tuning because the 307-image development split remains unannotated, and the negative effect of the semantic filter means the reported full-pipeline accuracy (42.65%) is lower than the leave-one-out configuration without that filter (46.14%). Second, in plate segmentation, class remapping from 104 FoodSeg103 and 102 UEC categories into five *Isi Piringku* groups assigned 24.0% of FoodSeg103 classes to the catch-all `Others` group (explaining its lower AP of 51.24%), training data exhibited a 3.04× instance-level and 4.12× image-level class imbalance without resampling, label remapping was performed by a single author without inter-annotator κ, internal validation used a single seed, and only a five-epoch screening pilot was available for alternative YOLO architectures because full-budget retraining was constrained by the CPU-only local PyTorch environment. Third, plate analysis computes 2D visual polygon area ratios rather than 3D volume or mass (grams); without depth sensing, fiducial scale markers, or weighed-food ground truth, camera tilt, occlusion, and pile height introduce geometric uncertainty, which is why 2D area proportions are strictly restricted to qualitative plate-balance feedback and never converted automatically into nutrient intake. Fourth, end-to-end propagation across the 131 verified images (786 decisions) showed that OCR errors changed the clinical status in 15.0% of all decisions (21.4% on table-bearing images) and converted 18 of 25 ground-truth DANGER decisions into false-SAFE outcomes when missing fields defaulted to zero, while 76.3% of images required at least one human field edit (median 3 edits/image) — proving empirically why the decision layer must never run on unverified OCR outputs. Fifth, on CPU-only hardware, while YOLO11s-seg inference takes 0.130 s, the full OCR pipeline requires a median of 20.58 s and 3.76 GB RSS, limiting real-time responsiveness without GPU acceleration or model quantisation. Finally, no registered-dietitian review was performed (so clinical rules are verified functionally rather than clinically), the usability evaluation relied on a two-evaluator severity panel rather than end users, and no public Indonesian plated-meal segmentation benchmark exists for quantitative local-cuisine evaluation.

## CONCLUSIONS AND SUGGESTIONS

### Conclusion

This research developed and evaluated Healthier as an integrated, edge-oriented nutrition-assistance system that combines YOLO11s-seg plate segmentation, PaddleOCR (PP-OCRv4) nutrition-table extraction, a deterministic Clinical Rules Engine, and HITL verification. On internal validation the segmentation model reached a mask mAP50 of **71.66%** (mAP50-95 **56.91%**). Across external sources it retained a mask mAP50 of **69.15%** (UECFoodPIXCOMPLETE) and **63.66%** (FoodSeg103) but dropped to **17.06%** on the cross-domain FruitSeg30 set, a quantified domain shift reported as a limitation rather than concealed. For nutrition labels, a manually verified 131-image test set across five product categories showed that the full pipeline reached **42.65%** field-level accuracy against 35.11% for raw PaddleOCR and 14.89% for Tesseract, with grid reconstruction contributing **+13.05 points** to an otherwise identical configuration; the cumulative ablation did not improve monotonically, and the semantic filter proved harmful (46.14% without it, +3.49 points, p = 8.8e-4), which is reported as a negative finding rather than concealed. In HITL workload evaluation, 42.6% of fields (232/544) were accepted unchanged while 76.3% of images required at least one correction (423 total edits, median 3 edits/image). The deterministic rule layer matched its specification on all 66 checks (36/36 daily-limit values and 30/30 status decisions), while end-to-end propagation across the 131 verified images (786 decisions) showed that extraction errors changed the clinical status in 15.0% of all decisions (21.4% on table-bearing images) and converted 18 of 25 ground-truth DANGER decisions into false-SAFE outcomes under zero-imputation, justifying the mandatory HITL verification step. On CPU, YOLO11s-seg segmentation ran in a median of 0.130 s (770.0 MB peak RSS, 20.5 MB `.pt`), whereas the full OCR pipeline required 20.58 s (3,760.8 MB peak RSS, 20.7 MB total OCR weights). A Nielsen heuristic evaluation found no severity-4 problems, with remaining findings at severity 2–3. Within the documented scope boundaries — category-level rather than brand-level OCR stratification, 38 unreadable fields excluded, 2D visual plate area proportions separated from quantitative gram/calorie estimation, a catch-all `Others` class (24.0% of source classes), and a screening-only segmentation baseline — the study demonstrates a transparent, reproducible mechanism for converting uncertain visual and OCR predictions into verified, guideline-aligned feedback.

### Suggestion

Four directions are recommended for future work. First, data coverage and baselines: build a public Indonesian plated-meal segmentation benchmark (including soups and texturally complex dishes) with multi-annotator inter-rater reliability (Cohen's κ), and train a matched 100-epoch alternative segmentation architecture on GPU to complement the five-epoch screening baseline. Second, cross-domain generalisation and portion mass validation: improve cross-source robustness and move beyond 2D visual area ratios by incorporating depth sensing, fiducial scale markers, or multi-angle imaging validated against weighed food portions (grams). Third, OCR refinement and brand stratification: augment the 131-image five-category test set with verified brand-level annotations, annotate the 307-image development split to enable out-of-sample hyperparameter tuning (such as calibrating the fuzzy-matching threshold and removing or redesigning the semantic filter), and complete leave-one-out ablations for the image-level preprocessing stages. Fourth, edge acceleration and clinical validation: apply model quantisation and GPU/NPU acceleration to reduce the 20.58 s CPU OCR latency to interactive speeds, consider an optional offline small language model strictly as a subordinate explanation layer that never overrides deterministic rules or human verification, and conduct prospective studies with end users and registered dietitians to validate clinical utility.

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

Zhou, Y., Zuo, S., Yang, Z., He, J., Shi, J., & Zhang, R. (2023). A review of document image enhancement based on document degradation problem. Applied Sciences, 13(13), 7855. https://doi.org/10.3390/app13137855

Ultralytics. (2024). YOLO11 [Computer software]. https://github.com/ultralytics/ultralytics

Wang, W., Min, W., Li, T., Dong, X., Li, H., & Jiang, S. (2022). A review on vision-based analysis for automatic dietary assessment. Trends in Food Science & Technology, 122, 1–12.

Wang, X., Tang, Z., Guo, J., Meng, T., Wang, C., Wang, T., & Jia, W. (2025). Empowering edge intelligence: A comprehensive survey on on-device AI models. ACM Computing Surveys.

Wu, X., Fu, X., Liu, Y., Lim, E.-P., Hoi, S. C. H., & Sun, Q. (2021). A large-scale benchmark for food image segmentation. In Proceedings of the 29th ACM International Conference on Multimedia (pp. 506–515). https://doi.org/10.1145/3474085.3475201
