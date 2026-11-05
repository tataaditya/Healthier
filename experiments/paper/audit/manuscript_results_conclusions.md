# Healthier — Abstract, Results & Discussion, Conclusions (fact-checked draft)

> Companion to `experiments/paper/audit/manuscript_intro_methods.md` (Introduction & Research
> Methods) and `experiments/paper/references.bib`. Every number traces to
> `experiments/paper/final_numbers.md` or `experiments/results/`.
>
> Markers: `[TO COMPLETE]` = experiment not finished (do not invent a number);
> `[TABLE n / FIGURE n — take from …]` = ready asset; `[VERIFY]` = confirm in Mendeley.
>
> **Deliberately excluded** from this draft (unsupported / not ours): the old 83.8%
> OCR accuracy, the 168 ms/image latency, FP16 export, RTX 3050 training, "weighted
> loss", and PP-OCRv5. The public cloud-VLM comparison is future work only (n=6) and
> is not a main result.

---

# ABSTRACT

Food-related non-communicable diseases remain a major health burden in Indonesia, yet
many users struggle to interpret everyday nutrition information. This study develops and
evaluates **Healthier**, a nutrition-assistance application that combines plate-image
segmentation, nutrition-table extraction from packaging, and a deterministic clinical
rules engine with explicit user verification. Plate images are segmented into five *Isi
Piringku* groups (Staple Food, Side Dishes, Vegetables, Fruits, Others) with a
YOLO11s-seg model trained on 11,684 images and validated on 2,017 images drawn from
public datasets that are not specific to Indonesian cuisine. Three external test sets
(5,013 images) that were never used for training or model selection were used to measure
generalization. On internal validation the model reached a mask mAP50 of 71.66% and a
mask mAP50-95 of 56.91% (precision 72.95%, recall 65.50%). It transferred well to two
food-photography sources (mask mAP50 69.15% on UECFoodPIXCOMPLETE and 63.66% on
FoodSeg103) but degraded sharply on a fruit-only, in-the-wild dataset (17.06%),
confirming a domain shift that is reported rather than hidden. Nutrition-label
extraction uses PaddleOCR (PP-OCRv4) with a grid-based table parser; an exploratory
six-image pilot raised field-level accuracy from 21.74% to 47.83%, so the OCR evaluation
is reported as a pilot pending a larger brand-stratified set. A deterministic rules
engine translates verified values into SAFE/WARNING/DANGER feedback aligned with
PERKENI 2024 and hypertension guidance, and a Human-in-the-Loop layer lets users correct
segmentation and extracted values. The result is a transparent, edge-oriented
decision-support pipeline whose limitations, especially cross-source generalization and
small-sample OCR evaluation, are stated explicitly.

**Keywords:** Instance Segmentation; OCR; YOLO11s-seg; PaddleOCR; Nutrition Literacy

# ABSTRAK

Penyakit tidak menular yang berkaitan dengan pola makan masih menjadi beban kesehatan
utama di Indonesia, namun banyak pengguna masih sulit memahami informasi gizi harian.
Penelitian ini mengembangkan dan mengevaluasi **Healthier**, aplikasi bantuan gizi yang
menggabungkan segmentasi citra piring, ekstraksi tabel gizi dari kemasan, dan mesin
aturan klinis deterministik dengan verifikasi pengguna. Citra piring disegmentasi ke
dalam lima kelompok *Isi Piringku* menggunakan model YOLO11s-seg yang dilatih pada
11.684 citra dan divalidasi pada 2.017 citra dari dataset publik yang tidak spesifik
masakan Indonesia. Tiga set uji eksternal (5.013 citra) yang tidak pernah dipakai untuk
pelatihan atau pemilihan model digunakan untuk mengukur generalisasi. Pada validasi
internal, model mencapai mask mAP50 71,66% dan mask mAP50-95 56,91% (presisi 72,95%,
recall 65,50%). Model bertransfer baik ke dua sumber fotografi makanan (mask mAP50
69,15% pada UECFoodPIXCOMPLETE dan 63,66% pada FoodSeg103) tetapi turun tajam pada
dataset khusus buah di lingkungan alami (17,06%), menegaskan *domain shift* yang
dilaporkan secara terbuka. Ekstraksi label gizi memakai PaddleOCR (PP-OCRv4) dengan
parser tabel berbasis grid; pilot enam citra menaikkan akurasi tingkat field dari 21,74%
menjadi 47,83%, sehingga evaluasi OCR dilaporkan sebagai pilot yang menunggu set uji
lebih besar. Mesin aturan deterministik menerjemahkan nilai terverifikasi menjadi status
SAFE/WARNING/DANGER sesuai PERKENI 2024 dan pedoman hipertensi, dan lapisan
Human-in-the-Loop memungkinkan pengguna mengoreksi hasil segmentasi dan nilai
terekstraksi. Hasilnya adalah pipeline bantuan keputusan yang transparan dan berorientasi
edge, dengan keterbatasan—terutama generalisasi antar-sumber dan evaluasi OCR sampel
kecil—yang dinyatakan secara eksplisit.

**Kata kunci:** Segmentasi Instan; OCR; YOLO11s-seg; PaddleOCR; Literasi Gizi

---

# RESULTS AND DISCUSSION

This section reports the quantitative evaluation of the two recognition modules
(instance segmentation and OCR text extraction), their behaviour under domain shift, and
the integration of their outputs through the deterministic Clinical Rules Engine, the
Human-in-the-Loop layer, and the usability assessment. Segmentation and OCR were tested
separately before being examined together in the integrated pipeline.

## Instance Segmentation on Internal Validation

On the 2,017-image internal validation split (best checkpoint, epoch 48), the YOLO11s-seg
model reached a mask mAP50 of **71.66%** and a mask mAP50-95 of **56.91%**, with mask
precision **72.95%** and recall **65.50%**. Box metrics were slightly higher (box mAP50
**72.37%**, box mAP50-95 **60.11%**; box precision 72.87%, box recall 66.04%). The gap
between precision and recall is expected in instance segmentation of plated food, where
overlapping and heterogeneous regions make some instances easier to localize than to
recall fully; this recall gap is one reason the system keeps an interactive correction
step (Section "Human-in-the-Loop and Usability").

`[TABLE 3 — Internal validation metrics: `tables/table_segmentation_internal.md`.]`
`[FIGURE 3 — Training curves: `experiments/runs/seg_main/results.png`.]`
`[FIGURE 4 — Internal normalized confusion matrix:
`experiments/runs/seg_main/confusion_matrix_normalized.png`.]`
`[FIGURE 5 — Internal qualitative GT vs prediction:
`experiments/runs/seg_main/val_batch0_labels.jpg` + `val_batch0_pred.jpg`.]`

## Generalization to External Sources

Because the training data are not Indonesian-specific, external evaluation is the central
generalization test. The model was applied unchanged to three held-out sources. Mask
mAP50 was **69.15%** on UECFoodPIXCOMPLETE test and **63.66%** on the FoodSeg103
validation set — a drop of only a few points relative to internal validation — but fell
to **17.06%** on FruitSeg30 (Table 4). Since the training and evaluation pipelines,
classes, and metrics were identical across the three sources, the drop isolates a
**domain shift** rather than a training error.

`[TABLE 4 — External test metrics: `tables/table_segmentation_external.md`.]`
`[FIGURE 6 — External normalized confusion matrices:
`experiments/results/segmentation/{uec_test_instance,foodseg_val_instance,fruitseg_instance}/confusion_matrix_normalized.png`.]`
`[FIGURE 7 — External mask PR / F1 curves: each set's `MaskPR_curve.png` and
`MaskF1_curve.png`.]`

**Per-class behaviour.** Class-level average precision shows where the model succeeds
and fails (Table 5). On UECFoodPIXCOMPLETE the strongest class was Staple Food (mask
AP50 **85.70%**) and the weakest was Vegetables (**55.42%**); on FoodSeg103 the strongest
was Fruits (**74.96%**) and the weakest was the heterogeneous Others class (**51.24%**).
On FruitSeg30 the only class present is Fruits, which scored **17.29%**. The pixel-level
view sharpens the failure: on FruitSeg30 the Fruits class had pixel precision 97.17% but
recall only **2.14%** (F1 **4.19%**), i.e. the few pixels it labelled as fruit were mostly
correct, but it recovered almost none of the fruit in the image and instead assigned fruit
regions to other groups (Table "pixel-level"). This is consistent with a genuine
cross-dataset failure rather than a metric artefact.

`[TABLE 5 — Per-class external AP: `tables/table_segmentation_per_class.md`.]`
`[FIGURE 8 — Per-class external mask AP50 bar chart (optional): build from
`experiments/results/segmentation/per_class_all.csv`.]`

The pattern is interpretable: UEC and FoodSeg103 contain plated meals and kitchen food
photographs that share low-level statistics with the training pool, whereas FruitSeg30
contains isolated fruit in natural, uncontrolled environments (Shamrat et al., 2024) — a
different visual domain that the model was never trained on, especially because the
Fruits class in training comes only from FoodSeg103. We therefore treat the FruitSeg30
result as an explicit limitation and as evidence for the generalization/failure-analysis
contribution, not as a headline strength.

## OCR Extraction from Nutrition Tables

The OCR module was evaluated on Indonesian snack-label photographs with hand-transcribed
ground truth, using field-level accuracy (a field is correct when it is exact or within
5% of the per-serving value). This evaluation is an **exploratory pilot on six images
(46 fields)**; it is reported to document the effect of the layout parser, not as a
generalization result. The baseline parser achieved **21.74%** field-level accuracy (95%
cluster-bootstrap CI [16.67%, 25.58%]), and the revised grid parser achieved **47.83%**
(95% CI [29.17%, 66.67%]) — more than double, with missing fields falling from 65.22% to
34.78% of the reference fields (Table 6).

`[TABLE 6 — OCR field-level pilot: `tables/table_ocr.md`.]`

**Error analysis.** The dominant failure mode of the baseline was structural: PP-OCR
merges tall numeric boxes across adjacent table rows and columns, so values were assigned
to the wrong row (for example, gula read as 3 instead of 10, and 9 instead of 5 on
different images). The revised parser addresses this by detecting the value columns
relative to the row labels and by splitting merged numeric boxes, which improved the
sodium and energy fields most. Residual errors concentrate on serving-size formatting and
on distinguishing per-serving from per-100 g values. These are the error classes the
per-stage ablation is intended to quantify.

`[TO COMPLETE — per-stage ablation: raw → +deskew → +deglare → +CLAHE → +ROI → +semantic
→ +fuzzy → full, using `ablate_ocr.py`; and the ≥100-image brand-stratified evaluation
that replaces the n=6 pilot.]`

`[FIGURE 9 — OCR pipeline diagram (redraw from `docs/architecture.md`); FIGURE 10 — OCR
qualitative example from `experiments/results/ocr_vlm/gt_audit/*.png`.]`

## Clinical Rules Engine and Human-in-the-Loop Integration

The Clinical Rules Engine computes a user's TDEE from BMR and activity factor, derives
personal limits (sugar, sodium, saturated fat, cholesterol, protein), and classifies
accumulated intake as SAFE, WARNING, or DANGER using the thresholds implemented in
`frontend/js/clinical-rules.js` (WARNING at sugar/saturated fat >50%, sodium >40%,
nephropathy protein >30%, sodium density >600/800 mg per 100 g; DANGER only when sugar
>100% of the daily limit). Because no registered-dietitian review was performed, these
results are reported as **rule verification** rather than clinical validation.

`[TO COMPLETE — 10–20 profile × food scenarios with expected SAFE/WARNING/DANGER status,
compared against manual calculation; and the end-to-end error-propagation study that
measures how segmentation and OCR errors affect the final status.]`

**Human-in-the-Loop.** Healthier exposes a Segmentation Editor for correcting predicted
polygons and a Nutrient Value Verification Form for editing extracted values before the
rules are applied. This is a design response to the recall gap in segmentation and to the
residual OCR errors, and it follows established HITL practice (Mosqueira-Rey et al., 2023;
Yuan et al., 2024).

## Usability Assessment

`[TO COMPLETE — Nielsen heuristic findings per severity (0–4), per evaluator, must come
from the author's own notes. If no data exist, report usability as future work. Do not
retain the phrase "no design flaws".]`

`[TABLE 8 — Evaluator profile: current draft Table 2.]`

`[FIGURE 11 — Clinical decision flow (redraw from `clinical-rules.js`); FIGURE 12 — HITL
UI screenshots (run the app).]`

## Discussion

The results separate cleanly into what generalizes and what does not. Segmentation is
competitive on in-domain plated-food sources and collapses on a cross-domain, fruit-only
source; the external numbers therefore support a claim of **supervised in-domain
performance with a quantified domain shift**, not of universal generalization. This is
the same practical concern raised by recent food-segmentation benchmarks, which report
limited data diversity and degradation under new viewpoints and domains (AlMughrabi et
al., 2026), and it is consistent with the dataset-diversity limitation noted in
dietary-assessment reviews (Chotwanvirat et al., 2024; Shonkoff et al., 2023).

For label reading, the pilot indicates that **layout reconstruction, not raw recognition,
is the main bottleneck** for Indonesian nutrition tables: the same OCR engine gained more
than twenty accuracy points once merged cells and value columns were parsed correctly.
This matches studies showing that dense tabular packaging layouts are a major source of
OCR error (Nagayi et al., 2025) and that preprocessing and layout-aware correction are
necessary in practice (Rosyadi et al., 2024; Saputra et al., 2024, 2026; Alireza &
Kurniawardhani, 2026). The honest framing is that the OCR evidence is currently a pilot
and that the headline number must be re-established on a larger, brand-stratified set.

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
registered-dietitian review was performed, so the clinical layer is verified functionally,
not clinically. (5) No public Indonesian plate-segmentation benchmark exists, so
Indonesian-specific generalization could not be measured quantitatively.

---

# CONCLUSIONS AND SUGGESTIONS

## Conclusion

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
the heterogeneous Others class remains the weakest category. These limitations align with
challenges identified in recent systematic reviews of AI-based dietary assessment
(Dalakleidi et al., 2022; Shonkoff et al., 2023).

The deterministic rule layer and the Human-in-the-Loop controls provide a transparent
mechanism for converting uncertain model outputs into guideline-aligned feedback that the
user can correct before it is used. The system therefore demonstrates the technical
feasibility of integrating visual food analysis and packaged-food label extraction in a
single local workflow, while making its generalization boundaries explicit.

## Suggestions and Future Work

Future work should address four directions. First, **data coverage**: build an Indonesian
plate-segmentation and nutrition-label corpus, including soups and texturally complex
foods, since no public Indonesian plated-meal segmentation benchmark currently exists.
Second, **generalization and estimation**: improve cross-source robustness and estimate
portions more reliably than the current 2D area ratio, for example with depth or
multi-angle imaging. Third, **OCR scale and rigor**: expand the label evaluation to at
least 100 brand-stratified images and complete the per-stage preprocessing ablation.
Fourth, **deployment**: apply quantization and further edge optimization, and consider an
optional offline small language model for explanation or normalization — but only as a
subordinate component that never overrides the deterministic nutrition rules or the
explicit human verification. Prospective studies with real users and nutrition
professionals are needed to validate the clinical usefulness of the feedback.
