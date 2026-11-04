# Response to Reviewers (template)

Manuscript: **Application of Deep Learning for Nutritional Segmentation and
Information Extraction in the Healthier Application**

Gunakan tabel ini untuk menjawab **setiap** komentar reviewer. Jangan menjawab
dengan “already addressed”; tunjukkan lokasi section/tabel/gambar dan angka baru.

---

## Summary of major changes

1. **Contributions:** We now state three explicit contributions: (i) a
   guideline-grounded integration mechanism between plate segmentation and
   package-label OCR, (ii) a reproducible edge OCR pipeline with quantified
   stage-wise ablation and human-in-the-loop correction, and (iii) an
   independent-source generalization and failure analysis.
2. **Reproducibility:** We rebuilt the dataset pipeline, fixed all splits, and
   released the configuration, seeds, commands, and evaluation scripts.
3. **Dataset consistency:** We reconciled all dataset counts and added a
   train/validation/test table with per-class distribution and annotation
   protocol.
4. **Independent test:** We added a held-out test set that was never used for
   training or model selection, plus an external-source test.
5. **Error analysis:** We added confusion matrices, per-class metrics, and an
   OCR error taxonomy.
6. **Ablation:** We added a stage-wise OCR preprocessing ablation.
7. **Baselines:** We compared against [Tesseract / PaddleOCR raw / YOLO11n-seg /
   prior work], using the same test set and metrics.
8. **Corrections:** We corrected the following inconsistencies from the
   original submission: [list].

> **Honesty note:** If a number from the original submission could not be
> reproduced, say so explicitly and report the reproducible run instead. Do not
> silently replace numbers without explanation.

---

## Reviewer comment 1 — 2–3 explicit contributions

**Comment:** “Penulis perlu merumuskan 2–3 kontribusi utama secara eksplisit…”

**Response:**
We agree. The revised manuscript now lists three contributions in the
Introduction (end of Section 1) and revisits them in the Conclusion:

- **C1 — Integration mechanism:** [one sentence]. Evidence: Table X (end-to-end
  evaluation), Table Y (unimodal vs integrated), Section 4.Z.
- **C2 — Reproducible OCR + HITL:** [one sentence]. Evidence: Table Z
  (stage-wise ablation), Table W (HITL correction gain), Section 4.Q.
- **C3 — Generalization/failure analysis:** [one sentence]. Evidence:
  Table V (independent-source test), Figure U (confusion matrix/error taxonomy),
  Section 5.

**Location:** Introduction, Section 4, Section 5, Conclusion.

---

## Reviewer comment 2 — Reproducibility, parameters, equations, flow diagram, ablation

**Comment:** “Metodologi perlu dibuat lebih reproducible…”

**Response:**
We added:
- A complete dataset/annotation/split protocol (Section 3.1–3.3, Table 1).
- Full hyperparameters and hardware/software versions (Table 2).
- Equations for IoU, precision, recall, F1, mAP, bootstrap CI, Levenshtein
  similarity, BMR, TDEE, and clinical thresholds (Eq. 1–X).
- A system flow diagram (Figure 1).
- A stage-wise OCR ablation (Table X).
- A reproducibility appendix with exact commands, seeds, and repository link.

**Location:** Section 3, Section 4, Appendix A.

---

## Reviewer comment 3 — Dataset consistency, annotation, split, independent test, OCR expansion

**Comment:** “Penulis perlu memperjelas konsistensi jumlah dataset, distribusi
kelas, proses anotasi, dan strategi pembagian data….”

**Response:**
- We reconciled the counts: `N_seg = ...`, `N_ocr = ...`, `N_total = ...`.
- Table 1 now reports per-class distribution for train/val/test.
- Annotation protocol: [who, tool, guidelines, adjudication].
- Inter-annotator agreement: Cohen’s κ = ... on ...% of the data.
- Split: stratified by class and source; deduplicated with perceptual hash;
  test set frozen before training.
- Independent test: [external source], never used for training or model
  selection.
- OCR test set expanded from 30 to **N** images from **M** brands/stores.

**Location:** Section 3.1–3.3, Table 1, Table 3, Section 4.

---

## Reviewer comment 4 — Confusion matrix, per-class errors, baselines, OCR ablation, generalization

**Comment:** “Perlu Confusion matrix dan analisis kesalahan setiap kelas,
perbandingan dengan metode baseline, ablation study pada preprocessing OCR, dan
pengujian generalisasi….”

**Response:**
- Confusion matrix: Figure X (segmentation, test set), Figure Y (OCR
  field-level error matrix).
- Per-class analysis: Table X (P/R/mAP per class), Section 5.1.
- Error taxonomy: Table Y + Figure Z (glare, curved packaging, small font,
  multi-column, serving-size format, per-100g vs per-serving).
- Baselines: Table W.
- OCR ablation: Table V (raw → +deskew → +deglare → +CLAHE → +ROI → +semantic →
  +fuzzy → full).
- Generalization: Table U (in-domain vs external-source test), with discussion
  of performance drop.

**Location:** Section 5.1–5.4, Tables V–Y, Figures X–Z.

---

## Changes to numbers (transparency table)

| Item | Original | Revised | Reason |
| :--- | ---: | ---: | :--- |
| Training images | 19,087 | ... | original notebook used 4,840 unique / 12,082 oversampled |
| Validation images | 3,529 | ... | original notebook used 960 |
| Epochs | 50 | ... | original: max 100, early stop 80 |
| Training hardware | RTX 3050 6 GB | ... | notebook ran on Kaggle Tesla T4 15.6 GB |
| Class imbalance method | “weighted loss” | ... | notebook used oversampling |
| OCR version | PP-OCRv5 | ... | code default was PP-OCRv4 |
| OCR test images | 30 | ... | expanded test set |
| Field accuracy | 83.8% | ... | rerun with expanded test set |
| 95% CI | [78.6; 87.9] | ... | cluster bootstrap by image |
| FP16 model | claimed | ... | FP16 export failed; artifact is FP32 |
| P/R 72.07/60.46 | instance metric | ... | notebook values are pixel-level CM |

> Every revised number must point to a file in `experiments/results/`.

---

## Reproducibility statement

```text
Code: <repository URL>
Commit: <git SHA>
Dataset splits: experiments/dataset/split.csv
Training config: experiments/runs/<run>/experiment_config.json
Evaluation: experiments/results/
Environment: experiments/results/pip_freeze.txt
Hardware: <GPU, VRAM, CPU, RAM>
```

We confirm that the independent test set was not used for training, early
stopping, threshold selection, or ablation design.
