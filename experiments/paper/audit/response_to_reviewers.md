# Point-by-Point Response to Reviewers — Round 2 Major Revision

**Manuscript Title:** Application of Deep Learning for Nutritional Segmentation and Information Extraction in the Healthier Application  
**Authors:** Tata Aditya Pamungkas, Nuke Lu'lu Ul Chusna, Harry Dwiyana Kartika  
**Journal:** Jurnal Riset Informatika (JRI)  
**Revision Date:** 8 November 2026 (Follow-up to 5 November 2026 Revision / Rekap No. 310, OJS Check ID 653)

---

## Ringkasan Eksekutif untuk Editor & Reviewer (Bahasa Indonesia)

Terima kasih atas pemeriksaan menyeluruh pada revisi tanggal 5 November 2026 (Rekap No. 310 / Cek Reviewer ID 653 terhadap 4 poin Reviewer A dan 13 komentar Reviewer B). Pada revisi lanjutan ini, kami telah menyelesaikan seluruh 8 butir catatan yang masih terbuka, menyinkronkan seluruh angka antara naskah (`manuscript_jri.md` dan `manuscript_jri.docx`) dengan laporan eksperimen terverifikasi (`experiments/results/`), serta mendokumentasikan secara jujur batasan perangkat keras pada poin yang tidak dapat dieksekusi ulang secara penuh.

| No | Butir Evaluasi Ronde 2 (ID 653) | Status & Tindakan pada Naskah Revisi | Lokasi di Naskah Baru |
| :- | :--- | :--- | :--- |
| **1** | **Dataset OCR $\ge 100$ citra + prosedur GT** (A3/B6) | **Selesai dievaluasi pada 131 citra terverifikasi** (544 field terbaca dari 582 field tercetak; 38 *unreadable* dikeluarkan dari denominator; 60 citra tanpa tabel) di **5 kategori produk** (dari 438 citra unik setelah deduplikasi MD5+dHash, *locked split* `seed=42`). Seluruh angka pilot 6 citra telah digantikan oleh hasil 131 citra (Konfigurasi A–E, ablasi kumulatif, dan *leave-one-out*). Prosedur GT (*Per Sajian*, pengodean `absent` vs `unreadable`, dan 177 koreksi manusia pada 36 citra) dijelaskan rinci. Karena atribut merek tidak dicatat sebagai field, stratifikasi dinyatakan per kategori produk dan ketiadaan stratifikasi merek ditulis sebagai keterbatasan eksplisit. | *Abstract*, *Abstrak*, *Research Target / Subject*, *Data, Instruments, and Data Collection Techniques*, *Data Analysis Technique*, **Tabel 10**, **Tabel 11**, *Discussion*, *Conclusion* |
| **2** | **Baseline segmentasi yang sebanding (full-budget)** (A4) | **Dilaporkan sebagai keterbatasan eksplisit (Opsi C).** Pelatihan model utama YOLO11s-seg (100 epoch, 11.684 citra) memerlukan waktu 4,01 jam pada GPU cloud Kaggle Tesla T4, sedangkan lingkungan lokal menggunakan build PyTorch CPU-only (`torch 2.15.0+cpu`) dan tidak menyimpan ribuan citra mentah dataset publik secara lokal. Hasil *screening* 5-epoch untuk tiga arsitektur pembanding (YOLOv8m-seg 54,67%, YOLO11m-seg 52,93%, YOLO26m-seg 53,45% mask mAP50) dipertahankan murni sebagai *exploratory screening* yang tidak sebanding (*non-comparable*) dan tidak digunakan untuk klaim keunggulan arsitektur. | *Ablation and Baseline Design*, *Baseline Comparison*, *Discussion*, *Conclusion*, *Suggestion* |
| **3** | **Validasi proporsi area poligon vs massa/porsi + ketidakpastian tanpa depth** (B3/B11) | **Dipisahkan secara tegas di Metodologi & dibahas ketidakpastian geometrisnya di Diskusi.** Persamaan (1) kini didefinisikan secara eksplisit sebagai **proporsi area visual 2D (*2D visual area proportion*)** untuk panduan keseimbangan visual piring (*Isi Piringku*), **bukan** estimasi massa (gram) maupun kalori. Sistem tidak pernah mengonversi piksel poligon menjadi gram/kkal secara otomatis; *Clinical Rules Engine* hanya memproses nilai gizi terverifikasi. Sumber ketidakpastian proyeksi 3D ke 2D tanpa sensor *depth* (kemiringan kamera/*perspective foreshortening*, oklusi antar-lauk, variasi tinggi tumpukan seperti nasi vs sayuran pipih, dan ketiadaan tabel densitas) dijelaskan secara eksplisit sebagai keterbatasan. | *Plate Proportion Estimation* (Eq. 1), *Discussion*, *Conclusion*, *Suggestion* |
| **4** | **Bukti kuantitatif manfaat koreksi manusia (HITL)** (B5) | **Selesai dihitung pada 131 citra terverifikasi.** Dari 544 field pada pipeline penuh (E), **232 field (42,6%)** diterima tanpa perubahan, namun diperlukan total **423 aksi edit** (98 koreksi nilai salah, 213 pengisian nilai hilang, dan 111 penghapusan nilai berlebih/*spurious* pada field `absent`), dengan **100 dari 131 citra (76,3%)** membutuhkan $\ge 1$ koreksi (median **3 edit/citra**, rata-rata **3,23**, maksimum **10**). Pada propagasi keputusan klinis (**786 keputusan** = 131 citra $\times$ 6 profil), galat OCR mengubah status pada **15,0%** seluruh keputusan (**21,4%** pada 71 citra bertabel) dan mengubah **18 dari 25 keputusan DANGER menjadi false-SAFE (72%)** akibat nilai hilang yang bernilai bawaan nol. | *Abstract*, *Abstrak*, *Nutrition-Label Extraction and Preprocessing Ablation*, *Clinical Rules Engine and HITL Integration*, **Tabel 13**, *Discussion*, *Conclusion* |
| **5** | **Pengukuran deployment edge: latensi, memori, ukuran model** (B10) | **Selesai diukur pada CPU target dan ditambahkan sebagai sub-bagian khusus & Tabel 15.** Pada CPU, segmentasi YOLO11s-seg ($n=30$) mencatat latensi median **0,1304 s** (P95 **0,1854 s**, RAM puncak **770,0 MB**), *Clinical Rules Engine* **0,039 ms**/panggilan, sedangkan pipeline penuh OCR ($n=12$, PP-OCRv4 + SLANet_plus + grid) mencatat median **20,5777 s** (P95 **45,2932 s**, RAM puncak **3.760,8 MB**), dengan total sekuensial **20,7081 s**. Ukuran model di disk: YOLO11s-seg `.pt` **20,5 MB** (`.onnx` **40,6 MB**), PP-OCRv4 det **4,9 MB**, rec **7,8 MB**, SLANet_plus **8,0 MB** (total OCR **20,7 MB**). Alasan tidak adanya klaim latensi GPU lokal (`torch.cuda.is_available() = False`) dinyatakan secara eksplisit. | *Data Analysis Technique*, *Edge Deployment Benchmark: Latency, Memory, and Artifact Size*, **Tabel 15**, *Discussion*, *Conclusion*, *Suggestion* |
| **6** | **Audit anotasi/remapping kelas & analisis class imbalance** (A3/B1) | **Selesai diaudit dan dilaporkan di Metodologi & Diskusi.** Pemetaan 104 kelas FoodSeg103 (12 Staple, 12 Side, 34 Vegetables, 21 Fruits, 25 Others) dan 102 ID UEC (38 Staple, 42 Side, 11 Vegetables, 0 Fruits, 11 Others) ke 5 grup *Isi Piringku* menempatkan **24,0% kelas FoodSeg103 (25/104 kelas)** ke grup *catch-all* `Others`, yang menjelaskan secara struktural mengapa `Others` memiliki AP terendah (51,24%). Rasio *class imbalance* pada *train split* adalah **3,04×** (instan: `Vegetables` 28,0% vs `Fruits` 9,2%) dan **4,12×** (citra), serta 3,37× pada UEC test (tanpa `Fruits`), 2,90× pada FoodSeg103 val, dan 1,00× pada FruitSeg30 (100% `Fruits`). Ketiadaan $\kappa$ antar-anotator (1 pemeta) dinyatakan sebagai keterbatasan. | *Dataset Construction, Annotation, and Splits*, **Tabel 1**, **Tabel 2**, *Discussion*, *Conclusion* |
| **7** | **Referensi PERKENI 2024, James et al., YOLO11 (Khanam & Hussain) + reference implementation aturan klinis** (B4/B7) | **Selesai dan terverifikasi di naskah & `.docx`.** Ketiga referensi (PERKENI, 2024; James et al., 2014; Khanam & Hussain, 2024) dikutip di teks *Methods* dan tercantum lengkap di daftar `REFERENCES`. Rujukan eksplisit ke implementasi acuan aturan klinis (`frontend/js/clinical-rules.js`) dan skrip verifikasi independen (`experiments/scripts/verify_clinical_rules.js`) telah ditambahkan di teks. | *Segmentation Model and Training*, *Clinical Rules Engine*, **Tabel 4**, **Tabel 5**, **Tabel 12**, *REFERENCES* |
| **8** | **Inkonsistensi minor: jumlah evaluator (4 vs 2), penomoran tabel/gambar, PP-OCRv5→v4, dan surat point-by-point** | **Selesai diperbaiki seluruhnya.** (a) Peran 4 evaluator (E1–E2 independen untuk pembobotan *severity*, E3 review arsitektur asinkron, E4 konsistensi dokumentasi–implementasi di luar pembobotan *severity*) diseragamkan di *Methods* dan *Usability Assessment* (**Tabel 14**). (b) Penomoran seluruh **Tabel 1–15** dan **Gambar 1–10** beserta rujukan di teks telah diverifikasi berurutan tanpa loncatan atau typo. (c) Perubahan penyebutan PP-OCRv5 (kesalahan dokumentasi draf awal) ke **PP-OCRv4** (`NUTRILABEL_OCR_VERSION="PP-OCRv4"` + `SLANet_plus`) dijelaskan secara eksplisit di *Methods*. (d) Surat tanggapan *point-by-point* ini dilampirkan. | *Nutrition-Label Extraction Pipeline*, *Data Analysis Technique*, *Usability Assessment* (**Tabel 14**), seluruh **Tabel 1–15** & **Gambar 1–10** |

---

## Detailed Point-by-Point Response (English)

### Item 1 — Expanded OCR Dataset ($\ge 100$ Images), Stratification Scope, Ground-Truth Protocol, and Negative Ablation Finding (Reviewer A3 / Reviewer B6)

**Reviewer / Editor Comment:**  
*The OCR dataset has not yet been expanded — the previous revision only reported a 6-image pilot (46 fields); a more representative evaluation (at least 100 images, as proposed by the authors) and a clear ground-truth annotation procedure are required (A3/B6).*

**Authors' Response:**  
We have replaced the 6-image pilot throughout the manuscript with a locked, human-verified evaluation on **131 Indonesian packaged-food photographs** drawn from a deduplicated pool of **438 unique images** (453 raw candidates minus 15 MD5/dHash duplicates; locked split `seed=42`, SHA-256 manifest `a0c7a4942526120b...`, with 131 test images and 307 unannotated development images, test $\cap$ dev $= \emptyset$).

1. **Product-Category Coverage and Honest Scope on Brand Stratification:**  
   The 131 verified test images span five product categories collected from local stores: *Makanan Ringan* (53 images), *Saos dan Perbumbuan* (24), *Frozen Food* (20), *Minuman Sachet* (18), and *Makanan Kaleng* (16). Because brand identity was not recorded as a verified metadata field during annotation, we explicitly report this evaluation as **stratified across five product categories rather than brand-stratified**, and state the absence of brand-level stratification as an explicit limitation in *Research Methods*, *Discussion*, and *Conclusion*.
2. **Documented Ground-Truth Protocol:**  
   As detailed in *Data, Instruments, and Data Collection Techniques* and *Data Analysis Technique* (backed by `experiments/results/ocr_eval/annotation_guideline.md` and `gt_verified_audit.md`), all 131 test records were verified image-by-image by the human annotator in a dedicated browser tool (`viewer.html`) without viewing pipeline predictions:
   - Values were transcribed strictly from the absolute **Per Sajian** (per-serving) column, ignoring `%AKG` / `%DV` percentages.
   - Nutrient rows not printed on a package were coded as `absent` and excluded from accuracy denominators (60 of the 131 photographs are composition-only or front/side packaging without a nutrition table, retained to test whether pipelines hallucinate values on `absent` fields).
   - Printed nutrient rows whose numeric values were visually illegible in the photograph due to blur, packaging creases, or specular glare were coded as `unreadable` (**38 fields across 19 images**, primarily trans fat [13], fibre [11], and cholesterol [10]) and excluded from all denominators rather than imputed.
   - Out of **582 printed field annotations**, **544 legible fields** form the exact denominator for every evaluated configuration. During verification of the visual pre-annotation draft, the human annotator corrected **177 cells across 36 images** (leaving 95 images unchanged) before locking `ground_truth_verified.json`.
3. **Updated Main OCR Comparison (Table 10) and Ablation Findings (Table 11):**  
   - Across the 544 verified fields (**Table 10**), the full pipeline with grid reconstruction (**E**) achieved **42.65%** field-level accuracy (95% cluster-bootstrap CI [36.23%, 49.00%]), significantly outperforming raw PaddleOCR (**C**: **35.11%**, $\Delta = +7.54\text{ pp}$, McNemar $p = 0.0021$), the layout parser without grid reconstruction (**D**: **23.53%**, $\Delta = +19.12\text{ pp}$, $p = 1.38 \times 10^{-17}$), EasyOCR (**B**: **19.67%**, $p = 4.11 \times 10^{-18}$), and Tesseract (**A**: **14.89%**, $p = 1.56 \times 10^{-25}$).
   - In the cumulative and leave-one-out (LOO) ablations (**Table 11**), we replaced the old monotonic pilot narrative with two rigorous empirical findings: (i) **grid/table-structure reconstruction** is the single largest positive contributor, raising field accuracy from **29.60%** (`+ clinical validation` without grid) to **42.65%** (**+13.05 pp**, 95% CI [+8.73%, +17.36%], $p = 1.43 \times 10^{-13}$), followed by fuzzy field matching (**+10.29 pp** in LOO, $p = 1.2 \times 10^{-10}$); and (ii) **the semantic filter is a net loss**, as removing it in LOO raises full-pipeline accuracy from 42.65% to **46.14%** (**+3.49 pp**, 95% CI [+1.08%, +6.23%], McNemar $p = 8.8 \times 10^{-4}$). Rather than retroactively tuning or hiding this stage on the locked test set, we report the semantic filter's negative impact transparently as an empirical finding.

**Location in Revised Manuscript:** *Abstract*, *Abstrak*, *Research Target / Subject*, *Data, Instruments, and Data Collection Techniques*, *Data Analysis Technique*, *Nutrition-Label Extraction and Preprocessing Ablation* (**Table 10** and **Table 11**), *Discussion*, and *Conclusion*.

---

### Item 2 — Comparable Full-Budget Segmentation Baseline (Reviewer A4)

**Reviewer / Editor Comment:**  
*A comparable full-budget segmentation baseline is not yet available — only a 5-epoch screening run that is not comparable was reported (A4).*

**Authors' Response:**  
We acknowledge this limitation transparently and report it as an explicit resource and scope constraint rather than making any claim of architectural superiority. Specifically:
- The primary `YOLO11s-seg` model required **4.01 hours** (14,420.7 s) of training on a cloud **Kaggle Tesla T4 GPU (15.6 GB VRAM)** over 11,684 training and 2,017 validation images (100-epoch budget, early-stopping patience 20, stopped at epoch 67, best checkpoint epoch 48).
- In the local environment available for this revision, PyTorch is installed as a CPU-only build (`torch 2.15.0+cpu`, `torch.cuda.is_available() = False`) and only the polygon label manifests (`experiments/dataset/`) rather than the raw multi-gigabyte training image corpus are stored locally. Consequently, retraining a second architecture under an identical 100-epoch GPU budget could not be executed locally.
- In *Baseline Comparison*, *Discussion*, and *Conclusion*, we retain the 5-epoch screening numbers from the earlier notebook (`YOLOv8m-seg` 54.67%, `YOLO11m-seg` 52.93%, and `YOLO26m-seg` 53.45% mask mAP50) strictly as a non-comparable exploratory screening pilot, explicitly stating that they must not be interpreted as a fair comparison against the 100-epoch `YOLO11s-seg` model (71.66% internal mask mAP50), and we designate a matched 100-epoch GPU baseline as priority future work.

**Location in Revised Manuscript:** *Ablation and Baseline Design*, *Baseline Comparison*, *Discussion*, *Conclusion*, and *Suggestion*.

---

### Item 3 — Polygon Area Proportions vs. Mass/Portion Ground Truth and Uncertainty Without Depth (Reviewer B3 / Reviewer B11)

**Reviewer / Editor Comment:**  
*Validation of polygon area proportions against mass/portion ground truth is not available; clearly distinguish visual area proportions from quantitative nutritional estimation, and discuss/quantify uncertainty resulting from the absence of depth sensing (B3/B11).*

**Authors' Response:**  
We have revised *Plate Proportion Estimation* (Equation 1), *Discussion*, and *Conclusion* to establish a strict conceptual and architectural separation between **2D visual area proportions** and **quantitative mass/nutrient intake**:
1. **Strict Terminology and Functional Separation:**  
   In *Plate Proportion Estimation*, we changed the description of Equation (1) from *"nutritional proportion"* to **2D visual area proportion** ($Proportion_g = A_g / \sum_g A_g$). We explicitly state that $Proportion_g$ is used solely for qualitative visual *Isi Piringku* plate-balance guidance and is **never** automatically converted into mass (grams) or energy (kilocalories). Quantitative risk classification in the *Clinical Rules Engine* operates exclusively on verified nutrient values (from the verified label or user-entered portion fields), never on uncalibrated 2D polygon areas.
2. **Geometric Uncertainty Analysis Without Depth:**  
   We explicitly detail the physical sources of uncertainty when projecting 3D plated food onto a 2D image plane without a depth sensor, fiducial scale marker, or food-specific density lookup table: (i) perspective foreshortening under non-nadir camera angles, (ii) inter-item occlusion among overlapping side dishes and garnishes, and (iii) vertical pile-height and bulk-density differences (e.g., mounded staple rice or dense protein cuts versus flat, low-density leafy vegetables). Because no kitchen-scale weighed-portion dataset was collected in this study, empirical mass validation is stated as an explicit scope limitation and recommended for future work using depth/multi-angle imaging.

**Location in Revised Manuscript:** *Plate Proportion Estimation* (Equation 1), *Discussion*, *Conclusion*, and *Suggestion*.

---

### Item 4 — Quantitative Evidence of Human-in-the-Loop (HITL) Correction Workload and Clinical Decision Impact (Reviewer B5)

**Reviewer / Editor Comment:**  
*Quantitative evidence of the benefit of human correction (HITL) is missing — report correction frequency and accuracy before vs. after correction, as well as clinical decision impact (B5).*

**Authors' Response:**  
We computed both the **field-level HITL correction workload** (`experiments/results/ocr_eval/hitl_E_full_pipeline.md`) and the **end-to-end clinical decision propagation** (`experiments/results/ocr_eval/e2e_propagation_verified.md`) across all **131 human-verified test images**:
1. **HITL Correction Frequency and Edit Workload (n = 131 images, 544 verified fields):**  
   When starting from the full pipeline's output (`E_full_pipeline`, pre-correction field accuracy **42.65%** vs. 100% post-verification by definition):
   - **232 of 544 fields (42.6%)** were accepted unchanged by the user.
   - **423 total edit actions** were required across 655 actionable cells (64.6% edit rate), consisting of **98** wrong-value replacements, **213** missing-value entries, and **111** spurious-value clearances (values emitted by the pipeline on fields marked `absent` in the ground truth, particularly on snack packaging).
   - **100 of 131 images (76.3%)** required at least one field correction, with a **median of 3 edits per image** (mean **3.23**, maximum **10**).
2. **End-to-End Clinical Decision Propagation on 131 Verified Images (Table 13):**  
   Replacing the old 6-product pilot (36 decisions), we evaluated all **131 verified images across 6 clinical profiles = 786 clinical decisions** (including **426 decisions** on the 71 images that contain a printed nutrition table):
   - Raw PaddleOCR (**C**) preserved the ground-truth clinical status (`SAFE`/`WARNING`/`DANGER`) in **600/786 (76.3%)** of all decisions (**284/426 [66.7%]** on table-bearing images).
   - The layout parser without grid (**D**) matched the ground-truth status in **644/786 (81.9%)** of all decisions (**308/426 [72.3%]** on table-bearing images).
   - The full pipeline (**E**) matched the ground-truth status in **668/786 (85.0%)** of all decisions (**335/426 [78.6%]** on table-bearing images) — meaning extraction errors still altered **15.0%** of all clinical decisions (**21.4%** on table-bearing images).
   - Crucially, we highlight a critical safety finding: because missing fields default to zero if passed unverified to the rules engine, the full pipeline converted **18 of 25 ground-truth `DANGER` decisions (72.0%) into false-`SAFE` outcomes** and **55 of 117 `WARNING` decisions into `SAFE`**. This provides rigorous quantitative proof that HITL verification is mandatory to prevent under-warning caused by missing OCR fields.

**Location in Revised Manuscript:** *Abstract*, *Abstrak*, *Nutrition-Label Extraction and Preprocessing Ablation*, *Clinical Rules Engine and Human-in-the-Loop Integration* (**Table 13**), *Discussion*, and *Conclusion*.

---

### Item 5 — Edge Deployment Measurements: End-to-End Latency, Memory Consumption, and Model Size (Reviewer B10)

**Reviewer / Editor Comment:**  
*Edge deployment measurements are missing — report end-to-end latency, memory consumption, and model artifact sizes (B10).*

**Authors' Response:**  
We executed a dedicated local benchmark (`experiments/scripts/bench_edge.py` $\rightarrow$ `experiments/results/edge/cpu/bench.md`) and added a new subsection (*Edge Deployment Benchmark: Latency, Memory, and Artifact Size*) and **Table 15** to the manuscript:
- **YOLO11s-seg Plate Segmentation ($n = 30$ images, 640 px, 3 warm-up runs, CPU):** median latency **0.1304 s** (130.4 ms), P95 **0.1854 s** (185.4 ms), peak RSS memory **770.0 MB**.
- **Full Nutrition-Label OCR Pipeline ($n = 12$ images, PP-OCRv4 + SLANet_plus + grid, CPU):** median latency **20.5777 s** (20.58 s), P95 **45.2932 s** (45.29 s), peak RSS memory **3,760.8 MB**.
- **Clinical Rules Engine (CPU):** median **0.000039 s** (0.039 ms) per evaluation call.
- **Sequential End-to-End Sum (CPU):** **20.7081 s** (20.71 s).
- **On-Disk Model Artifact Sizes:** `YOLO11s-seg` PyTorch checkpoint (`yolo_last_pt`) **20.5 MB**, ONNX export (`yolo_best_onnx`) **40.6 MB**, `PP-OCRv4_mobile_det` **4.9 MB**, `en_PP-OCRv4_mobile_rec` **7.8 MB**, and `SLANet_plus` **8.0 MB** (total OCR weight footprint **20.7 MB**; combined `.pt` + OCR footprint **41.2 MB**).
- **Why Local GPU Latency Is Not Claimed:** We state transparently that the local PyTorch environment is a CPU-only build (`torch 2.15.0+cpu`, `torch.cuda.is_available() = False`), so combined GPU latency was not measured and no GPU runtime claims are made. We also discuss the engineering implication candidly: while segmentation (~0.13 s) and rule evaluation (<0.04 ms) are interactive on CPU, multi-stage OCR with table-structure recognition (~20.6 s median on CPU) is the primary latency and memory bottleneck, motivating GPU/NPU acceleration and quantisation in future work.

**Location in Revised Manuscript:** *Data Analysis Technique*, *Edge Deployment Benchmark: Latency, Memory, and Artifact Size* (**Table 15**), *Discussion*, *Conclusion*, and *Suggestion*.

---

### Item 6 — Annotation/Class-Remapping Audit and Class Imbalance Analysis (Reviewer A3 / Reviewer B1)

**Reviewer / Editor Comment:**  
*The class remapping/annotation audit has not been reported, and class imbalance handling has not been analysed (A3/B1).*

**Authors' Response:**  
We performed a complete audit of the source-to-target class mapping and per-split class imbalance (`experiments/scripts/audit_dataset_classes.py` $\rightarrow$ `experiments/results/segmentation/dataset_audit.md`) and integrated the findings into *Dataset Construction, Annotation, and Splits* and *Discussion*:
1. **Class Remapping Audit & Catch-All Risk in `Others`:**  
   A frozen lookup table maps **104 source classes in FoodSeg103** (12 Staple Food, 12 Side Dishes, 34 Vegetables, 21 Fruits, and 25 Others) and **102 category IDs in UECFoodPIXCOMPLETE** (38 Staple Food, 42 Side Dishes, 11 Vegetables, 0 Fruits, and 11 Others) into the five *Isi Piringku* groups. Notably, **24.0% of FoodSeg103 classes (25 of 104)** — spanning visually diverse items such as candy, chocolate, cream, biscuit, jam, honey, and condiments — are assigned to `Others`. This explains why `Others` is a heterogeneous catch-all category and accounts for its lower external Mask AP50 (**51.24%** on FoodSeg103, **65.17%** on UEC). Because the mapping table was constructed deterministically by a single author without a second independent annotator, inter-annotator agreement (Cohen's $\kappa$) was not computed and is reported as an explicit limitation.
2. **Class Imbalance Analysis Across Splits:**  
   - In the **training split** (11,684 images), the max/min instance imbalance ratio is **3.04×** (`Vegetables` 10,689 instances [28.0%] vs. `Fruits` 3,513 instances [9.2%]) and the image-presence imbalance ratio is **4.12×** (`Staple Food` 5,708 images [26.6%] vs. `Fruits` 1,384 images [6.4%]).
   - In the **internal validation split** (2,017 images), the ratios are **3.04×** (instance) and **4.30×** (image).
   - Across **external test splits**, UEC test has an instance ratio of **3.37×** among present classes and **0 instances of `Fruits`**, FoodSeg103 validation has an instance ratio of **2.90×**, and FruitSeg30 consists of **100% `Fruits`** (1,993 instances). Consequently, unweighted macro-averages across external sources are not directly comparable without the per-class breakdown provided in **Table 8** and **Table 9**.
   - As noted in *Methods*, plain Ultralytics training was used without oversampling or class-weighted loss so that external evaluation reflects natural source priors rather than resampling artefacts.

**Location in Revised Manuscript:** *Dataset Construction, Annotation, and Splits*, **Table 1**, **Table 2**, **Table 8**, **Table 9**, *Discussion*, and *Conclusion*.

---

### Item 7 — Supporting Clinical and Model References (PERKENI 2024; James et al. 2014; Khanam & Hussain 2024) and Clinical Rules Reference Implementation (Reviewer B4 / Reviewer B7)

**Reviewer / Editor Comment:**  
*Supporting references for clinical thresholds (PERKENI 2024; James et al.) and the YOLO11 citation (Khanam & Hussain) were missing from REFERENCES; include a reference implementation of the clinical rules if available (B4/B7).*

**Authors' Response:**  
1. **References Verified in Both Markdown and Rebuilt `.docx`:**  
   All three references are cited in *Research Methods* and included in full APA format in the `REFERENCES` section of both `manuscript_jri.md` and the regenerated `manuscript_jri.docx`:
   - **PERKENI (2024):** *Perkumpulan Endokrinologi Indonesia (PERKENI). (2024). Pedoman pengelolaan dan pencegahan diabetes melitus tipe 2 di Indonesia 2024. PB PERKENI.*
   - **James et al. (2014):** *James, P. A., Oparil, S., Carter, B. L., Cushman, W. C., Dennison-Himmelfarb, C., Handler, J., et al. (2014). 2014 evidence-based guideline for the management of high blood pressure in adults: Report from the panel members appointed to the Eighth Joint National Committee (JNC 8). JAMA, 311(5), 507–520. https://doi.org/10.1001/jama.2013.284427*
   - **Khanam & Hussain (2024):** *Khanam, R., & Hussain, M. (2024). YOLOv11: An overview of the key architectural enhancements. arXiv:2410.17725.*
2. **Reference Implementation of the Clinical Rules Engine:**  
   In *Clinical Rules Engine* (**Tables 4 and 5**) and *Clinical Rules Engine and Human-in-the-Loop Integration* (**Table 12**), we explicitly cite the browser reference implementation (`frontend/js/clinical-rules.js`) and the independent specification-verification script (`experiments/scripts/verify_clinical_rules.js`), which confirmed 36/36 daily-limit matches and 30/30 status-decision matches across six clinical profiles and five food scenarios.

**Location in Revised Manuscript:** *Segmentation Model and Training*, *Clinical Rules Engine* (**Table 4**, **Table 5**), *Clinical Rules Engine and Human-in-the-Loop Integration* (**Table 12**), and *REFERENCES*.

---

### Item 8 — Minor Inconsistencies: Evaluator Count (4 vs. 2), Table/Figure Numbering, PP-OCRv5 $\rightarrow$ PP-OCRv4 Explanation, and Point-by-Point Response Letter

**Reviewer / Editor Comment:**  
*Minor inconsistencies remain: number of usability evaluators (4 vs. 2), table/figure numbering, unexplained change from PP-OCRv5 to PP-OCRv4; attach a point-by-point response letter with the next revision.*

**Authors' Response:**  
All minor inconsistencies identified in the 5 November `.docx` have been resolved and synchronised across `manuscript_jri.md` and `manuscript_jri.docx`:
1. **Usability Evaluator Count (4 Evaluators Total; 2 Independent Severity Raters):**  
   Both *Data Analysis Technique* and *Usability Assessment* (**Table 14**) now state consistently that **four evaluators (E1–E4)** participated in defined roles: **E1 and E2** (independent ML engineers/data scientists) conducted the severity-weighted heuristic evaluation, **E3** (lead data scientist) performed an asynchronous architecture review, and **E4** (author/self-evaluator) checked documentation–implementation consistency and was excluded from severity weighting. The typo in the old table caption (*"Usubality evaluator Assesment"*) has been corrected to *Table 14. Nielsen heuristic evaluation evaluator profile.*
2. **Sequential Table and Figure Numbering:**  
   All tables (**Table 1** to **Table 15**) and figures (**Figure 1** to **Figure 10**) are numbered sequentially without gaps, duplicates, or mismatched in-text cross-references.
3. **Explicit Explanation of PP-OCRv5 $\rightarrow$ PP-OCRv4:**  
   In *Nutrition-Label Extraction Pipeline*, we added an explicit clarification: *"An earlier draft of this manuscript referred to PP-OCRv5; that was a documentation error. The implementation pins the model version explicitly (`NUTRILABEL_OCR_VERSION`, default `PP-OCRv4`), and every result reported here was produced with PP-OCRv4; the table-structure stage uses the SLANet_plus table-recognition pipeline."*
4. **Point-by-Point Response Letter:**  
   This document provides the complete point-by-point response mapped to every section, table, and figure of the revised manuscript.
