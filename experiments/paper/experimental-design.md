# Experimental design for the revision

Dokumen ini menjawab empat komentar reviewer dengan desain yang **jujur dan bisa
dieksekusi di Kaggle T4 dalam 4 hari**. Tidak ada klaim Indonesian segmentation
benchmark yang tidak ada.

---

## 1. Tiga kontribusi eksplisit

**C1 — Guideline-grounded integration.**
Mekanisme yang menerjemahkan proporsi segmentasi piring (Isi Piringku) dan nilai
gizi dari label kemasan menjadi keputusan klinis deterministik (PERKENI 2024 /
JNC 8), diuji end-to-end terhadap baseline unimodal dan “tanpa clinical rules”.

**C2 — Reproducible edge OCR + HITL.**
Pipeline OCR dengan ablation per tahap preprocessing, evaluasi field-level
dengan cluster bootstrap, dan kuantifikasi gain Human-in-the-Loop.

**C3 — Generalization & failure analysis.**
Evaluasi pada held-out source yang berbeda + primary Indonesian test set,
dengan confusion matrix, per-class error, dan taksonomi kegagalan.

---

## 2. Dataset dan split

### 2.1 Desain yang disarankan (memakai seluruh 19.087 citra)

| Bagian | Source | Jumlah | Peran |
| :--- | :--- | ---: | :--- |
| Train pool | UECFood train | 9.000 | train/val |
| Train pool | FoodSeg103 train | 4.983 | train/val |
| **Internal val** | 15% stratified dari train pool | ±2.097 | early stopping / pemilihan model |
| **Train final** | 85% dari train pool | ±11.886 | training |
| **Test 1** | UECFood test | 1.000 | held-out in-domain |
| **Test 2** | FoodSeg103 validation | 2.135 | held-out, 5 kelas |
| **Test 3** | FruitSeg30 | 1.969 | cross-dataset, fruit-only |
| **Total** | | **19.087** | |

Kenapa desain ini:
- Memakai **seluruh** corpus, jadi klaim 19.087 bisa dipertanggungjawabkan.
- UEC test, FoodSeg val, dan FruitSeg30 **tidak pernah** dipakai training/early
  stopping.
- FruitSeg30 adalah **sumber berbeda** (Malaysia/Bangladesh/Australia) untuk
  generalisasi; FoodSeg val menguji semua 5 kelas.
- Tidak perlu Indonesian segmentation dataset publik.

**Peringatan:** FruitSeg30 seluruhnya menjadi test, sehingga kelas `Fruits` di
train harus berasal dari FoodSeg103 (21 kelas buah). Cek `dataset_summary.json`;
jika `buah` di train = 0, pindahkan sebagian FruitSeg30 ke train.

### 2.2 Jika ingin mempertahankan split official (Run A)

Gunakan 15.558 train / 3.529 val seperti rekonstruksi di
[`notebook-audit.md`](notebook-audit.md), lalu **pecah val menjadi val/test**
(50:50) sebelum training. Test tidak boleh dipakai untuk early stopping.
Ini lebih cepat, tetapi external test tetap perlu (FruitSeg30 atau primary
Indonesian set).

### 2.3 OCR

- Kumpulkan **100–150 citra** dari primary data.
- Split **by product/brand**, bukan random per image, untuk menghindari leakage.
- 70/15/15 train/val/test.
- External test: 30–50 citra dari toko/waktu berbeda.
- Field: kalori, lemak_total, lemak_jenuh, lemak_trans, kolesterol, natrium,
  karbohidrat, serat, gula, protein, takaran_saji_g.
- Aturan: nilai **per sajian**; normalisasi g; field tidak ada = kosong, bukan 0.

---

## 3. Screening vs final comparison

Niat “bandingkan model dengan epoch kecil lalu ambil yang terbaik” **boleh**,
tetapi harus diformulasikan sebagai **architecture screening**, bukan final
baseline comparison.

### Tahap 1 — Screening (5–10 epoch)
- Model: YOLO11s, YOLOv8s, YOLO11n, YOLO26n/s (pilih 3–4).
- Dataset: subset tetap (mis. 5.000 train / 1.000 val) atau full train, **sama
  untuk semua model**.
- Budget: epoch sama, seed sama, augmentasi sama.
- Metrik: **mask mAP50-95 pada validation** (bukan test).
- Jika memungkinkan: 3 seed, laporkan mean ± SD.
- Output: tabel screening di **appendix**, dengan label “pilot, not final”.

### Tahap 2 — Final (100 epoch, patience 20)
- Latih **2 model terbaik** dari screening dengan budget penuh pada train final:
  kandidat utama + satu baseline.
- Evaluasi pada test yang sudah dibekukan.
- Jika waktu hanya cukup 1 baseline, pilih **YOLO11n-seg** (lebih cepat) atau
  **YOLOv8s-seg** (lebih kuat). Jangan memakai 5-epoch sebagai satu-satunya
  baseline di tabel utama.

**Aturan anti-bias:** jangan memilih model berdasarkan test. Semua keputusan
arsitektur/threshold memakai validation. Test dibuka sekali di akhir.

---

## 4. Proses anotasi

### 4.1 Dataset publik (UECFood, FruitSeg30, FoodSeg103)
- Anotasi asli berasal dari dataset; **kutip protokol aslinya**.
- Jelaskan aturan remapping 102/30/103 kelas → 5 kelompok Isi Piringku.
- Lakukan **audit sample** 200 citra stratified (40 per kelas) oleh 2 annotator:
  - Apakah kelompok hasil remapping benar?
  - Hitung Cohen’s κ; disagreement diselesaikan diskusi.
- Tulis: “We did not re-annotate the public masks; we audited the remapping.”

### 4.2 Primary Indonesian test set (disarankan)
- 30–50 foto piring makanan Indonesia.
- Anotasi polygon dengan LabelMe/CVAT; boleh SAM-assisted + koreksi manual
  (dokumentasikan).
- 2 annotator untuk minimal 20%; hitung κ/IoU; adjudikasi.
- Jika tidak sempat membuat mask: lakukan **qualitative error analysis** dan
  sebut sebagai “qualitative domain-shift analysis”, bukan metrik kuantitatif.

### 4.3 OCR primary
- 2 annotator untuk minimal 20%; Cohen’s κ per field.
- Adjudikasi field yang berbeda.
- Simpan foto asli; pipeline yang melakukan crop, bukan annotator.

---

## 5. Metrik dan analisis

### Segmentasi
1. **Instance-level**: box/mask P, R, mAP50, mAP50-95 (`eval_seg.py`).
2. **Pixel-level**: TP/FP/FN/TN, accuracy, P/R/F1 per kelas (`eval_seg_pixel.py`).
3. **Confusion matrix**: instance dan pixel; normalized.
4. **Per-class error**: kelas terburuk + contoh gambar.
5. **Latency**: mean/median/P95, GPU T4 dan CPU/edge.

### OCR
1. Field accuracy = (exact + within 5%) / total GT fields.
2. Exact / close / wrong / missing rates.
3. **Cluster bootstrap CI per gambar** (`eval_ocr.py`), bukan Wilson n=30.
4. Per-field table + error taxonomy (glare, curved, small font, multi-column,
   serving-size format, per-100g vs per-serving).
5. Ablation semua tahap preprocessing (`ablate_ocr.py`).

### Clinical rules
- Tabel 10–20 skenario (profil + makanan) dengan expected SAFE/WARNING/DANGER.
- Bandingkan dengan perhitungan manual; sebut “rule verification”, bukan
  “clinical validation” kecuali ada review ahli gizi.

### Statistik
- Screening: mean ± SD (3 seed jika memungkinkan).
- Final: 1 seed + limitation; atau 3 seed jika waktu.
- OCR: bootstrap 2.000 iterasi, cluster by image.
- Baseline: paired bootstrap atau selisih CI; jangan hanya klaim “lebih baik”.

---

## 6. Baseline dan ablation

| Komponen | Baseline / ablation | Biaya |
| :--- | :--- | :--- |
| Segmentasi | YOLO11n/YOLOv8s full-budget | sedang |
| Segmentasi | tanpa augmentation / tanpa oversampling | ablation (opsional) |
| OCR | raw OCR (tanpa preprocessing) | murah |
| OCR | +deskew, +deglare, +CLAHE, +ROI, +semantic, +fuzzy | murah |
| OCR | Tesseract (jika sempat) | murah |
| Clinical | tanpa rules vs dengan rules | murah |
| HITL | simulasi koreksi top-k error | murah |

Prioritas jika waktu habis: **OCR ablation > external test > baseline full >
segmentation ablation > HITL study**.

---

## 7. Jawaban untuk “tidak ada dataset segmentation Indonesia”

Tulis di manuscript dan response letter:

> “To the best of our knowledge, there is no public Indonesian plate-image
> segmentation benchmark with pixel-level annotations. To avoid overclaiming,
> we evaluate generalization in two ways: (i) held-out official test splits from
> three public food-segmentation datasets, including a cross-dataset source
> (FruitSeg30), and (ii) a primary Indonesian plate test set of N images with
> coarse polygon annotations for domain-shift analysis. We report the
> performance drop and limit our claims accordingly.”

Jika primary Indonesian set hanya kualitatif:

> “Because pixel-level annotation of Indonesian dishes is not available at
> scale, the primary Indonesian set is used for qualitative error analysis, not
> as a quantitative benchmark. We explicitly treat this as a limitation.”

Ini jujur, dan reviewer biasanya menerima selama tidak mengklaim generalisasi
Indonesia tanpa bukti.

---

## 8. Checklist konsistensi dataset untuk manuscript

- [ ] Satu tabel: source, license, total, train, val, test, annotation type.
- [ ] Jumlah unik + jumlah setelah oversampling (jika ada).
- [ ] Distribusi kelas: images containing class dan instance count per split.
- [ ] Split seed, stratifikasi, dedup hash, no-overlap statement.
- [ ] Test set dibekukan sebelum training; tidak dipakai untuk model selection.
- [ ] Anotasi: protokol, annotator, κ, adjudikasi.
- [ ] Hardware: Kaggle T4 15.6 GB (training), RTX 3050 (deployment jika diuji).
- [ ] Versi OCR, epoch, FP16/FP32, dan metrik pixel vs instance dijelaskan.
