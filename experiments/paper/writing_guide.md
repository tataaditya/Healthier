# Panduan menulis manuscript — Healthier / NutriLabel

Dokumen ini memetakan **setiap bagian template** (`Publishing (1).docx`) ke
angka dan bukti yang benar-benar ada di repo. Tujuannya agar manuscript tidak
mengulang klaim lama yang sudah terbukti tidak konsisten.

> **Aturan utama:** satu angka, satu sumber. Semua angka di manuscript harus
> berasal dari [`final_numbers.md`](final_numbers.md) atau folder
> `experiments/results/`. Jangan menyalin angka dari draft lama tanpa cek ulang.

---

## 0. Status kesiapan (baca ini dulu)

| Bagian manuscript | Status | Keterangan |
| :--- | :--- | :--- |
| Abstrak | ⚠️ Bisa ditulis, tetapi angka OCR belum ada | Ganti angka segmentasi; jangan klaim OCR 83,8% sebelum rerun |
| Introduction | ✅ Siap | Tambahkan 3 kontribusi eksplisit + sitasi dataset |
| Methods — segmentasi | ✅ Siap | Dataset, split, hyperparameter, hardware sudah lengkap |
| Methods — OCR | ⚠️ Sebagian | Pipeline ada; versi OCR dan dataset evaluasi harus dipastikan |
| Methods — clinical rules | ⚠️ Perlu disamakan dengan kode | Sodium 2.000 vs 1.500/2.300; TDEE 2.069 vs 2.008,5 |
| Results — segmentasi | ✅ Siap (agregat) | Metrik internal + 3 external test sudah ada |
| Results — per-class | ⚠️ Sebagian | Per-class external sudah ada; per-class internal belum |
| Results — OCR | ⚠️ Pilot n=6 | Sudah ada `experiments/results/ocr_full`; belum layak untuk klaim generalisasi |
| Results — latency | ❌ Belum | 168 ms belum diukur untuk model baru |
| Results — usability | ⚠️ Perlu bukti | Klaim “no design flaws” harus diganti severity rating |
| Discussion | ✅ Siap | Domain shift FruitSeg adalah temuan utama, bukan kegagalan tersembunyi |
| Conclusion | ⚠️ Ikut angka baru | Jangan pertahankan angka lama |
| References | ⚠️ Perlu dirapikan | Ada sitasi numerik campur author-date; dataset belum dikutip |

**Kesimpulan:** manuscript **belum siap submit**, tetapi **sudah siap mulai
menulis Methods dan Results segmentasi**. Bagian OCR adalah blocker utama.

---

## 1. Ganti angka lama (find & replace)

| Di draft lama | Ganti menjadi | Alasan |
| :--- | :--- | :--- |
| “trained on 19,087 images” | “trained on **11,684** images; validated on **2,017**; tested on **5,013** external images” | 19.087 adalah kandidat corpus; 373 duplikat (dHash) dibuang → 18.714 unik *(angka 373 perlu diregenerasi dari `dataset_summary.json` yang gitignored)* |
| “3,529 validation images” | **2,017** validation images | Split baru 85:15, bukan 80:20 |
| “80:20 split” | **85:15 split** | `training_config.json`: `val_frac = 0.15` |
| "50 epochs" | **max 100 epochs, early stop at 67, best epoch 48** | `experiments/runs/seg_main/results.csv` |
| “RTX 3050 6 GB / 8 GB RAM” (training) | **Kaggle Tesla T4 15.6 GB** untuk training; RTX 3050 6 GB untuk deployment | `args.yaml` + `training_config.json` |
| Mask mAP50 **67.06** | **71.66** | checkpoint baru, epoch 48 |
| Mask mAP50-95 **52.84** | **56.91** | checkpoint baru, epoch 48 |
| Mask P/R **72.07 / 60.46** | **72.95 / 65.50** | checkpoint baru |
| Box mAP50 / mAP50-95 (jika ada) | **72.37 / 60.11** | checkpoint baru |
| “168 ms per image” | **ukur ulang** (jangan pakai angka lama) | belum ada benchmark model baru |
| OCR 83.8% [78.6; 87.9] | **pilot n=6: baseline 21.74% → grid parser 47.83% [29.17; 66.67] (GT terkoreksi)** | `experiments/results/ocr_full{,_grid}`; klaim lama tidak tereproduksi |
| “PP-OCRv5” | **PP-OCRv4** | `backend/app.py:75` default `PP-OCRv4`; `paddleocr==3.7.0` adalah versi **library**, dan v4 termasuk `_SUPPORTED_OCR_VERSIONS` di `paddleocr/_pipelines/ocr.py:58` |
| Sodium **2,000 mg** | **1,500 mg/hari (hipertensi); 2,300 mg/hari (umum)** | `frontend/js/clinical-rules.js` |
| TDEE contoh **2,069 kcal** | **2,008.5 kcal** | BMR 1,673.75 × activity factor 1.2 |
| “best_float16.tflite” | **best_float32.tflite** (atau ekspor ulang FP16) | artefak yang ada FP32 |
| “no design flaws” | **“no severity-4 issues; remaining issues severity 2–3”** | klaim lama overclaim |
| “Fruits best 81.44%, Others worst 53.37%” | **jelaskan ini metrik internal**; untuk external, Fruits jatuh ke **17.06%** | jangan campur internal vs external |

---

## 2. Abstrak / Abstract

Pertahankan alur draft, tetapi ganti angka dan tambahkan kejujuran domain shift.
Template yang aman:

> Healthier integrates plate-image segmentation with nutrition-label extraction.
> The segmentation model was trained on **11,684** images and validated on
> **2,017** images drawn from a **18,714-image** unique corpus (19,087 candidate
> images after removing 373 duplicates). On internal validation, YOLO11s-seg
> reached **mask mAP50 71.66%** and **mAP50-95 56.91%** (precision 72.95%,
> recall 65.50%). On external tests, mask mAP50 was **69.15%** on UECFood test,
> **63.66%** on FoodSeg103 validation, and **17.06%** on FruitSeg30, revealing a
> clear cross-dataset domain shift for fruit-only imagery. The OCR module
> achieved **field-level accuracy [ISI SETELAH RERUN]** on **N** packaging
> photographs with a cluster-bootstrap 95% CI of **[ ... ]**. Clinical rules
> based on PERKENI 2024 and JNC 8/DASH translate these outputs into deterministic
> SAFE/WARNING/DANGER recommendations. Human-in-the-loop correction remains the
> main mitigation for segmentation and OCR errors.

**Jangan** menulis “trained on 19,087” dan **jangan** mempertahankan 83,8% tanpa
rerun. Jika OCR belum selesai, tulis abstrak versi segmentasi saja dan jelaskan
bahwa evaluasi OCR dilaporkan di bagian hasil; tetapi untuk submit, angka OCR
harus ada.

---

## 3. Introduction

Struktur yang disarankan (5–6 paragraf):

1. **Masalah:** NCD/diabetes/hipertensi + rendahnya literasi gizi. Pakai sitasi
   yang sudah ada (Wahidin 2024; Aprianti 2023; Bonaccio 2023).
2. **Mengapa sekarang:** FoPNL/Nutri-Score membantu, tetapi label kecil,
   melengkung, dan reflektif; makanan di piring tidak punya label sama sekali.
3. **Dua masalah pengenalan:** segmentasi porsi piring (Isi Piringku) + OCR tabel
   gizi kemasan, di bawah satu lapisan keputusan klinis.
4. **Gap:** belum ada pipeline yang menggabungkan segmentasi berbasis guideline
   + OCR edge-based tanpa LLM cloud + aturan klinis deterministik. Sitasi
   Utami 2023, Permadani & Gunawan, Saputra 2024, Rosyadi 2024, Freihat 2025.
5. **Kontribusi eksplisit (wajib, reviewer meminta):**
   - **C1:** mekanisme integrasi guideline-grounded antara segmentasi piring dan
     OCR label → keputusan klinis deterministik.
   - **C2:** pipeline OCR edge-based + HITL yang reproducible, dengan ablation
     per tahap dan evaluasi field-level.
   - **C3:** analisis generalisasi dan kegagalan pada sumber data berbeda
     (UEC/FoodSeg/FruitSeg), termasuk domain shift.
6. **Scope + pertanyaan riset:** batasi ke masakan Indonesia dan tabel gizi
   kemasan Indonesia; sebutkan batasan data publik.

Tambahkan sitasi dataset: **UECFoodPIXCOMPLETE**, **FoodSeg103**, **FruitSeg30**.
Tanpa sitasi dataset, reviewer akan mempertanyakan asal data.

---

## 4. Research Methods

### 4.1 Hardware dan environment

Ganti kalimat hardware:

> All training and evaluation were performed on a Kaggle Tesla T4 GPU with
> 15.6 GB VRAM (Python 3.12.13, PyTorch 2.10.0+cu128, Ultralytics 8.4.199).
> Deployment latency was measured on a laptop with an NVIDIA RTX 3050 6 GB GPU.

Jangan klaim “8 GB RAM” untuk training. Jika ingin menyebut RAM laptop, pisahkan
antara training dan deployment.

### 4.2 Segmentation Model Development

Tulis persis:

- Arsitektur: **YOLO11s-seg**.
- Epoch: **max 100**, patience **20**, berhenti di **67**, checkpoint terbaik
  **epoch 48**.
- Optimizer: **AdamW**, `lr0=0.001`, `lrf=0.01`, weight decay `0.0005`,
  momentum `0.9`.
- Batch **16**, `imgsz=640`, seed **42**, deterministic.
- Augmentasi: `hsv_h=0.01`, `hsv_s=0.3`, `hsv_v=0.2`, `degrees=10`,
  `translate=0.1`, `scale=0.5`, `fliplr=0.5`, `mosaic=0.5`, `erasing=0.4`,
  `close_mosaic=10`.
- Split: **85:15** stratified, seed 42, tidak ada overlap.
- Dedup: **dHash perceptual 8×8** (grayscale → resize 9×8 → bit
  `img[:,1:] > img[:,:-1]`) di `prepare_yolo_seg.py:234-243`, diterapkan di
  `:458-466`. Jadi duplikat yang di-resize/dikompres pun tertangkap. **Bukan**
  exact byte-hash.
- Training baru (`train_seg.py`) memakai `model.train(...)` Ultralytics polos:
  **TIDAK ada oversampling dan TIDAK ada weighted loss**. Jangan tulis klaim
  "weighted loss" atau "oversampling" sama sekali. (Oversampling 4.840 → 12.082
  adalah run **lama** di `notebook-audit.md`, bukan run ini.)
- ⚠️ Angka "373 duplikat dibuang" berasal dari
  `experiments/dataset/dataset_summary.json` yang **tidak ada** sekarang
  (gitignored). Jalankan ulang `prepare_yolo_seg.py` untuk meregenerasi summary
  sebelum memakai angka 373 di naskah.

### 4.3 Dataset (Table 1)

Gunakan [`tables/table_dataset.md`](tables/table_dataset.md) apa adanya. Poin
yang harus muncul di teks:

- Kandidat 19.087; 373 duplikat dHash dibuang; **18.714 unik** *(verifikasi ulang
  angka 373 via `prepare_yolo_seg.py`; summary gitignored)*.
- Train **11.684** (UEC 7.478 + FoodSeg 4.206).
- Val **2.017** (UEC 1.285 + FoodSeg 732).
- External test **5.013** (UEC 948; FoodSeg 2.096; FruitSeg 1.969).
- **FruitSeg30 seluruhnya external**; kelas `Fruits` di train hanya dari
  FoodSeg (3.513 instance; 1.384 gambar).
- Pindahkan baris “Supermarket Nutrition Table 446” ke tabel OCR terpisah.
  Tabel segmentasi tidak boleh mencampur OCR dan segmentasi.
- Jangan klaim “3 folder dataset besar” tanpa angka per sumber.

### 4.4 OCR Extraction Pipeline

- Pastikan versi: `backend/app.py:75` memakai **PP-OCRv4** (`NUTRILABEL_OCR_VERSION`,
  default `PP-OCRv4`), diteruskan ke `ocr_version` di `:225-226`. `paddleocr==3.7.0`
  adalah versi **library**, bukan model; `PP-OCRv4` termasuk
  `_SUPPORTED_OCR_VERSIONS` di `paddleocr/_pipelines/ocr.py:58`. Jadi naskah,
  README, dan kode aman menulis **PP-OCRv4** — bukan kontradiksi.
  Catatan: label log `[Config] OCR version=...` (`:213`) adalah versi yang
  **diminta**, bukan bukti versi model yang benar-benar dimuat. Untuk bukti akhir,
  cek field `status.model` (`:2458`,`:2745`) setelah satu run nyata. Jangan klaim
  v5 (atau v4) sebelum itu diverifikasi dari log init PaddleOCR.
- Tulis 4 tahap preprocessing: deskew (Hough), deglare (Telea inpainting),
  CLAHE, ROI crop. Tambahkan filter semantik + fuzzy matching.
- Persamaan Levenshtein dan similarity sudah ada; pertahankan.
- Dataset OCR: minimal **100 citra** dari brand/toko berbeda, split **by
  product/brand** (bukan random per gambar), 70/15/15. Jika tetap 30 citra,
  tulis sebagai pilot dan jangan klaim generalisasi.
- Metrik: field-level accuracy = (exact + within 5%) / total GT field; laporkan
  exact/close/wrong/missing; CI **cluster bootstrap per gambar**, bukan Wilson
  n=30.
- Ablation: raw → +deskew → +deglare → +CLAHE → +ROI → +semantic → +fuzzy →
  full. Ini yang diminta reviewer; jangan dipotong.

### 4.5 Clinical Rules Engine

- Ganti sodium: **1.500 mg/hari untuk hipertensi**, **2.300 mg/hari untuk
  umum** (`clinical-rules.js:39,74`); sebut DASH/JNC 8 dengan benar. Jangan tulis
  2.000 mg sebagai angka tunggal.
  ⚠️ **Verifikasi halaman guideline sumber** (JNC 8/DASH) langsung dari dokumen
  asli sebelum mengutip nomor halaman — dokumen tidak ada di repo ini.
- Ganti contoh TDEE: BMR laki-laki 70 kg/175 cm/25 tahun =
  `10(70)+6.25(175)-5(25)+5 = 1.673,75`; TDEE = `1.673,75 × 1.2 = 2.008,5 kcal`.
  Sebut activity factor 1.2 (sedentary).
- Gula: **diabetes 5% TDEE** (`clinical-rules.js:52` → `0.05 × TDEE / 4` gram);
  **non-diabetes 10% TDEE** (`:38` → `0.10 × TDEE / 4`). Jadi "5%" itu **khusus
  diabetes** — jangan ditulis sebagai batas universal. Sumber PERKENI 2024 tidak
  ada di repo; **verifikasi 5% dari dokumen asli** sebelum mengutip.
- **Ambang risiko harus sama persis dengan kode** (`clinical-rules.js`), bukan 75%:

  | Pemicu WARNING | Ambang di kode |
  | :--- | :--- |
  | Gula (diabetes & umum) | > **50%** batas harian (`:129`,`:168`) |
  | Lemak jenuh (dislipidemia/diabetes) | > **50%** (`:157`) |
  | Natrium aktual (hipertensi) | > **40%** batas harian (`:149`) |
  | Natrium per 100 g (hipertensi) | > **600 mg** (`:75`); umum > **800 mg** (`:46`) |
  | Protein (nefropati) | > **30%** (`:137`) |
  | **DANGER** | **hanya** saat gula **> 100%** batas harian (`:126`,`:165`) |

  Sodium dan lemak jenuh mentok di WARNING; hanya gula yang bisa DANGER. Jangan
  tulis "WARNING ≥75%".
- Sebut ini **rule verification**, bukan “clinical validation”, kecuali ada
  reviewer ahli gizi. Buat tabel 10–20 skenario profil × makanan dengan expected
  status dan bandingkan manual.

### 4.6 Data Engineering

- Ganti `isipiringku_v3` dengan **revision dataset** yang benar.
- FP16: artefak yang ada **best_float32.tflite**. Ganti klaim menjadi FP32, atau
  ekspor ulang FP16 dan ukur ukuran file + akurasi. Jangan klaim “50% compression”
  tanpa bukti.
- Sebutkan konversi ONNX/TFLite dan fallback browser apa adanya.

### 4.7 HITL dan Usability

- HITL: deskripsikan Segmentation Editor + form koreksi. Jangan klaim “validated
  by Nielsen” kecuali usability benar-benar dilakukan dan ada severity rating.
- Nielsen: **tidak ada data temuan (jumlah per severity, jumlah evaluator, skor)
  di repo ini** — semua hanya referensi. Semua angka Nielsen **harus dari catatan
  usability milikmu sendiri**. Jika data evaluator nyata ada, laporkan **severity
  0–4 per temuan** dan setujui/disagree; ganti “no design flaws” dengan
  “no severity-4 issues; the remaining issues were severity 2–3”. Jika **tidak
  ada**, hapus klaim dan tulis sebagai future work. Jangan mengarang severity.
- Sebutkan jumlah evaluator dan bahwa E4 dikecualikan dari severity (jika itu
  yang dilakukan). Sample 2 evaluator untuk severity kecil; sebut sebagai
  limitation.

---

## 5. Results and Discussion

### 5.1 Instance Segmentation Evaluation

Gunakan:

- Internal val: mask mAP50 **71.66%**, mAP50-95 **56.91%**, P **72.95%**,
  R **65.50%**; box mAP50 **72.37%**, mAP50-95 **60.11%**.
- External: table [`tables/table_segmentation_external.md`](tables/table_segmentation_external.md).
- Pixel: table [`tables/table_segmentation_pixel.md`](tables/table_segmentation_pixel.md).

Narasi yang benar:

> The model exceeds the 60% mAP50 target on internal validation and on two
> external sources (UEC 69.15%, FoodSeg 63.66%), but drops sharply on FruitSeg30
> (17.06%). The drop is a domain-shift effect: FruitSeg30 contains fruit-only
> imagery, while training fruit instances come from FoodSeg103 plate scenes.
> Pixel-level results confirm the failure mode: on FruitSeg30 the model predicts
> many non-fruit pixels as Staple Food and recovers only 4.19% F1 for Fruits.

Jangan menulis “Fruits is the best class” tanpa kualifikasi “on internal
validation”. Untuk external, Fruits adalah kelas terburuk. Ini justru memperkuat
kontribusi C3.

**Per-class external** sudah tersedia di
[`tables/table_segmentation_per_class.md`](tables/table_segmentation_per_class.md)
(dihitung ulang dengan `per_class_metrics.py` dari `predictions.json` + label).
Untuk internal, angka 81.44/53.37 dari draft lama **jangan dipakai**; ekspor
`per_class.csv` dari checkpoint baru saat citra internal masih ter-mount.

### 5.2 Inference time

- Jangan pakai 168 ms. Ukur dengan `scripts/benchmark.py latency` atau
  `benchmark.py api-latency` pada model baru, di GPU deployment (RTX 3050) dan
  CPU. Laporkan mean/median/P95 dan batch size. Jika tidak sempat, hapus klaim
  latency dan sebut sebagai future work.
- Training time **4,01 jam** di T4 boleh dilaporkan sebagai biaya training.

### 5.3 OCR Evaluation

Blocker. Sudah dijalankan pada pilot n=6 (`experiments/results/ocr_full` =
parser lama, `ocr_full_grid` = parser revisi):

- Field-level accuracy: baseline **21,74%**, parser grid **47,83%**
  (95% CI [29,17; 66,67]). GT pilot sudah dikoreksi lewat audit visual
  (`gt_audit.md`) — dua dari enam GT salah transkripsi.
- Laporkan CI cluster bootstrap per gambar dan exact/close/wrong/missing
  (`table_ocr.md`).
- Perbaikan parser: grid baris-label x kolom-nilai + pemecahan bbox angka yang
  tergabung vertikal (mis. `1725` = 1,7,2,5) sehingga kolom Per Sajian dan %AKG
  terpisah; token satuan `mg`/`g` dipertahankan.
- Buat tabel error taxonomy: glare, curved, small font, multi-column,
  serving-size format, per-100g vs per-serving.
- Jangan pakai 83,8% atau CI [78,6; 87,9] dari draft.
- Karena n=6, tulis sebagai **pilot/eksploratif domain-shift**, bukan klaim
  generalisasi.

Jika waktu habis: jangan submit dengan angka OCR lama. Turunkan scope (mis.
laporkan OCR sebagai pilot 30 citra) dan minta extension.

### 5.4 Clinical Rules + HITL

- Ganti contoh TDEE dan sodium.
- Tambahkan tabel verifikasi aturan 10–20 skenario.
- Jangan klaim “clinical validation”.
- HITL: jika ada simulasi koreksi, laporkan gain (mis. selisih akurasi sebelum/
  sesudah koreksi). Jika tidak ada, tulis sebagai desain mitigasi, bukan hasil
  kuantitatif.

### 5.5 Usability

- Laporkan severity rating dan jumlah temuan.
- Ganti “no design flaws” dengan kalimat yang sesuai data.
- Jika evaluator hanya 2 untuk severity, sebut limitation.

### 5.6 Table 3 (summary of key findings)

Ganti isi dengan angka baru:

> Segmentation reached mask mAP50 71.66% (mAP50-95 56.91%) on internal validation,
> with external mAP50 of 69.15% (UEC), 63.66% (FoodSeg103), and 17.06%
> (FruitSeg30). The FruitSeg30 drop is a cross-dataset domain-shift finding.
> OCR field-level accuracy is [N]% (95% CI [..]) on [N] images. Clinical rules
> correctly reproduced the expected SAFE/WARNING/DANGER status in [N]/[N]
> verification scenarios.

---

## 6. Conclusions and Suggestions

- Ganti angka lama dengan angka baru.
- Jangan klaim “no design flaws”.
- Sebutkan kontribusi C1–C3 dan batasan:
  - dataset publik didominasi makanan non-Indonesia;
  - FruitSeg30 menunjukkan domain shift;
  - estimasi porsi masih 2D;
  - OCR masih rentan pada kemasan melengkung/glare;
  - usability sample kecil.
- Future work: enrich dataset lokal, INT8 quantization, SLM offline, estimasi
  volume multi-angle, integrasi Google Fit/Apple Health.
  - **VLM**: analisis pilot n=6 menunjukkan VLM cloud (gemma4) menaikkan field
    accuracy 44,7% → **68,1%**, terutama menghapus mode gagal "missing"
    (36% → 6%), tetapi menggeser error ke "salah tapi yakin" (risiko klinis).
    Arah yang disarankan: **hybrid** (VLM untuk struktur/kolom, OCR
    deterministik untuk digit) + lapisan verifikasi. Catat sebagai future work,
    JANGAN ubah klaim C1/C2 menjadi VLM. Detail:
    `experiments/paper/vlm_analysis.md`.

---

## 7. References

Perbaiki:

- Ganti semua sitasi numerik (`[4]`, `[6]`, ...) menjadi author-date **atau**
  sebaliknya; jangan campur. Template memakai author-date di body, tetapi
  conclusion memakai numerik. Pilih satu.
- Hapus duplikasi **Wahidin 2024a/2024b** (DOI sama).
- Perbaiki “Aprianti Aprianti” → “Aprianti et al.”.
- Tambahkan sitasi dataset: UECFoodPIXCOMPLETE, FoodSeg103, FruitSeg30.
- Tambahkan sitasi YOLO11 (Ultralytics) dan PaddleOCR PP-OCR.
- Cek DOI dan tahun.
- Tambahkan **Data Availability Statement**, **Code Availability**, dan
  **Ethics Statement** (jika usability melibatkan manusia).

---

## 8. Peta tabel dan gambar

| Item | Sumber | Letak |
| :--- | :--- | :--- |
| Table 1 dataset | `tables/table_dataset.md` | Methods 4.3 |
| Table 2 hyperparameter | `training_config.json` / `args.yaml` | Methods 4.2 |
| Table 3 evaluator | draft Table 2 (verifikasi bukti) | Methods 4.7 |
| Table 4 internal metrics | `tables/table_segmentation_internal.md` | Results 5.1 |
| Table 5 external metrics | `tables/table_segmentation_external.md` | Results 5.1 |
| Table 6 pixel metrics | `tables/table_segmentation_pixel.md` | Results 5.1 |
| Table 7 OCR per-field | `eval_ocr.py` output | Results 5.3 |
| Table 8 OCR ablation | `ablate_ocr.py` output | Results 5.3 |
| Figure 1 pipeline | buat dari arsitektur | Methods |
| Figure 2 confusion matrix | `results/segmentation/*_instance/confusion_matrix*.png` | Results 5.1 |
| Figure 3 PR curves | `results/segmentation/*_instance/MaskPR_curve.png` | Results 5.1 |
| Figure 4 error examples | pilih dari `val_batch*_pred.jpg` + foto Indonesia | Results 5.1/5.3 |

---

## 9. Checklist sebelum submit

- [ ] Semua angka di abstract = results = tables = conclusion.
- [ ] Tidak ada “19,087 trained” atau “3,529 validation”.
- [ ] Tidak ada “PP-OCRv5” jika kode memakai v4.
- [ ] Tidak ada “FP16” jika artefak FP32.
- [ ] OCR sudah dijalankan ulang; CI cluster bootstrap; n ≥ 100 (atau scope
      diturunkan secara jujur).
- [ ] Per-class internal sudah diekspor.
- [ ] Latency diukur atau klaim dihapus.
- [ ] Baseline full-budget ada, atau screening dilabeli “pilot” di appendix.
- [ ] Sodium/TDEE paper = kode.
- [ ] Klaim usability sesuai severity rating.
- [ ] Sitasi dataset + YOLO11 + PaddleOCR ada.
- [ ] Data/code availability statement ada.
- [ ] Tidak ada klaim “state-of-the-art” tanpa perbandingan setara.
