# 4-day revision plan (hard deadline)

> **BACA DULU:** [`notebook-audit.md`](notebook-audit.md). Notebook asli memakai
> **4.840 train / 960 val** (oversampling 12.082), bukan 19.087/3.529, dan
> training di **Tesla T4**, bukan RTX 3050. Kalau kamu tetap ingin klaim
> 19.087, kamu **harus retrain di seluruh 19.087 citra unik** dan melaporkan
> hasil baru.

> **Aturan:** kalau checkpoint molor, **potong scope**, jangan mengarang angka.
> Prioritas: (1) konsistensi data, (2) independent test, (3) OCR ablation,
> (4) baseline, (5) retrain penuh.

---

Ikuti split dan protokol di [`experimental-design.md`](experimental-design.md).
Training di **Kaggle T4**; upload dataset hasil `prepare_yolo_seg.py` sebagai
Kaggle Dataset, lalu mount di `/kaggle/input/...`.

## Hari 1 — Data, split, mulai training

**Pagi (0–4 jam)**
- [ ] Buka `experiments/notebooks/revision_kaggle.ipynb`, pilih GPU T4.
- [ ] (Opsional, hanya jika tidak memakai Kaggle)
      `python experiments/scripts/local/check_environment.py`
- [ ] Install/fix CUDA PyTorch + PaddlePaddle GPU (atau siapkan Kaggle).
- [ ] Download UECFoodPIXCOMPLETE, FoodSeg103, FruitSeg30.
- [ ] Siapkan `classes.txt` untuk tiap dataset.
- [ ] Tentukan external test: FruitSeg30 **atau** UEC test.

**Siang (4–8 jam)**
- [ ] Jalankan `prepare_yolo_seg.py` sesuai desain (FruitSeg30 sebagai external
      test, UEC test + FoodSeg val sebagai held-out test).
- [ ] Review `class_map.csv`; perbaiki mapping yang salah.
- [ ] Cek `dataset_summary.json`: `buah` di train > 0 (dari FoodSeg).
- [ ] Screening 5–10 epoch untuk 3–4 model di T4; pilih top-2 berdasarkan
      **validation** mask mAP50-95.
- [ ] Mulai training final top-2 (YOLO11s + baseline) overnight.
      Jika VRAM/epoch tidak muat, batch 8/imgsz 512.

**Malam (8–12 jam)**
- [ ] Siapkan OCR test: 100+ citra, `*_gt.json`.
- [ ] Isi `experiments/paper/claims.csv` untuk audit angka lama.
- [ ] Jalankan `eval_ocr.py` pada 30 citra lama sebagai sanity check.
- [ ] Jalankan `ablate_ocr.py` pada 30 citra lama (preliminary).

**Checkpoint 1:** training berjalan, dataset summary valid, ≥50 OCR GT siap.
Jika gagal: gunakan model existing + external test, dan minta extension untuk retrain.

---

## Hari 2 — Evaluasi & baseline

**Pagi**
- [ ] Cek training selesai; catat epoch terbaik & `results.csv`.
- [ ] `eval_seg.py --split test`.
- [ ] Cek `confusion_matrix_normalized.png`; identifikasi kelas terburuk.
- [ ] Mulai baseline `yolo11n-seg` / `yolov8s-seg` dengan split sama (overnight
      jika belum selesai).

**Siang**
- [ ] `eval_ocr.py` pada 100+ citra.
- [ ] `ablate_ocr.py` semua variant.
- [ ] Bandingkan PP-OCRv4 vs v5 jika tersedia.
- [ ] Kumpulkan `errors.csv`; pilih 5–8 contoh error untuk figure.

**Malam**
- [ ] `make_tables.py`.
- [ ] Hitung latency (GPU/CPU) untuk paper.
- [ ] Mulai menulis Methods: dataset, annotation, split, training, metrics.

**Checkpoint 2:** seg test metrics + OCR summary + ablation table selesai.
Jika tidak: drop baseline training, pakai `raw` OCR sebagai baseline dan jelaskan.

---

## Hari 3 — Analisis, figure, writing

**Pagi**
- [ ] Finalkan 3 kontribusi (C1 integrasi klinis, C2 OCR+HITL, C3 generalisasi).
- [ ] Tulis ulang Introduction (gap + contributions).
- [ ] Tulis Methods lengkap: parameter, equations, flow diagram, split, annotation.
- [ ] Buat Figure 1 (pipeline) dan Figure 2 (contoh error).

**Siang**
- [ ] Tulis Results: seg overall + per-class + confusion matrix; OCR overall +
      per-field + ablation; independent test; baseline.
- [ ] Tulis Discussion: RQ answers, comparison, threats to validity, limitations.
- [ ] Update abstract dengan angka baru (satu sumber: `experiments/results/`).

**Malam**
- [ ] Isi `response_to_reviewers.md` per komentar.
- [ ] Cek semua tabel/figure dirujuk di teks.
- [ ] Cek sitasi/referensi.

**Checkpoint 3:** draft lengkap dengan angka final.
Jika belum: submit subset (OCR + independent test) dan minta extension untuk seg.

---

## Hari 4 — QA & submit

- [ ] **Consistency check:** abstract = results = tables = conclusion.
- [ ] Cek dataset counts: train/val/test + per-class.
- [ ] Cek unit analisis CI: cluster bootstrap per gambar, bukan n=30.
- [ ] Cek versi OCR, epoch, FP16/FP32, sodium/TDEE.
- [ ] Cek reproducibility checklist di `experiments/README.md`.
- [ ] Minta advisor baca 1 jam.
- [ ] Submit + response letter.

---

## Fitur yang bisa dipotong jika waktu habis

| Fitur | Dampak reviewer | Keputusan |
| :--- | :--- | :--- |
| Retrain penuh | Tinggi | Wajib jika data siap; jika tidak, jujur + extension |
| External test | Tinggi | Jangan dipotong; pakai public test split |
| OCR ablation | Tinggi | Jangan dipotong; paling murah |
| Baseline YOLO11n | Sedang | Bisa dipotong dengan penjelasan |
| Baseline Tesseract | Sedang | Bisa dipotong jika OCR ablation sudah kuat |
| 3 seeds | Sedang | Bisa 1 seed + limitation |
| HITL user study | Rendah | Ganti dengan simulasi koreksi |

---

## Kalimat jujur yang aman untuk paper

- “The original 50-epoch result could not be reproduced; we therefore report the
  new 80-epoch run with a fixed split.”
- “Due to the four-day revision window, the segmentation baseline is limited to
  YOLO11n-seg; a full architecture sweep is left for future work.”
- “The confidence interval is clustered by image; treating individual fields as
  independent would overstate precision.”
- “The external test set was frozen before training and was not used for model
  selection.”
