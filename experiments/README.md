# Reproducible revision experiments

Folder ini menyiapkan **satu sumber kebenaran** untuk revisi jurnal: dataset,
split, training, evaluasi, ablation, dan tabel paper. Semua angka di manuscript
harus berasal dari file di `experiments/results/`.

> **Baca dulu:**
> - [`paper/notebook-audit.md`](paper/notebook-audit.md) — angka notebook vs manuscript.
> - [`paper/experimental-design.md`](paper/experimental-design.md) — desain split,
>   screening vs final comparison, independent test, dan protokol anotasi.
>
> **Aturan utama:** jangan mencampur angka dari run yang berbeda. Setiap tabel
> harus bisa ditelusuri ke `metrics.json` / `summary.json` / `ocr_ablation.csv`.

---

## 0. Yang harus disiapkan

- Dataset publik: UECFoodPIXCOMPLETE, FruitSeg30, FoodSeg103.
- Dataset OCR private + ground truth `*_gt.json`.
- GPU RTX 3050 6 GB (atau cloud/Kaggle) + Python 3.10+.
- Waktu: rencana 4 hari ini sangat padat; jalankan training di malam hari.
- **Cek GPU**: `check_environment.py` harus menunjukkan `CUDA avail. : True`
  dan PaddlePaddle GPU terpasang. Jika masih `+cpu`, install build CUDA yang
  sesuai driver sebelum training. Alternatif cepat: jalankan training di Kaggle
  (dataset publik + notebook), lalu unduh `best.pt` dan `results.csv`.

```powershell
# dari root repo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Cek environment terlebih dahulu:

```powershell
python experiments/scripts/local/check_environment.py
```

Untuk reproduksibilitas, jalankan:

```powershell
New-Item -ItemType Directory -Force experiments/results | Out-Null
pip freeze > experiments/results/pip_freeze.txt
git rev-parse HEAD > experiments/results/git_commit.txt
```

---

## 1. Struktur folder

```text
experiments/
├── notebooks/
│   ├── revision_kaggle.ipynb   # ★ JALUR UTAMA (Kaggle): dataset, split, training,
│   │                           #   evaluasi seg, uji foto Indonesia
│   ├── training_model.ipynb    # notebook asli (bukti audit)
│   └── archive/                # salinan run lama / varian backup
├── scripts/
│   ├── eval_ocr.py             # ★ OCR: field-level + bootstrap CI
│   ├── ablate_ocr.py           # ★ OCR: ablation preprocessing
│   ├── make_tables.py          # generator tabel gaya lama
│   ├── make_final_numbers.py   # ★ sumber angka final + tabel paper
│   ├── per_class_metrics.py    # ★ AP per kelas dari predictions.json + label
│   └── local/                  # alternatif jalur lokal (tidak wajib untuk Kaggle)
│       ├── check_environment.py
│       ├── make_class_map.py
│       ├── prepare_yolo_seg.py
│       ├── train_seg.py
│       ├── eval_seg.py
│       └── eval_seg_pixel.py
├── configs/                    # mapping kelas + contoh
├── ocr_data/                   # private OCR images (gitignored)
├── dataset/
│   ├── revision/               # train/val + manifest + summary (gitignored)
│   └── external/               # UEC test, FoodSeg val, FruitSeg30 (gitignored)
├── runs/seg_main/              # checkpoint + results.csv + args.yaml (gitignored)
├── results/segmentation/       # metrics, predictions, kurva (gitignored)
├── environment/                # pip freeze, nvidia-smi, versi Python (tracked)
├── artifacts/                  # arsip zip hasil Kaggle (gitignored)
└── paper/                      # audit, desain, response letter, plan, tabel
    ├── tables/                 # tabel siap tempel ke Word/LaTeX
    ├── final_numbers.md        # ★ satu sumber kebenaran angka
    └── writing_guide.md        # ★ panduan menulis per bagian manuscript
```

★ = dipakai di jalur Kaggle. Folder `scripts/local/` hanya untuk yang ingin
menjalankan pipeline di luar Kaggle (datanya identik dengan notebook).

### Jalur mana yang harus dipakai?

| Kebutuhan | Pakai |
| :--- | :--- |
| Training & evaluasi segmentasi | `notebooks/revision_kaggle.ipynb` |
| Evaluasi uji foto Indonesia | `notebooks/revision_kaggle.ipynb` (cell 14) |
| Evaluasi & ablation OCR | `scripts/eval_ocr.py`, `scripts/ablate_ocr.py` |
| Tabel & angka final paper | `scripts/make_final_numbers.py` |
| Build dataset/training di PC sendiri | `scripts/local/*` |

Tambahkan ke `.gitignore` bila belum ada:

```gitignore
experiments/dataset/
experiments/runs/
experiments/results/
experiments/ocr_data/
experiments/artifacts/
```

---

## 2. Dataset segmentasi

Sumber unduhan, lisensi, dan format tiap dataset ada di
[`data_sources.md`](data_sources.md).

> **Untuk jalur Kaggle, lewati bagian 2–4.** Notebook
> `notebooks/revision_kaggle.ipynb` membangun dataset, split, dan training secara
> otomatis. Bagian di bawah hanya untuk eksekusi lokal via `scripts/local/`.

### 2.1 Susun source

Contoh struktur after download:

```text
raw/
├── UECFoodPIXCOMPLETE/
│   ├── images/
│   ├── masks/          # PNG mask, pixel value = class id
│   └── classes.txt     # satu kelas per baris
├── FruitSeg30/
│   ├── images/
│   ├── masks/
│   └── classes.txt
└── FoodSeg103/
    ├── images/
    ├── masks/
    └── classes.txt
```

Jika nama file mask berbeda (mis. `_mask`), gunakan `--mask-stem-suffix _mask`.

### 2.2 Buat mapping kelas

Mapping manual dari notebook asli (UECFood + FoodSeg103) bisa diekstrak dengan:

```powershell
python experiments/scripts/local/make_class_map.py `
  --uec-category raw/UECFOODPIXCOMPLETE/.../category.txt `
  --out-dir experiments/configs
```

Lalu pakai:

```powershell
--class-map experiments/configs/uecfood_class_map.csv `
--class-map experiments/configs/foodseg_class_map.csv
```

Jika tidak memakai mapping manual, script memakai heuristic nama kelas.
**Wajib review** `class_map.csv`; heuristic bisa salah untuk hidangan Jepang.

Contoh lengkap:

```powershell
python experiments/scripts/local/prepare_yolo_seg.py `
  --source uec=raw/UECFoodPIXCOMPLETE/images=raw/UECFoodPIXCOMPLETE/masks=raw/UECFoodPIXCOMPLETE/classes.txt `
  --source fruit=raw/FruitSeg30/images=raw/FruitSeg30/masks=raw/FruitSeg30/classes.txt `
  --source foodseg=raw/FoodSeg103/images=raw/FoodSeg103/masks=raw/FoodSeg103/classes.txt `
  --out experiments/dataset `
  --test-frac 0.15 `
  --copy-mode hardlink
```

Ini membuat **internal test** 15% yang stratified. Untuk **external test**
(reviewer meminta sumber berbeda), siapkan folder terpisah dan evaluasi dengan
`python experiments/scripts/local/eval_seg.py --data experiments/dataset/external/<source>/data.yaml`.
Jangan memakai external test untuk memilih checkpoint.

> **Peringatan:** `--test-source fruit` akan mengeluarkan **semua** FruitSeg30
> dari training. Jika buah hanya ada di FruitSeg30, kelas `Fruits` menjadi 0 di
> train. Cek `dataset_summary.json`; tambahkan subset FruitSeg30 ke train atau
> pakai external test terpisah.

Edit `experiments/dataset/class_map.csv`, lalu jalankan ulang dengan
`--class-map experiments/dataset/class_map.csv` jika ada kelas yang salah.

Catatan format:
- UEC-FoodPix: label ada di channel **R**; gunakan `--mask-channel auto`
  (default) atau `--mask-channel red`.
- FruitSeg30: kelas ditentukan nama folder; gunakan `--recursive
  --folder-class-source fruit`.
- FoodSeg103: cek `--mask-offset` (0 vs 1) sesuai encoding mask.

**Rekomendasi split untuk reviewer:**
- Default: split stratified **70/15/15** (train/val/test) per source/class.
- Script **tidak melakukan oversampling**. Jika kamu ingin meniru notebook,
  duplikasi train secara terpisah dan laporkan **unique vs effective** count.
- `--test-source fruit`: FruitSeg30 menjadi **external test**; pastikan kelasnya
  masih ada di train (lihat peringatan di bawah).
- Jangan memakai test untuk memilih checkpoint atau threshold.

### 2.3 Cek hasil

```powershell
python -c "import json; print(json.load(open('experiments/dataset/dataset_summary.json')))"
```

Pastikan:
- jumlah train/val/test masuk akal,
- semua kelas muncul di train dan test,
- `dataset_summary.json` mencatat warnings (duplikat, mask hilang).

---

## 3. Training segmentasi

Contoh untuk RTX 3050 6 GB:

```powershell
python experiments/scripts/local/train_seg.py `
  --data experiments/dataset/data.yaml `
  --model yolo11s-seg.pt `
  --epochs 100 --patience 20 --imgsz 640 --batch 8 --device 0 `
  --seed 42 --project experiments/runs --name seg_main
```

Catatan:
- Paper **wajib menyebut angka yang benar**: jika early stop di epoch 80 dari
  maksimum 100, tulis “80 epochs (early stopping, patience 20, max 100)”.
- Jika VRAM habis, turunkan `--batch 4` atau `--imgsz 512`, dan catat di paper.
- Jangan mengubah threshold/test set setelah melihat hasil test.

Output: `experiments/runs/seg_main/weights/best.pt`, `results.csv`, `args.yaml`.

---

## 4. Evaluasi segmentasi

```powershell
python experiments/scripts/local/eval_seg.py `
  --model experiments/runs/seg_main/weights/best.pt `
  --data experiments/dataset/data.yaml `
  --split test `
  --imgsz 640 --batch 1 `
  --out experiments/results

# Pixel-level TP/FP/FN/TN (Tabel 4.3/4.5 di skripsi)
python experiments/scripts/local/eval_seg_pixel.py `
  --model experiments/runs/seg_main/weights/best.pt `
  --data experiments/dataset/data.yaml `
  --split test --conf 0.25 `
  --out experiments/results
```

Output:
- `experiments/results/seg_test/metrics.json`
- `per_class.csv`
- `confusion_matrix.png` / `confusion_matrix_normalized.png`
- `summary.md`
- `experiments/results/seg_pixel_test/pixel_metrics.csv`,
  `pixel_confusion_matrix.csv`, `summary.md`

**Penting:** jangan mencampur metrik *instance-level* (`model.val`) dengan
*pixel-level* (TP/FP/FN/TN). Laporkan keduanya di tabel terpisah dan jelaskan
satuannya.

Untuk baseline, latih model pembanding (mis. `yolo11n-seg.pt` atau
`yolov8s-seg.pt`) dengan split yang sama, lalu simpan hasilnya sebagai
`experiments/results/baselines/<method>.json` dengan field:
`method`, `mask_map50`, `mask_map50_95`, `latency_ms`.

---

## 5. Dataset & evaluasi OCR

### 5.1 Format ground truth

```json
{
  "takaran_saji_g": 20.4,
  "nutrisi": {
    "kalori": 100,
    "lemak_total": 4,
    "lemak_jenuh": 2,
    "protein": 1,
    "karbohidrat": 15,
    "serat": 1,
    "gula": 8,
    "natrium": 110
  }
}
```

Letakkan `foto.jpg` + `foto_gt.json` di `experiments/ocr_data/test/`.
Gunakan **minimal 100 citra** dari brand/toko berbeda; 30 citra terlalu kecil
untuk klaim generalisasi.

### 5.2 Evaluasi

```powershell
python experiments/scripts/eval_ocr.py `
  --data experiments/ocr_data/test `
  --out experiments/results/ocr_full `
  --bootstrap 2000
```

Output:
- `summary.json`: field accuracy, **95% CI cluster bootstrap per gambar**,
  image-level accuracy, status exact/close/wrong/missing.
- `per_field.csv`: akurasi per field.
- `errors.csv`: semua error + flag.
- `predictions.jsonl` bila `--save-raw`.

**Penting untuk paper:** laporkan unit analisis dengan jelas. Jika CI dihitung
atas field, jangan klaim n = 30 gambar. Gunakan cluster bootstrap per gambar
seperti script ini.

### 5.3 Pilot dataset `makanan_ringan`

`experiments/ocr_data/makanan_ringan/` berisi 6 foto kemasan + ground truth
manual (`*_gt.json`). Ini **pilot/smoke test**, bukan benchmark final.

```powershell
python experiments/scripts/eval_ocr.py `
  --data experiments/ocr_data/makanan_ringan `
  --out experiments/results/ocr_full_grid `
  --bootstrap 2000 --save-raw --label "Makanan Ringan pilot (grid parser)"
```

Hasil pilot (n=6, 47 field):

| Parser | Field accuracy | 95% CI | Exact | Wrong | Missing |
| :--- | ---: | :--- | ---: | ---: | ---: |
| Baseline (`ocr_full`) | 14,89% | [4,65; 22,92] | 14,89% | 17,02% | 68,09% |
| Grid revisi (`ocr_full_grid`) | **44,68%** | [24,49; 66,67] | 44,68% | 19,15% | 36,17% |

Perbaikan parser tabel (`backend/app.py`):
- `rekonstruksi_tabel_grid`: baris dari label nutrisi (fuzzy match), bukan
  kedekatan bbox mentah; kolom nilai diambil dari token numerik paling kiri.
- `_pisah_bbox_bergabung`: pecah bbox angka yang tergabung vertikal oleh PP-OCR
  (mis. `1725` = 1, 7, 2, 5) memakai grid baris label, sehingga kolom Per Sajian
  dan %AKG terpisah.
- Semantic filter mempertahankan token satuan tunggal (`mg`, `g`) agar field
  seperti natrium dapat di-parse.

Catatan teknis:
- Foto ponsel menyimpan orientasi di EXIF. Pipeline sekarang memakai
  `PIL.ImageOps.exif_transpose` (flag `HEALTHIER_OCR_EXIF_ORIENT`) supaya
  koordinat tabel tidak kacau. Matikan lewat ablation untuk mengukur efeknya.
- PaddleOCR 3.x memakai `device=` dan hasil `rec_texts/rec_scores/rec_polys`;
  PaddleOCR 2.x memakai `use_gpu=` dan format list. `PPOCREngine` menangani
  keduanya.
- Jalankan di GPU dengan `NUTRILABEL_OCR_DEVICE=gpu:0` (diuji pada RTX 3050
  6 GB, `paddlepaddle-gpu==3.2.2`). Set `NUTRILABEL_USE_TABLE_STRUCTURE=0`
  untuk melewati PP-Structure (SLANet) yang belum menghasilkan tabel HTML pada
  foto kemasan plastik ini.

> **n=6 terlalu kecil.** Laporkan sebagai pilot/eksploratif; jangan mengutip
> angka 83,8% dari draft. Target untuk klaim utama tetap >=100 citra.

### 5.4 Analisis pembanding VLM (future work, bukan inti)

Alat bantu pertimbangan ke depan; **bukan** bagian pipeline yang dipublikasikan.
Harness: `experiments/scripts/eval_vlm.py`. Dokumen lengkap:
`experiments/paper/vlm_analysis.md`.

Hasil pilot n=6 (metrik identik dengan `eval_ocr.py`):

| Pipeline | Field accuracy | 95% CI | Missing | Wrong |
| :--- | ---: | :--- | ---: | ---: |
| Edge baseline | 14,89% | [4,65; 22,92] | 68,09% | 17,02% |
| Edge grid (revisi) | 44,68% | [24,49; 66,67] | 36,17% | 19,15% |
| VLM `gemma4:cloud` | **68,09%** | [44,90; 91,11] | **6,38%** | 25,53% |

Temuan: VLM nyaris menghapus mode gagal "missing", tetapi errornya bergeser ke
"salah tapi yakin" (berisiko di konteks klinis). Arah future work: **hybrid**
(VLM untuk struktur/kolom, OCR deterministik untuk digit) + lapisan verifikasi.
Model cloud lain (`minimax-m3`, `deepseek-v4.1-flash`) menolak dengan HTTP 402.

```powershell
python experiments/scripts/eval_vlm.py `
  --data experiments/ocr_data/makanan_ringan `
  --model gemma4:cloud `
  --out experiments/results/ocr_vlm/gemma4_cloud `
  --label "Makanan Ringan pilot (VLM gemma4:cloud)" --bootstrap 2000
```

---

## 6. Ablation preprocessing OCR

```powershell
python experiments/scripts/ablate_ocr.py `
  --data experiments/ocr_data/test `
  --out experiments/results/ocr_ablation `
  --bootstrap 2000
```

Variant: `raw`, `+deskew`, `+deglare`, `+clahe`, `+roi`, `+semantic`,
`+fuzzy`, `+fallback`, `+validation`, `full`.

Untuk membandingkan versi OCR:

```powershell
python experiments/scripts/ablate_ocr.py --data ... --out experiments/results/ocr_ablation_v4 --ocr-version PP-OCRv4
python experiments/scripts/ablate_ocr.py --data ... --out experiments/results/ocr_ablation_v5 --ocr-version PP-OCRv5
```

Baseline OCR yang murah:
- Tesseract (tanpa preprocessing),
- PaddleOCR tanpa preprocessing (`raw` variant),
- pipeline penuh.

---

## 7. Tabel paper

```powershell
# Sumber angka final + tabel segmentasi (dipakai untuk manuscript)
python experiments/scripts/make_final_numbers.py

# AP per kelas dari predictions.json + label (tanpa perlu citra)
python experiments/scripts/per_class_metrics.py

# Generator tabel gaya lama (membaca struktur seg_test/seg_pixel_test)
python experiments/scripts/make_tables.py
```

Output utama:
- `experiments/paper/final_numbers.md` — satu sumber kebenaran angka.
- `experiments/paper/writing_guide.md` — panduan menulis per bagian manuscript.
- `experiments/paper/tables/table_dataset.md`
- `experiments/paper/tables/table_segmentation_internal.md`
- `experiments/paper/tables/table_segmentation_external.md`
- `experiments/paper/tables/table_segmentation_pixel.md`
- `table_ocr_summary.md`, `table_ocr_per_field.md`, `table_ocr_ablation.md`,
  `table_baselines.md` (dari `make_tables.py`, jika hasil OCR sudah ada).

Catatan: `make_tables.py` membaca `experiments/results/`. Untuk hasil dari
Kaggle, ekstrak `revision_results.zip` ke `experiments/results/` terlebih dulu.

Salin ke Word/LaTeX, lalu **cek ulang setiap angka** di abstract, results, dan
conclusion.

---

## 7b. Audit notebook asli

Baca [`paper/notebook-audit.md`](paper/notebook-audit.md). Notebook asli
memakai **4.840 train / 960 val** (lalu oversampling menjadi 12.082), bukan
19.087/3.529 seperti di manuscript. Notebook juga membuktikan export FP16 gagal
dan training berjalan di **Tesla T4**, bukan RTX 3050. Ini harus diperbaiki
sebelum menulis hasil baru.

## 8. Jadwal 4 hari (hard deadline)

### Hari 1 — data & mulai training
- Siapkan Kaggle Dataset: UECFoodPIXCOMPLETE + FruitSeg30.
- Jalankan `notebooks/revision_kaggle.ipynb` sampai cell dataset summary.
- Review distribusi kelas; lanjutkan training utama (overnight).
- Siapkan/annotasi OCR test 100+ citra.
- Audit angka lama vs artefak (`experiments/paper/claims.csv`).

### Hari 2 — evaluasi & baseline
- Jalankan cell evaluasi external test di notebook.
- Set `RUN_BASELINE=True` (sesi terpisah) bila waktu memungkinkan.
- `eval_ocr.py` pada 100+ citra.
- `ablate_ocr.py` (semua variant).

### Hari 3 — analisis & tulis
- `make_tables.py`.
- Buat confusion matrix + error gallery (segmentation & OCR).
- Tulis ulang Methods/Results dengan angka baru.
- Susun 3 kontribusi eksplisit.
- Isi response-to-reviewer.

### Hari 4 — QA & submit
- Cek konsistensi angka (abstract = results = tables).
- Cek sitasi/referensi.
- Cek reproducibility checklist.
- Submit + response letter.

> Jika Hari 1 molor (download/annotation), **jangan** memaksakan retrain penuh.
> Prioritaskan OCR ablation + independent test, dan minta extension untuk
> segmentasi. Jangan pernah mengarang angka.

---

## 9. Checklist reproducibility untuk paper

- [ ] Dataset source, versi, lisensi, tanggal akses.
- [ ] Jumlah citra per source/class/split.
- [ ] Kriteria anotasi + inter-annotator agreement (minimal subset).
- [ ] Deduplikasi (hash) dan split tanpa leakage.
- [ ] Environment: Python, torch, CUDA, ultralytics, paddleocr, paddle, OS.
- [ ] Hardware: GPU, VRAM, RAM, CPU.
- [ ] Hyperparameter lengkap (epochs, patience, batch, imgsz, optimizer, lr, augmentation).
- [ ] Seed dan deterministic setting.
- [ ] Perintah training/evaluasi persis.
- [ ] Checkpoint dan split files tersedia.
- [ ] Semua tabel berasal dari `experiments/results/`.
- [ ] Independent test tidak dipakai untuk tuning.
- [ ] Code & data availability statement.
