# Handoff analisis lengkap — Healthier / NutriLabel (revisi jurnal)

> **Tujuan dokumen ini.** Satu tempat yang menjelaskan *seluruh* analisis revisi:
> latar belakang, apa yang diperbaiki, data mentah, angka final, arsitektur kode,
> audit ground truth, perbandingan VLM, keterbatasan, dan langkah berikutnya.
> Dokumen ini ditulis agar agen/AI lain (mis. Claude) atau penulis berikutnya bisa
> memahami konteks **tanpa harus membaca seluruh repo**.
>
> **Status:** snapshot per commit `d856764` (working tree bersih).
> **Aturan emas:** satu angka, satu sumber. Setiap angka di sini bisa ditelusuri ke
> `experiments/results/` atau ke `experiments/paper/final_numbers.md`.
> **Jangan mengarang angka.** Jika sesuatu belum diukur, tertulis "belum diukur".

---

## 0. Ringkasan eksekutif (baca ini dulu)

**Paper.** "Application of Deep Learning for Nutritional Segmentation and
Information Extraction in the Healthier Application". Sistem bernama **Healthier
(NutriLabel v3.5)**: menggabungkan segmentasi piring makan (*Isi Piringku*) +
OCR tabel gizi kemasan + aturan klinis deterministik (PERKENI 2024, JNC 8/DASH)
→ status SAFE / WARNING / DANGER.

**Kenapa direvisi.** Reviewer meminta (1) 2–3 kontribusi eksplisit,
(2) metodologi reproducible (parameter, persamaan, flow diagram, ablation),
(3) konsistensi jumlah dataset + anotasi + split + independent test + perluasan OCR,
(4) confusion matrix + per-class error + baseline + OCR ablation + generalisasi.
Selain itu, **audit internal menemukan banyak angka draft lama tidak konsisten
dengan artefak** (notebook vs manuscript), lihat §3.

**Apa yang berubah (hasil revisi, semua terverifikasi):**

1. **Segmentasi di-retrain bersih** pada split baru anti-leakage, hardware benar
   (Kaggle Tesla T4, bukan RTX 3050). Headline internal: **mask mAP50 71.66%,
   mAP50-95 56.91%** pada checkpoint epoch 48.
2. **Tiga external test** dijalankan (UEC test, FoodSeg103 val, FruitSeg30) →
   domain shift nyata, terutama **FruitSeg30 mAP50 17.06%** (buah murni).
3. **Per-class AP** dihitung ulang offline (pycocotools) untuk semua external test.
4. **OCR parser diperbaiki**: parser grid (baris-label × kolom-nilai) + pemecahan
   bbox angka yang tergabung vertikal. Pada pilot n=6: **21.74% → 47.83%**.
5. **Audit GT** menemukan 2 dari 6 GT OCR salah transkripsi → dikoreksi, semua
   pipeline di-skor ulang. Ini menaikkan plafon semua angka pilot.
6. **Perbandingan 9 pipeline** (edge baseline, edge grid, 7 VLM cloud) di atas
   data & metrik yang sama, sebagai bahan *future work* — **bukan** klaim inti.

**Keputusan scope yang mengikat (jangan diubah tanpa persetujuan user):**

- **Klaim inti C1/C2 tetap:** "OCR edge-based **tanpa LLM cloud** + aturan klinis
  **deterministik**". VLM hanya future work.
- **Perbandingan VLM lokal di-drop** sepenuhnya (user: "tidak butuh adu Lokal").
- **Angka OCR 83.8% lama TIDAK dipakai** (tidak tereproduksi).
- OCR headline belum layak diklaim; n=6 adalah **pilot eksploratif**, target ≥100 citra.
- 10–20 foto piring Indonesia hanya **analisis kualitatif domain-shift**, bukan benchmark.

---

## 1. Peta dokumen & file (di mana mencari apa)

| Kebutuhan | File |
| :--- | :--- |
| **Satu sumber kebenaran angka** | `experiments/paper/final_numbers.md` |
| Panduan menulis per bagian manuscript | `experiments/paper/audit/writing_guide.md` |
| Audit notebook lama vs manuscript | `experiments/paper/audit/notebook-audit.md` |
| Desain eksperimen (kontribusi, split, metrik) | `experiments/paper/audit/experimental-design.md` |
| Rekonsiliasi klaim draft OLD vs NEW | `experiments/paper/audit/claims.csv` |
| Template jawaban reviewer | `experiments/paper/audit/response_to_reviewers.md` |
| Rencana 4 hari (hard deadline) | `experiments/paper/audit/4-day-plan.md` |
| Audit ground truth pilot OCR | `experiments/paper/audit/gt_audit.md` |
| Patch GT (tracked) | `experiments/paper/audit/gt_patch_makanan_ringan.json` |
| Analisis VLM multi-model | `experiments/paper/audit/vlm_analysis.md` |
| Tabel siap tempel | `experiments/paper/tables/*.md` |
| Metrik per-kelas (mentah) | `experiments/results/segmentation/per_class_all.csv` |
| Ringkasan per-kelas (JSON) | `experiments/results/segmentation/per_class_summary.json` |
| Hasil edge OCR | `experiments/results/ocr_full{,_grid}/` |
| Hasil VLM per model | `experiments/results/ocr_vlm/<model>/` |
| Sumber & format dataset publik | `experiments/data_sources.md` |
| Pipeline revisi (cara jalan) | `experiments/README.md` |
| Arsitektur aplikasi + pipeline OCR | `backend/app.py` (section 3–10) |
| README proyek | `README.md` |

> **Catatan gitignore.** `experiments/ocr_data/`, `experiments/results/`,
> `experiments/dataset/`, `experiments/runs/`, `experiments/artifacts/` **tidak
> ter-commit** (data privat/besar). Yang ter-commit: dokumen `paper/`, `tables/`,
> `scripts/`, `configs/`, `environment/`. Jadi angka bisa *dibaca* dari tracked
> `final_numbers.md`/`tables/`, tetapi raw `predictions.jsonl` bersifat lokal.

---

## 2. Struktur repositori

```text
Healthier/
├── backend/app.py            # FastAPI + PP-OCR + YOLO-seg (inti pipeline)
├── frontend/                 # web statis (js/config, clinical-rules.js, app.js,
│                             #   assets/models/best_float32.tflite)
├── models/isipiringku_v2/    # artefak training YOLO lama (args.yaml, results.csv, weights)
├── Makanan Ringan/           # foto sumber pilot (privat)
├── scripts/                  # evaluasi/benchmark/diagnostik umum
├── experiments/              # SELURUH pipeline revisi
│   ├── README.md             # cara jalan (panjang)
│   ├── data_sources.md       # sumber + format + sitasi dataset
│   ├── configs/              # kelas FoodSeg + contoh class map
│   ├── environment/          # pip_freeze.txt, nvidia_smi.txt, python_version.txt
│   ├── notebooks/            # revision_kaggle.ipynb + training_model.ipynb (bukti audit)
│   ├── ocr_data/             # citra OCR + *_gt.json (privat, gitignored)
│   ├── runs/seg_main/        # checkpoint + results.csv + args.yaml (gitignored)
│   ├── results/              # semua output eval (gitignored)
│   ├── scripts/              # eval_ocr, eval_vlm, audit_gt, per_class_metrics, make_final_numbers
│   └── paper/                # dokumen ini + angka final + tabel + audit
├── tests/
└── docs/
```

Perintah inti (ringkas; detail di `experiments/README.md`):

```powershell
python experiments/scripts/eval_ocr.py       --help   # evaluasi OCR field-level
python experiments/scripts/eval_vlm.py       --help   # harness VLM (future work)
python experiments/scripts/audit_gt.py       --help   # re-score offline di bawah patch GT
python experiments/scripts/per_class_metrics.py       # AP per kelas dari predictions.json
python experiments/scripts/make_final_numbers.py      # regenerasi final_numbers.md + tabel
```

---

## 3. Masalah pada draft lama (kenapa semua ini dikerjakan)

Sumber: `experiments/paper/audit/notebook-audit.md` + `claims.csv`. Intinya **angka draft
lama mencampur beberapa run yang berbeda**, bukan satu run yang konsisten.

### 3.1 Rekonsiliasi 19.087 / 3.529

- 19.087 = **total corpus** (UEC 10.000 + FruitSeg30 1.969 + FoodSeg103 7.118),
  **bukan** jumlah training.
- Draft menulis "trained on 19,087" → salah penamaan.
- 15.558 train / 3.529 val (18.5%, bukan 20%) — ini split *official* tiap dataset.
- **Run nyata notebook** (output tersimpan) memakai **4.840 unique train / 960 val,
  lalu oversampling menjadi 12.082**; main training cell bahkan dikomentari.
- Main run di `models/isipiringku_v2/results.csv` = 80 baris, **8,85 jam**, `save_dir`
  Kaggle. Jadi **training di Kaggle T4**, bukan RTX 3050.

### 3.2 Daftar koreksi yang harus masuk manuscript (`claims.csv`)

| Bagian | Klaim lama | Nilai/​tindakan benar |
| :--- | :--- | :--- |
| Abstrak | trained on 19,087 | **11.684 train / 2.017 val / 5.013 external**; 18.714 unik |
| Table 1 | total 19.533 | Pisahkan segmentasi (18.714) vs OCR (446 → pilot) |
| Results | validation 3.529 | **2.017** (split 85:15) |
| Methods | 50 epoch | **max 100, early stop 67, best 48** |
| Methods | training RTX 3050 6 GB / 8 GB RAM | **Kaggle Tesla T4 15.6 GB**; RTX 3050 = deployment |
| Methods | "weighted loss" | **Run ini TIDAK memakai weighted loss maupun oversampling** (`train_seg.py`: `model.train(...)` polos). Oversampling/loss hanya ada di **run notebook lama** — jangan dipakai |
| Methods/Abstract | PP-OCRv5 | `backend/app.py:75` default **PP-OCRv4** (`paddleocr==3.7.0` = versi library, bukan model; v4 valid) |
| Results | field accuracy 83.8% | **Tidak tereproduksi** → pilot: 21.74% → 47.83% |
| Results | CI [78.6; 87.9] | CI **cluster bootstrap per gambar** (jauh lebih lebar) |
| Results | FP16 TFLite | Artefak **best_float32.tflite** (FP16 export gagal) |
| Results | mask mAP50 67.06 / 95 52.84 | **71.66 / 56.91** (checkpoint baru) |
| Results | P/R 72.07/60.46 | Itu **pixel-level**; instance mask P/R = **72.95/65.50** |
| Clinical | sodium 2.000 mg | **1.500 mg/hari (hipertensi), 2.300 mg/hari (umum)** (`clinical-rules.js:39,74`); verifikasi halaman sumber JNC 8/DASH dari dokumen asli |
| Clinical | TDEE 2.069 kcal | **2.008,5 kcal** (BMR 1.673,75 × 1.2) |
| Clinical | gula 5% universal | **5% TDEE hanya diabetes** (`:52`); non-diabetes **10%** (`:38`); sumber PERKENI 2024 perlu verifikasi dokumen asli |
| Clinical | WARNING ≥75% | **Bukan 75%** — lihat tabel ambang §3.4 |
| Usability | "no design flaws" | Ganti: "no severity-4; sisanya severity 2–3" **hanya jika data Nielsen milikmu ada**; jika tidak, hapus klaim |
| Results | Fruits 81.44 / Others 53.37 | Itu **internal**; external Fruits jatuh ke **17.06** |
| Baseline | perbandingan arsitektur | Notebook hanya 5-epoch subset → label **screening/pilot** |

### 3.3 Klaim yang gagal diverifikasi / artefak hilang

- `best.pt` run lama **tidak ada**; hanya `last.pt` (epoch 80). Klaim "best model"
  draft tidak bisa diverifikasi dari artefak lama.
- Export FP16 di notebook **gagal** (`quantize=16 not supported`).
- `experiments/results/ocr_*` **tidak ada** di draft → angka OCR 83.8% tanpa bukti.
- `experiments/dataset/dataset_summary.json` **tidak ada** (gitignored) → angka
  **373 duplikat** belum bisa direproduksi; regenerasi via `prepare_yolo_seg.py`.
- Data usability/Nielsen (jumlah temuan per severity, jumlah evaluator, skor)
  **tidak ada di repo** → wajib dari catatan milik user.

### 3.4 Clinical rules — source of truth (`frontend/js/clinical-rules.js`)

Ambang harus sama **persis** dengan kode. **Kode tidak memakai 75%.**

| Pemicu WARNING | Ambang di kode | Baris |
| :--- | :--- | :--- |
| Gula (diabetes & umum) | > **50%** batas harian | `:129`, `:168` |
| Lemak jenuh (dislipidemia/diabetes) | > **50%** | `:157` |
| Natrium aktual (hipertensi) | > **40%** batas harian | `:149` |
| Natrium per 100 g | HTN > **600 mg** (`:75`); umum > **800 mg** (`:46`) | `:145` |
| Protein (nefropati) | > **30%** | `:137` |
| **DANGER** | **hanya** saat gula **> 100%** batas harian | `:126`, `:165` |

Limit harian: gula diabetes `0.05×TDEE/4` (`:52`), non-diabetes `0.10×TDEE/4`
(`:38`); natrium umum **2.300 mg** (`:39`) / HTN **1.500 mg** (`:74`); lemak
jenuh `0.10×TDEE/9` (`:41`) / dislipidemia `0.07×TDEE/9` (`:58`); kolesterol
**200 mg** (`:59`); protein nefropati `weight_kg × 0.8` (`:65`).

Sodium dan lemak jenuh **mentok di WARNING**; hanya gula yang bisa DANGER.

---

## 4. Kontribusi inti & batas VLM

Dari `experimental-design.md`:

- **C1 — Guideline-grounded integration.** Menerjemahkan proporsi segmentasi piring
  (*Isi Piringku*) + nilai gizi label → keputusan klinis deterministik
  (PERKENI 2024 / JNC 8). Diuji end-to-end vs baseline unimodal dan "tanpa clinical rules".
- **C2 — Reproducible edge OCR + HITL.** Pipeline OCR dengan ablation per tahap
  preprocessing, evaluasi field-level dengan cluster bootstrap, dan kuantifikasi gain
  Human-in-the-Loop.
- **C3 — Generalization & failure analysis.** Evaluasi pada held-out source berbeda +
  primary Indonesian test set, dengan confusion matrix, per-class error, taksonomi gagal.

**Garis batas VLM (mengikat).**
Klaim C1/C2 menuntut *OCR edge-based tanpa LLM cloud + aturan deterministik*.
Menjadikan VLM jalur produksi menabrak klaim itu. Karena itu:

- VLM ditempatkan **hanya** sebagai *future work* (§11).
- VLM **lokal** tidak dibandingkan (di-drop atas permintaan user).
- Angka VLM **tidak masuk** `final_numbers.md`.

> **Jangan** menulis di manuscript bahwa "VLM mengalahkan pipeline edge" sebagai
> temuan utama. Itu hanya eksplorasi arah (n=6).

---

## 5. Dataset & split

### 5.1 Segmentasi (`tables/table_dataset.md`)

- Kandidat corpus **19.087** citra; **373 duplikat (dHash perceptual 8×8) dibuang**
  → **18.714 unik**. *(Angka 373 perlu diregenerasi; `dataset_summary.json`
  gitignored.)*
- Train **11.684** (UEC 7.478 + FoodSeg 4.206).
- Validation **2.017** (UEC 1.285 + FoodSeg 732) = 15% dari train pool.
- External test **5.013** (UEC test 948; FoodSeg103 val 2.096; FruitSeg30 1.969).

Instance count per kelas (train): Staple Food (UEC 4.188 + FoodSeg 3.369), Side
Dishes (3.614 + 3.685), Vegetables (1.466 + 9.223), Fruits (0 + 3.513), Others
(4.996 + 4.103).

**Penting:** FruitSeg30 **seluruhnya** external test → kelas `Fruits` di train hanya
dari FoodSeg103 (3.513 instance / 1.384 gambar). External test **tidak pernah** dipakai
untuk training / early stopping / pemilihan checkpoint.

Sumber & lisensi (`data_sources.md`):
- **UECFoodPIXCOMPLETE** — non-commercial research; label kelas di channel R.
- **FoodSeg103** — benchmark 103 kelas (remap ke 5 kelompok).
- **FruitSeg30** — CC BY 4.0, DOI 10.17632/vkht8pfsp3.3.

Remapping kelas → 5 kelompok *Isi Piringku*: Staple Food, Side Dishes, Vegetables,
Fruits, Others.

### 5.2 OCR

- Format GT: `{"takaran_saji_g": <gram>, "nutrisi": { kalori, lemak_total, lemak_jenuh,
  lemak_trans, kolesterol, natrium, karbohidrat, serat, gula, protein }}`.
- Aturan: nilai **per sajian** (bukan per 100 g); normalisasi satuan (kcal, g, mg, mcg);
  **field tidak ada = kosong, bukan 0**; desimal pakai titik.
- Desain yang disarankan: **100–150 citra**, split **by product/brand** (bukan random
  per gambar), 70/15/15, external test 30–50 citra dari toko/waktu berbeda.
- Realita saat ini: **pilot n=6** (`experiments/ocr_data/makanan_ringan/`), 46 field
  setelah koreksi GT. Ini **smoke test**, bukan benchmark.

---

## 6. Segmentasi — hasil lengkap

### 6.1 Konfigurasi run

| Item | Nilai |
| :--- | :--- |
| Arsitektur | YOLO11s-seg |
| Train / Val | 11.684 / 2.017 |
| External test | 5.013 (UEC 948; FoodSeg 2.096; FruitSeg 1.969) |
| Max epoch / patience | 100 / 20 |
| Early stop | epoch **67** |
| Best checkpoint | epoch **48** |
| Optimizer | AdamW (lr0 0.001, lrf 0.01, wd 0.0005, momentum 0.9) |
| Batch / imgsz | 16 / 640 |
| Seed | 42 (deterministic) |
| AMP | true |
| Augmentasi | hsv_h .01, hsv_s .3, hsv_v .2, degrees 10, translate .1, scale .5, fliplr .5, mosaic .5, erasing .4, close_mosaic 10 |
| Hardware | Kaggle Tesla T4, 15.6 GB |
| Training time | **4.01 h (14.420,7 s)** menurut `results.csv` |
| Ultralytics / Torch | 8.4.199 / 2.10.0+cu128 |
| Python | 3.12.13 |

> ⚠️ **Inkonsistensi kecil yang perlu dirapikan:** `tables/table_segmentation_internal.md`
> menulis training time **2,88 h**, sedangkan `final_numbers.md` dan `results.csv`
> (baris terakhir `time=14420.7`) menyatakan **4,01 h**. Pakai **4,01 h** (sumber =
> `results.csv`). Tabel internal tampaknya stale.
> ⚠️ `args.yaml` mencatat `cache: disk` dan `save_dir=/kaggle/working/...` (bukti T4).

### 6.2 Internal validation (best checkpoint, epoch 48)

| Metrik | Box | Mask |
| :--- | ---: | ---: |
| Precision | 72.87 | **72.95** |
| Recall | 66.04 | **65.50** |
| mAP50 | 72.37 | **71.66** |
| mAP50-95 | 60.11 | **56.91** |

### 6.3 External test (instance-level, mask & box)

| Dataset | Box P | Box R | Box mAP50 | Box 50-95 | Mask P | Mask R | Mask mAP50 | Mask 50-95 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FoodSeg103 val | 66.97 | 59.36 | 64.57 | 52.08 | 67.47 | 58.37 | 63.66 | 48.16 |
| UEC test | 67.82 | 68.08 | 70.30 | 57.86 | 67.42 | 67.44 | 69.15 | 54.30 |
| FruitSeg30 | 17.28 | 44.96 | 17.36 | 12.85 | 17.01 | 44.25 | 17.06 | 13.55 |

**Bacaan:** model memenuhi target di internal + dua sumber in-domain (UEC 69.15,
FoodSeg 63.66), tetapi **jatuh tajam di FruitSeg30 (17.06)**. Ini **domain shift**
(buah murni vs piring), **bukan typo**. Ini justru memperkuat C3.

### 6.4 Per-class mask AP (`per_class_all.csv`)

| Dataset | Kelas | GT inst. | Mask AP50 | Mask AP50-95 |
| :--- | :--- | ---: | ---: | ---: |
| UEC test | Staple Food | 543 | **85.70** | 73.28 |
| UEC test | Side Dishes | 463 | 70.71 | 53.46 |
| UEC test | Vegetables | 224 | 55.42 | 40.59 |
| UEC test | Others | 754 | 65.17 | 50.57 |
| UEC test | Fruits | 0 | — | — |
| FoodSeg val | Staple Food | 1.629 | 63.87 | 49.92 |
| FoodSeg val | Side Dishes | 1.751 | 61.61 | 47.21 |
| FoodSeg val | Vegetables | 4.718 | 65.83 | 48.27 |
| FoodSeg val | **Fruits** | 1.649 | **74.96** | 56.59 |
| FoodSeg val | Others | 2.004 | **51.24** | 39.15 |
| FruitSeg30 | Fruits | 1.993 | **17.29** | 13.90 |

Metodologi: `experiments/scripts/per_class_metrics.py` memakai **pycocotools**, GT
dibangun ulang dari **label polygon YOLO** → **tidak butuh citra**. Provenance:
`per_class_manifest.json` (`generated_at 2026-11-03`, predictions UEC 27.873,
FoodSeg 157.879, FruitSeg 26.230).

> **Jangan campur** "Fruits best 81.44% / Others worst 53.37%" (internal, draft lama)
> dengan external. Di external, Fruits **terburuk** (FruitSeg 17.06).

### 6.5 Pixel-level (`tables/table_segmentation_pixel.md`)

Contoh yang menjelaskan mode gagal FruitSeg30:

- FruitSeg30 Fruits: **precision 97.17, recall 2.14, F1 4.19** — model sangat
  "yakin" tapi melewatkan hampir semua buah; `tp=9.808.313`, `fn=448.632.975`.
- FruitSeg30 Staple Food: **FP 307.843.409** → model memprediksi banyak piksel
  non-buah sebagai Staple Food.
- Kelas lain di FruitSeg30: P/R/F1 = 0 (tidak ada GT).

Kelas buah di UEC juga P/R = 0 (memang tidak ada GT buah). Di FoodSeg103 semua 5
kelas masuk akal (F1 55–76).

### 6.6 Yang belum ada (blocker segmentasi)

- **Per-class internal** belum dihitung (citra internal tidak ter-mount saat ekspor).
  Angka internal 81.44/53.37 lama **jangan dipakai**.
- **Latency** belum diukur untuk model baru (angka 168 ms lama **jangan dipakai**).
- **Baseline full-budget** belum dilatih; yang ada hanya screening 5-epoch
  (`baselines_from_notebook.csv`): YOLOv8m-seg seg mAP50 54.67; YOLO11m-seg 52.93;
  YOLO26m-seg 53.45 — **tidak apple-to-apple** dengan run 100-epoch.

---

## 7. Arsitektur pipeline OCR (`backend/app.py`)

Section kode (untuk rujukan):

1. Cek dependencies.
2. Konfigurasi + `_env_flag` (baca env seperti `HEALTHIER_OCR_TABLE_GRID`).
3. **PP-OCR engine** (`PPOCREngine`, singleton; dukung PaddleOCR 2.x & 3.x;
   GPU via `NUTRILABEL_OCR_DEVICE`).
4. **Rekonstruksi tabel dari bbox** — bagian paling relevan:
   - `deteksi_layout_kolom`, `_header_kind`, `deteksi_layout_kolom`.
   - `_build_rows`, `_pecah_baris_multi_label`, `_pilih_nilai_untuk_label`.
   - **Grid parser:** `_baris_label_nutrisi`, `_pecah_teks_antar_baris`,
     `_pisah_bbox_bergabung`, `rekonstruksi_tabel_grid`.
   - Fallback energi/lemak: `ekstrak_aux_energi_lemak`, `terapkan_fallback_energi_lemak`.
   - `rekonstruksi_tabel_dari_bbox` (jalur utama).
5. **Pre-processing**: `analyze`, `deskew` (Hough), `deglare` (Telea inpainting),
   `normalisasi_cahaya` (CLAHE), `remove_overlay_watermark`, `crop_area_gizi`
   (ROI crop: landscape side / white panel / horizontal lines).
6. **Post-processing (ParseOCR v2)**: `levenshtein`, `fuzzy_match_nutrisi`,
   `ekstrak_angka`, `ekstrak_persen_akg`, `parse_baris_teks`, `_parse_baris_nutrisi`.
7. **Validasi klinis**: `validasi_klinis`.
8. **Evaluasi CER/WER + field accuracy**: `hitung_cer`, `hitung_wer`,
   `hitung_field_accuracy` (hanya menilai `per_sajian`; **mengabaikan `akg_persen`**;
   benar bila exact atau dalam toleransi **5%**).
9. **Pipeline utama**: `semantic_filter_ocr` (mempertahankan unit tunggal `mg`/`g`),
   `jalankan_ocr`, `hitung_field_accuracy`.
10. CLI.

### 7.1 Perbaikan grid parser (inti peningkatan OCR)

- Baris dibentuk dari **label nutrisi** (fuzzy match), bukan sekadar kedekatan bbox
  mentah.
- Kolom nilai diambil dari token numerik paling kiri (kolom "Per Sajian").
- `_pisah_bbox_bergabung` memecah bbox angka yang **tergabung vertikal** oleh PP-OCR
  (mis. `1725` = 1, 7, 2, 5) memakai grid baris label → kolom Per Sajian dan %AKG
  terpisah.
- **Semantic filter** mempertahankan token satuan tunggal (`mg`, `g`) agar field
  seperti natrium bisa di-parse.

### 7.2 Flag/environment penting

| Env | Fungsi |
| :--- | :--- |
| `HEALTHIER_OCR_TABLE_GRID` | **0** = baseline parser, **1** = grid parser (beda ablation OCR) |
| `NUTRILABEL_USE_TABLE_STRUCTURE` | **0** untuk melewati PP-Structure/SLANet (tidak berguna di kemasan plastik ini) |
| `NUTRILABEL_OCR_DEVICE` | `gpu:0` (diuji RTX 3050 6 GB) |
| `HEALTHIER_OCR_EXIF_ORIENT` | `exif_transpose` agar koordinat tabel tidak kacau (foto ponsel) |

> ⚠️ Perhatikan prefiksnya: grid = `HEALTHIER_...`, table-structure = `NUTRILABEL_...`.

---

## 8. OCR edge — hasil pilot lengkap (n=6, 46 field, GT terkoreksi)

Dua run di atas **kode terkini**, keduanya `NUTRILABEL_USE_TABLE_STRUCTURE=0`,
**hanya** `HEALTHIER_OCR_TABLE_GRID` yang berbeda (0 vs 1).

| Parser | Field accuracy | 95% CI | Exact | Wrong | Missing |
| :--- | ---: | :--- | ---: | ---: | ---: |
| Baseline (`ocr_full`, grid 0) | 21.74% (10/46) | [16.67, 25.58] | 21.74% | 13.04% | 65.22% |
| Grid revisi (`ocr_full_grid`, grid 1) | **47.83% (22/46)** | [29.17, 66.67] | 47.83% | 17.39% | 34.78% |

Metrik: field benar bila **exact atau dalam 5%** nilai per sajian; CI = **cluster
bootstrap per gambar**, 2.000 iterasi, seed 42.

### 8.1 Per-field (baseline)

| Field | n | exact | wrong | missing | accuracy | mean_rel_err |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| kalori | 6 | 2 | 1 | 3 | 0.333 | 0.29 |
| lemak_total | 6 | 2 | 1 | 3 | 0.333 | 2.33 |
| lemak_jenuh | 5 | 0 | 0 | 5 | 0.000 | — |
| lemak_trans | 2 | 1 | 0 | 1 | 0.500 | 0 |
| kolesterol | 1 | 0 | 0 | 1 | 0.000 | — |
| natrium | 6 | 2 | 0 | 4 | 0.333 | 0 |
| karbohidrat | 6 | 0 | 0 | 6 | 0.000 | — |
| serat | 2 | 1 | 0 | 1 | 0.500 | 0 |
| gula | 6 | 2 | 3 | 1 | 0.333 | ~10003 (outlier salah parse) |
| protein | 6 | 0 | 1 | 5 | 0.000 | 2.8 |

### 8.2 Per-field (grid revisi)

| Field | n | exact | wrong | missing | accuracy | mean_rel_err |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| kalori | 6 | 3 | 0 | 3 | 0.500 | 0 |
| lemak_total | 6 | 3 | 2 | 1 | 0.500 | 0.329 |
| lemak_jenuh | 5 | 1 | 1 | 3 | 0.200 | 0.5 |
| lemak_trans | 2 | 0 | 0 | 2 | 0.000 | — |
| kolesterol | 1 | 0 | 0 | 1 | 0.000 | — |
| natrium | 6 | 5 | 0 | 1 | 0.833 | 0 |
| karbohidrat | 6 | 3 | 2 | 1 | 0.500 | 0.771 |
| serat | 2 | 1 | 0 | 1 | 0.500 | 0 |
| gula | 6 | 3 | 2 | 1 | 0.500 | 2.754 |
| protein | 6 | 3 | 1 | 2 | 0.500 | 0.7 |

### 8.3 Per-image

| Gambar | baseline | grid |
| :--- | ---: | ---: |
| 20241108_131859 | 2/10 (low_confidence) | 4/10 (low_confidence) |
| 20241108_132045 | 1/7 (low_confidence) | 6/7 (low_confidence) |
| 20241108_132115 | 2/7 (low_confidence) | 4/7 (low_confidence) |
| 20241108_132308 | 2/8 (success) | 1/8 (low_confidence) |
| 20241108_133546 | 2/7 (low_confidence) | 5/7 (low_confidence) |
| 20241108_134337 | 1/7 (success) | 2/7 (success) |

**Bacaan:** gain terbesar dari grid = **natrium** (0.33 → 0.83) dan kalori;
`132308` justru turun (grid salah memisah baris di gambar itu). Mode gagal dominan
edge = **missing** (detektor menggabungkan grid angka / baris tak terdeteksi).

> **n=6 terlalu kecil.** Laporkan sebagai pilot/eksploratif. Jangan kutip 83.8%.

---

## 9. Audit ground truth (`gt_audit.md`)

### 9.1 Bagaimana ditemukan

Saat membandingkan 6 VLM berbeda, muncul pola: **semua model** sama-sama membaca
`natrium 60`, `karbohidrat 17`, `lemak_total 2` pada `20241108_134337`, padahal GT
lama menulis `40`, `17.9`, `2.5`. Karena banyak model independen sepakat, hipotesis:
**GT yang salah transkripsi**, bukan semua model salah seragam. Diverifikasi dengan
zoom crop foto asli.

### 9.2 Koreksi yang diterapkan

**`20241108_134337.jpg` (Milna Bubur Bayi):**

| Field | Label asli | GT lama | Jadi |
| :--- | ---: | ---: | ---: |
| lemak_total | 2 g | 2.5 | **2** |
| lemak_jenuh | tidak ada baris | 0.5 | **dibuang** |
| karbohidrat | 17 g | 17.9 | **17** |
| gula | 5 g (Gula Total) | 5.2 | **5** |
| natrium | 60 mg | 40 | **60** |

Baris asli: Lemak Total 2 g · Omega-3 50 mg · Omega-6 418 mg · Lemak Trans 0 g ·
Protein 3 g · Karbohidrat Total 17 g · Serat Pangan 1 g · Gula Total 5 g ·
Gula (Sukrosa) 2 g · Laktosa 3 g · Garam (Natrium) 60 mg. **Tidak ada** lemak jenuh.

**`20241108_132308.jpg` (Choco Pie):** gula **10 g** (GT lama 9).

Karena `lemak_jenuh` dibuang dari `134337`, total field menjadi **46** (dari 47).

### 9.3 Dampak (sensitivitas, dihitung offline oleh `audit_gt.py`)

| Pipeline | GT lama | GT terkoreksi (dipakai) |
| :--- | ---: | ---: |
| OCR baseline | 7/47 (14.9%) | 10/46 (21.7%) |
| OCR grid | 21/47 (44.7%) | 22/46 (47.8%) |
| VLM gemma4:cloud | 32/47 (68.1%) | 36/46 (78.3%) |
| VLM deepseek-v4.1-flash | 42/47 (89.4%) | 46/46 (100%) |
| VLM netra/deepseek-v4.1-flash | 42/47 (89.4%) | 46/46 (100%) |
| VLM kimi-k3 | 42/47 (89.4%) | 46/46 (100%) |
| VLM minimax-m3 | 41/47 (87.2%) | 45/46 (97.8%) |
| VLM glm-5.3-flash | 40/47 (85.1%) | 44/46 (95.7%) |
| VLM glm-5.3-flashx | 40/47 (85.1%) | 44/46 (95.7%) |

> Catatan: angka OCR "GT lama" (14.9%) berasal dari run **pra-perbaikan semantic
> filter**; run kode terkini + GT lama tidak lagi bisa direproduksi. Angka resmi OCR
> GT-baru di atasnya **tetap berbeda** dari 14.9% karena kode berubah, bukan hanya GT.

### 9.4 Bukti & privasi

Crop bukti disimpan **lokal** di `experiments/results/ocr_vlm/gt_audit/`
(`134337_kolom_nilai.png`, `134337_lemak_total.png`, `132308_gula.png`). Folder
`results/` di-gitignore → crop **tidak** ter-commit (citra label privat). Patch GT
di `experiments/paper/audit/gt_patch_makanan_ringan.json` (tracked, status "hasil audit").

### 9.5 Pelajaran metodologis

> Perbedaan antar-model yang **sangat besar** vs GT harus memicu **audit GT**
> sebelum disimpulkan "model lemah". Di sini mayoritas "wrong" VLM ternyata
> kesalahan GT.

---

## 10. Perbandingan VLM (future work, bukan klaim inti)

Data & metrik **identik** dengan evaluasi edge (exact atau ±5%, CI cluster
bootstrap). Harness: `experiments/scripts/eval_vlm.py`.

### 10.1 Hasil semua pipeline (GT terkoreksi)

| Pipeline | Akurasi | 95% CI | Exact | Wrong | Missing | Latensi/gambar |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| Edge baseline (PP-OCRv4) | 21.74% | [16.67, 25.58] | 21.74% | 13.04% | 65.22% | ±2 s (GPU) |
| Edge grid (revisi) | 47.83% | [29.17, 66.67] | 47.83% | 17.39% | 34.78% | ±2 s (GPU) |
| VLM `gemma4:cloud` (Ollama) | 78.26% | [64.44, 92.86] | 78.26% | 17.39% | 4.35% | 4.3 s |
| VLM `deepseek-v4.1-flash` | **100%** | [100, 100] | 100% | 0% | 0% | 8.6 s |
| VLM `netra/deepseek-v4.1-flash` | **100%** | [100, 100] | 100% | 0% | 0% | 4.7 s |
| VLM `kimi-k3` | **100%** | [100, 100] | 100% | 0% | 0% | 31.2 s |
| VLM `minimax-m3` | 97.83% | [93.02, 100] | 97.83% | 2.17% | 0% | 9.5 s |
| VLM `glm-5.3-flash` | 95.65% | [86.05, 100] | 95.65% | 4.35% | 0% | 15.0 s |
| VLM `glm-5.3-flashx` | 95.65% | [86.05, 100] | 95.65% | 4.35% | 0% | 7.5 s |

> **Jangan klaim 100% = "VLM sempurna".** Itu 46 field / 6 gambar; plafon naik
> setelah koreksi GT. Ini **indikasi arah**, bukan generalisasi.

### 10.2 Profil error — beban berpindah *jenis*, bukan sekadar mengecil

| | Edge grid | gemma4 | deepseek |
| :--- | ---: | ---: | ---: |
| Missing | 34.78% (16/46) | 4.35% (2/46) | 0% |
| Wrong | 17.39% (8/46) | 17.39% (8/46) | 0% |

- **Coverage/recall naik drastis.** Semua VLM nyaris selalu menemukan field-nya
  (missing 35% → 0–4%). Mode gagal khas edge (grid angka tergabung, baris tak
  terdeteksi) praktis hilang.
- **`gemma4` menukar missing dengan wrong** → "salah tapi yakin" (halusinasi digit).
- **Model frontier Helipod hampir bebas error**, sisa error khas per-model (bukan
  sistematis).

### 10.3 Error `gemma4:cloud` yang tersisa (10 error asli)

| Gambar | Field | GT | Prediksi | Rel. err |
| :--- | :--- | ---: | ---: | ---: |
| 131859 | lemak_jenuh | 1.5 | 1.0 | 33% |
| 131859 | gula | 10 | 3 | 70% |
| 131859 | natrium | 100 | 110 | 10% |
| 132115 | natrium | 65 | 85 | 31% |
| 132308 | kalori | 110 | 119 | 8% |
| 132308 | lemak_trans | 0 | (missing) | — |
| 132308 | lemak_jenuh | 3 | 0 | 100% |
| 132308 | protein | 1 | 2 | 100% |
| 134337 | serat | 1 | (missing) | — |
| 134337 | gula | 5 | 9 | 80% |

`131859` gula 3 vs 10 dan `134337` gula 9 vs 5 = **salah baris**.

### 10.4 Error model frontier (sangat sedikit)

- `minimax-m3`: natrium 85 vs 65 pada `132115`.
- `glm-5.3-flash` & `flashx`: lemak_jenuh & protein 1.1 vs 1 pada `132115`.
- `deepseek-v4.1-flash`, `netra/...`, `kimi-k3`: **nol error**.

### 10.5 Infrastruktur & akses

- **Ollama** (`--provider ollama`, `http://localhost:11434`): model vision gratis
  hanya `gemma4:cloud` (= alias `gemma4:31b-cloud`). `gpt-oss:120b-cloud` tanpa
  vision; lainnya 402/404/410.
- **Helipod** (`--provider openai`, `https://ai.helipod.io/v1`, key `hel_ai_...` di
  `.env`). `/models` = `hy3, deepseek-v4.1-flash, deepseek-v4-pro,
  netra/deepseek-v4.1-flash, glm-5.3, glm-5.2, glm-5.1, glm-5.3-flashx,
  glm-5.3-flash, kimi-k3, minimax-m3`.
  - **Harus kirim browser User-Agent** (jika tidak → Cloudflare 403 error 1010).
  - Format OAI: `[{type:text},{type:image_url,image_url:{url:"data:image/jpeg;base64,..."}}]`,
    `max_tokens=1024`, `temperature=0`.
  - Tanpa vision / error 400: `hy3`, `deepseek-v4-pro`, `glm-5.3`, `glm-5.2`, `glm-5.1`.
- **`--rescore`**: skor ulang offline dari `predictions.jsonl` (prediksi tersimpan),
  **tanpa** panggil API lagi (dipakai setelah koreksi GT).
- Prompt menekankan JSON per **sajian** (bukan per 100 g), suhu 0, gambar ≤1600 px,
  guard anti-halusinasi "jangan mengarang angka".

### 10.6 Risiko VLM (wajib dicatat kalau kelak dipakai)

1. **Halusinasi digit** — error "salah tapi yakin" lebih berbahaya di konteks klinis
   daripada missing, karena aturan deterministik hilir tidak bisa mendeteksinya.
   Perlu lapisan verifikasi + HITL wajib untuk field sensitif (natrium/gula).
2. **Non-determinisme** output cloud antar-run/versi → bertentangan dengan klaim C2.
3. **Biaya & ketersediaan** — model vision kuat butuh kredit.
4. **Klaim edge** — memakai cloud menabrak C1 ("tanpa LLM cloud").
5. **n=6** — terlalu kecil untuk klaim apa pun.

### 10.7 Arah future work (rekomendasi)

- **Hybrid:** VLM untuk **menetapkan struktur/kolom/baris** (kelemahan fatal edge),
  OCR deterministik untuk **membaca digit persis** (kelemahan VLM), lalu lapisan
  verifikasi silang untuk field sensitif.
- **On-device VLM** belum diuji serius (di luar scope; tidak dilaporkan).
- **Perbesar sampel dulu** (≥100 gambar) sebelum klaim apa pun.

---

## 11. Sintesis temuan

1. **Integrasi multi-sumber berhasil diukur secara jujur.** Segmentasi kuat
   in-domain (mAP50 71.66 internal, 69.15 UEC, 63.66 FoodSeg) tetapi gagal di
   FruitSeg30 (17.06) → domain shift nyata, bukan disembunyikan.
2. **Edge OCR punya batas struktural, bukan sekadar "salah baca".** Mode gagal
   dominan = **missing** (grid angka tergabung/baris tak terdeteksi). Parser grid
   memperbaiki sebagian (natrium 0.33 → 0.83; total 21.7% → 47.8%).
3. **VLM mengubah jenis error.** Nyaris menghapus missing, tetapi memunculkan
   halusinasi digit (gemma4) atau hampir sempurna (frontier). Ini *argumen* untuk
   hybrid, bukan untuk mengganti klaim inti.
4. **Audit GT wajib.** 2/6 GT salah → mengubah kesimpulan besar. Sensitivitas
   didokumentasikan.
5. **Semua angka lama yang tidak tereproduksi diganti** dengan run baru + sumber
   jelas (lihat `claims.csv`).

---

## 12. Hardware & environment

**Training segmentasi** (`experiments/environment/nvidia_smi.txt`):

```text
Tesla T4 (2 GPU terlihat), Driver 580.178.04, CUDA 13.0, 15360 MiB per GPU
```

**Deployment/OCR lokal:** NVIDIA RTX 3050 6 GB (driver 13.5),
`paddlepaddle-gpu==3.2.2` + `paddlex 3.7.2` + `paddleocr 3.7.0`, CUDA 12.6 build.
Verifikasi sm_86. `pip_freeze.txt` + `python_version.txt` (3.12.13) tersedia di
`experiments/environment/`.

> ⚠️ **Dua environment berbeda**: training di T4 (Kaggle), OCR/VLM di laptop
> RTX 3050. Jangan campur di manuscript.

Angka lingkungan lama: Ultralytics 8.4.199, Torch 2.10.0+cu128.

---

## 13. Riwayat git (relevan)

```text
d856764  fix(ocr): koreksi GT pilot & regenerasi angka edge + VLM
83dcca9  feat(audit): skrip re-score GT offline + patch kandidat pilot
1382af3  docs(vlm): bandingkan 6 model VLM cloud + audit ground truth pilot OCR
7ca094d  docs(ocr): analisis pilot VLM cloud sebagai bahan future work
131b5f4  feat(ocr): parser tabel berbasis grid + filter satuan, hasil pilot n=6
44398cd  refactor: merubah konfigurasi path menjadi lebih rapih
b333751  docs: add demo system screenshot to README
81077ed  docs: add honest evaluation of YOLOv11-Seg and PaddleOCR limitations to README
fece2a6  Initial commit: NutriLabel v3.5 (healthier) production ready
```

HEAD = `d856764665c0a72760307003f81c9c60bed2e730`; working tree **bersih**.

---

## 14. Keterbatasan & open items (jujur)

| # | Item | Status | Dampak | Tindakan |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Per-class internal | Belum | Reviewer minta per-class; baru ada external | Ekspor `per_class.csv` saat citra internal ter-mount |
| 2 | OCR headline | Blocked | n=6 tak layak klaim | Kumpulkan ≥100 citra, split by brand, rerun `eval_ocr.py` |
| 3 | Latency | Belum diukur | Klaim 168 ms tak berdasar | Ukur di RTX 3050 + CPU, mean/median/P95 |
| 4 | Baseline full-budget | Belum | Reviewer minta perbandingan setara | Latih YOLO11n/YOLOv8s budget penuh atau labeli screening |
| 5 | FP16 vs FP32 | Archived | Artefak FP32 | Klaim FP32 atau ekspor ulang + evaluasi |
| 6 | Usability severity | Perlu bukti | "no design flaws" overclaim | Laporkan severity 0–4 |
| 7 | Indonesian plate set | Kualitatif saja | Tak bisa klaim benchmark | 10–20 foto sebagai analisis domain-shift kualitatif |
| 8 | OCR ablation lengkap | Belum | Reviewer minta per tahap | Jalankan `ablate_ocr.py` semua varian |
| 9 | `takaran_saji_g` | Tak diskor | `hitung_field_accuracy` abaikan | Sebut sebagai batasan metrik / tambah metrik |
| 10 | Training-time tabel internal | Stale (2.88 vs 4.01 h) | Inkonsistensi kecil | Samakan ke `results.csv` (4.01 h) |

**Bottleneck internal per-class:** citra internal (train/val) tidak ada di disk saat
penghitungan; `per_class_metrics.py` butuh `predictions.json` + label yang sudah ada
untuk external, tetapi internal perlu citra ter-mount untuk mengekspor prediksi.

---

## 15. Langkah berikutnya (prioritas)

**Prioritas jika waktu mepet (dari `4-day-plan.md`):**
`OCR ablation > external test > baseline full > segmentation ablation > HITL study`.

1. **Perluas OCR pilot** ke ≥100 citra (split by brand) → satu-satunya cara mengklaim
   angka OCR headline. Ini blocker utama.
2. **Jalankan `ablate_ocr.py`** semua varian (raw → +deskew → +deglare → +CLAHE →
   +ROI → +semantic → +fuzzy → full) untuk memenuhi permintaan reviewer.
3. **Ekspor per-class internal** saat citra tersedia.
4. **Ukur latency** model baru di RTX 3050 + CPU.
5. **Baseline full-budget** (YOLO11n-seg atau YOLOv8s-seg) atau labeli screening.
6. **Rapikan inkonsistensi kecil** (training time 2.88 → 4.01 h; sodium/TDEE di paper
   = kode).
7. **Tulis manuscript** menurut `writing_guide.md`, semua angka dari `final_numbers.md`.

**Keputusan yang perlu konfirmasi user lanjutan:**
- Apakah menambah citra OCR (target ≥100) realistis dalam window deadline?
- Apakah retrain baseline full-budget dijalankan?
- Apakah Indonesian plate set dibuat kualitatif atau kuantitatif (perlu anotasi mask)?

---

## 16. Aturan integritas (jangan dilanggar)

- **Satu angka, satu sumber**: `final_numbers.md` / `experiments/results/`.
- Jangan mengutip **83.8%** atau **CI [78.6; 87.9]** lama.
- Jangan mengutip **mask mAP50 67.06 / 52.84** lama → pakai **71.66 / 56.91**.
- Jangan menulis "trained on 19.087" / "3.529 validation" / "50 epoch" /
  "RTX 3050 training" / "weighted loss" / "PP-OCRv5" / "FP16".
- Jangan jadikan VLM klaim inti; jangan klaim VLM 100% sebagai generalisasi.
- Jangan campur metrik **instance** vs **pixel**.
- Jangan menyatakan hasil yang belum diukur (latency, per-class internal).

---

## Lampiran A — GT pilot per gambar (setelah koreksi)

| Gambar | Saji (g) | kalori | lemak_total | lemak_jenuh | lemak_trans | kolesterol | natrium | karbohidrat | serat | gula | protein |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 131859 | 20 | 90 | 3 | 1.5 | 0 | 0 | 100 | 16 | 1 | 10 | 1 |
| 132045 | 10 | 50 | 2.5 | 1 | — | — | 5 | 7 | — | 2 | 1 |
| 132115 | 15 | 70 | 2.5 | 1 | — | — | 65 | 11 | — | 5 | 1 |
| 132308 | 12 | 110 | 4 | 3 | 0 | — | 40 | 18 | — | 10 | 1 |
| 133546 | 10 | 60 | 3.5 | 1.5 | — | — | 50 | 6 | — | 0 | 1 |
| 134337 | 25 | 100 | 2 | — | — | — | 60 | 17 | 1 | 5 | 3 |

Field yang tidak dicetak di label = tidak ada di GT (bukan 0).

## Lampiran B — Perintah reproduksi

```powershell
# --- VLM gemma4 via Ollama ---
python experiments/scripts/eval_vlm.py `
  --provider ollama --model gemma4:cloud `
  --data experiments/ocr_data/makanan_ringan `
  --out experiments/results/ocr_vlm/gemma4_cloud `
  --label "Makanan Ringan pilot (VLM gemma4:cloud)" --bootstrap 2000

# --- VLM Helipod (OpenAI-compatible; API_KEY dari .env) ---
python experiments/scripts/eval_vlm.py `
  --provider openai --base-url https://ai.helipod.io/v1 --model deepseek-v4.1-flash `
  --data experiments/ocr_data/makanan_ringan `
  --out experiments/results/ocr_vlm/helipod_deepseek_v4_1_flash `
  --label "VLM cloud deepseek-v4.1-flash" --bootstrap 2000

# --- Skor ulang offline (tanpa panggil model; setelah GT berubah) ---
python experiments/scripts/eval_vlm.py --rescore `
  --provider openai --model deepseek-v4.1-flash `
  --data experiments/ocr_data/makanan_ringan `
  --out experiments/results/ocr_vlm/helipod_deepseek_v4_1_flash `
  --label "VLM cloud deepseek-v4.1-flash" --bootstrap 2000

# --- Audit / sensitivitas GT offline ---
python experiments/scripts/audit_gt.py `
  --data experiments/ocr_data/makanan_ringan `
  --patch experiments/paper/audit/gt_patch_makanan_ringan.json

# --- OCR edge: baseline vs grid (GT terkoreksi) ---
$env:NUTRILABEL_OCR_DEVICE = "gpu:0"
$env:NUTRILABEL_USE_TABLE_STRUCTURE = "0"
$env:HEALTHIER_OCR_TABLE_GRID = "0"
python experiments/scripts/eval_ocr.py --data experiments/ocr_data/makanan_ringan `
  --out experiments/results/ocr_full --bootstrap 2000 --save-raw
$env:HEALTHIER_OCR_TABLE_GRID = "1"
python experiments/scripts/eval_ocr.py --data experiments/ocr_data/makanan_ringan `
  --out experiments/results/ocr_full_grid --bootstrap 2000 --save-raw

# --- Regenerasi angka & tabel ---
python experiments/scripts/make_final_numbers.py
python experiments/scripts/per_class_metrics.py
```

## Lampiran C — Glosarium istilah (ID → EN)

| Istilah | Arti |
| :--- | :--- |
| takaran saji | serving size |
| kalori | energy (kcal) |
| lemak_total / jenuh / trans | total / saturated / trans fat |
| kolesterol | cholesterol |
| natrium | sodium |
| karbohidrat / serat / gula | carbohydrate / fiber / sugar |
| protein | protein |
| per sajian | per serving |
| makanan ringan | snacks |
| Isi Piringku | Indonesian "My Plate" guideline |
| field accuracy | akurasi tingkat field |
| cluster bootstrap | bootstrap per gambar (unit = gambar, bukan field) |

## Lampiran D — Inventori hasil (mentah)

```text
experiments/results/
├── segmentation/
│   ├── per_class_all.csv, per_class_summary.json, per_class_manifest.json,
│   │   per_class_predictions_all.csv
│   ├── uec_test_instance/   (metrics.json, per_class.csv/.md, predictions.json, curves, CM)
│   ├── uec_test_pixel/      (pixel_metrics.csv, pixel_confusion_matrix.csv)
│   ├── foodseg_val_instance/
│   ├── foodseg_val_pixel/
│   ├── fruitseg_instance/
│   └── fruitseg_pixel/
├── ocr_full/       (summary.json/.md, per_field.csv, per_image.csv, errors.csv, predictions.jsonl)
├── ocr_full_grid/  (idem)
└── ocr_vlm/
    ├── gemma4_cloud/                       (idem + raw/)
    ├── helipod_deepseek_v4_1_flash/
    ├── helipod_netra_deepseek_v4_1_flash/
    ├── helipod_kimi_k3/
    ├── helipod_minimax_m3/
    ├── helipod_glm_5_3_flash/
    ├── helipod_glm_5_3_flashx/
    └── gt_audit/                           (crop bukti, lokal/privat)
```

---

*Dokumen ini merangkum seluruh analisis revisi Healthier per snapshot `d856764`.
Semua angka dapat dilacak ke `experiments/results/` dan `experiments/paper/final_numbers.md`.*
