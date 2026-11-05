# Audit: notebook vs manuscript

Sumber: `experiments/notebooks/training_model.ipynb` (output cell yang
tersimpan) dan artefak `models/isipiringku_v2/`.

> **Kesimpulan singkat:** notebook membuktikan bahwa ada **lebih dari satu run**
> yang tercampur. Output notebook yang tersimpan adalah run subset V5
> (4.840 train / 960 val, oversampling 12.082), sedangkan `args.yaml` +
> `results.csv` di repo menunjukkan run lain (100 epoch max, early stop 80)
> yang metriknya lebih dekat ke manuscript. Manuscript mencampur angka dari
> run/versi yang berbeda. Ini harus diperbaiki sebelum menambah eksperimen baru.

> **Caveat:** kalau metrics manuscript berasal dari run lain yang tidak ada di
> notebook/repo, kamu harus menyertakan config + output run tersebut. Kalau tidak,
> satu-satunya run yang bisa dipertanggungjawabkan adalah run yang bisa
> direproduksi ulang dengan split baru.

---

## 1. Rekonsiliasi 19.087 / 3.529 (kemungkinan besar bukan manipulasi)

Ada dua run berbeda di artefak:

### Run A — full corpus dengan official split (kemungkinan yang dipakai manuscript)

| Source | Train | Val/Test | Total |
| :--- | ---: | ---: | ---: |
| UECFoodPIXCOMPLETE | 9.000 | 1.000 | 10.000 |
| FruitSeg30 (80:20) | 1.575 | 394 | 1.969 |
| FoodSeg103 (official) | 4.983 | 2.135 | 7.118 |
| **Total** | **15.558** | **3.529** | **19.087** |

Jadi:
- **19.087 = total corpus**, bukan jumlah training.
- **15.558 = train**.
- **3.529 = validation (18,5%)**, bukan 20%.
- 15.558 + 3.529 = 19.087.

Ini menjelaskan angka manuscript tanpa perlu menganggapnya karangan. Kesalahannya
adalah **penamaan**: “trained on 19.087” seharusnya “total corpus 19.087;
15.558 train / 3.529 validation”. Klaim “80:20” juga tidak tepat; effective
split-nya 81,5:18,5 karena memakai official split tiap dataset.

### Run B — comparison run di output notebook (bukan main run)

| Tahap | Output notebook | Catatan |
| :--- | ---: | :--- |
| UECFood train | 3.500 | dipangkas dari 9.000 |
| UECFood test | 500 | dipangkas dari 1.000 |
| FruitSeg30 total | 800 | dipangkas dari 1.969; 640 train / 160 val |
| FoodSeg103 train | 700 | dipangkas dari 4.983 |
| FoodSeg103 val | 300 | dipangkas dari 2.135 |
| **Unique train** | **4.840** | 3.500 + 640 + 700 |
| **Unique val** | **960** | 500 + 160 + 300 |
| Setelah oversampling | **12.082 train** | duplikasi file minoritas (7.242 file) |
| Independent test | **tidak ada** | `val` dipakai untuk training/early stopping |

## 1b. Forensik artefak: mana main run, mana comparison run?

Ini bukti yang lebih kuat daripada ingatan:

1. **`results.csv` adalah main run.**
   - 80 baris/epoch, `time` terakhir **31.886 detik ≈ 8,85 jam**.
   - Mask mAP50 epoch 80 = **0,67136**, mAP50-95 = **0,52876**.
   - `args.yaml`: `epochs=100`, `patience=20`, `batch=16`, `cache=false`,
     `workers=2`, `save_dir=/kaggle/working/...`.
   - Waktu 8,85 jam untuk 80 epoch sangat konsisten dengan corpus besar
     (~15,5 ribu train), bukan subset 4.840.
2. **Notebook output adalah comparison run, bukan main run.**
   - Cell training utama (YOLO11s-seg) **dikomentari** (`# model.train(...)`),
     jadi main model tidak dilatih di notebook itu.
   - Cell comparison menjalankan 4 model Medium × 5 epoch dari
     `11:51` sampai `14:30` (2 jam 39 menit) pada subset 4.840/960.
   - Cell evaluasi comparison mencetak:
     `Weights YOLO11s-seg tidak ditemukan`.
   - Artinya notebook tidak punya main weights; yang ada hanyalah baseline
     comparison.
3. **`best.pt` main run tidak ada di repo.** Yang tersimpan hanya
   `models/isipiringku_v2/weights/last.pt` (epoch 80). Jadi angka “best model”
   di manuscript tidak bisa diverifikasi persis dari artefak yang ada.
4. **P/R 72,07/60,46 bukan metrik instance `results.csv`.**
   - `results.csv` epoch 80: mask P=72,778 / R=60,724; box P=72,877 / R=60,792.
   - Notebook menargetkan pixel-level CM: Accuracy 79,91 / P 72,07 / R 60,46 /
     F1 65,45. Jadi P/R manuscript kemungkinan **pixel-level**, bukan
     instance-level. Ini harus dipisahkan.
5. **Hardware:** `save_dir` Kaggle + waktu 8,85 jam → main run di **Kaggle T4**,
   bukan RTX 3050 6 GB. Klaim hardware manuscript salah.

**Kesimpulan forensik:**
- **Main run** = `results.csv` + `args.yaml` (80 epoch, 8,85 jam, Kaggle T4,
  cache false). Dataset count tidak tercatat di `args.yaml`, tetapi rekonsiliasi
  15.558/3.529/19.087 adalah penjelasan yang paling masuk akal.
- **Comparison run** = output notebook (4.840/960, 4 model Medium × 5 epoch).
- Manuscript mencampur **main-run metrics** dengan **comparison-run dataset
  counts** dan **pixel-level P/R**. Ini sumber utama inkonsistensi.

**Pertanyaan yang harus kamu jawab:** metrics manuscript (mask mAP50 67,06 /
mAP50-95 52,84) berasal dari checkpoint mana? `results.csv` epoch 80 (67,136 /
52,876) sangat dekat, tetapi `best.pt` tidak ada. Kalau tidak ada log `model.val()`
terpisah, anggap angka itu dari `last.pt`/epoch 80 dan verifikasi dengan
`eval_seg.py`. Jangan mengaku angka dari `best.pt` yang tidak ada.

## 2. Konfigurasi training yang benar-benar dipakai

Dari notebook dan `models/isipiringku_v2/args.yaml` / `results.csv`:

- Epoch maksimum: **100**
- Early stopping: **patience 20**
- Epoch berhenti: **80** (lihat `results.csv`)
- Optimizer: **AdamW**, `lr0=0.001`
- `imgsz=640`, `batch=16`, seed 42
- Augmentasi: `hsv_h=0.01`, `hsv_s=0.3`, `hsv_v=0.2`, `degrees=10`,
  `translate=0.1`, `scale=0.5`, `fliplr=0.5`, `mosaic=0.5`,
  `erasing=0.4`, `close_mosaic=10`
- Hardware training: **Kaggle Tesla T4, 15.6 GB VRAM** (bukan RTX 3050 6 GB)

Manuscript menulis **50 epoch** dan hardware **RTX 3050 6 GB / 8 GB RAM**.
Perbaiki: training di T4, deployment/lokal di RTX 3050 (jika memang diuji).

## 3. Klaim FP16

Notebook mencoba `model.export(format='tflite', half=True)` dan **gagal**:

```text
ERROR: quantize=16 (FP16) is not supported; format='litert' supports quantize=8
(INT8), 'w8a16', 'w8a32', 32 (FP32)
```

Artefak yang ada adalah **FP32** (`frontend/assets/models/best_float32.tflite`).
Jadi klaim “TFLite FP16, ~50% compression” tidak didukung. Pilihan:
- koreksi menjadi FP32, atau
- ekspor INT8/`w8a16` dengan Ultralytics versi baru dan evaluasi akurasi+latency,
  lalu laporkan hasilnya.

## 4. Klaim “weighted loss”

Notebook memakai **oversampling** kelas minoritas (duplikasi file), bukan
weighted loss. Manuscript menyebut “weighted loss”. Ganti menjadi
“class-balanced oversampling” atau implementasikan weighted loss yang benar.
Sebutkan juga jumlah unik vs efektif setelah oversampling.

## 5. Metrik precision/recall

Notebook menghitung **pixel-level confusion matrix** dan menargetkan:

```text
Accuracy=79.91% | Precision=72.07% | Recall=60.46% | F1=65.45%
```

Sementara `model.val()` menghasilkan metrik **instance-level** (box/mask).
Manuscript mencampur keduanya tanpa definisi. Perbaiki:
- Laporkan **pixel-level** (TP/FP/FN/TN, accuracy, P/R/F1) di satu tabel.
- Laporkan **instance-level** (box/mask P/R/mAP50/mAP50-95) di tabel terpisah.
- Jelaskan unit analisisnya.

## 6. Baseline yang sudah ada di notebook

Notebook sudah menjalankan fine-tuning 5 epoch untuk 4 model Medium:

| Model | Precision | Recall | Box mAP50 | Seg mAP50 |
| :--- | ---: | ---: | ---: | ---: |
| YOLOv8m-seg | 0.6196 | 0.5395 | 0.5627 | 0.5467 |
| YOLO11m-seg | 0.5962 | 0.5348 | 0.5458 | 0.5293 |
| YOLO26m-seg | 0.6190 | 0.5242 | 0.5489 | 0.5345 |

Catatan: hanya **5 epoch**, sedangkan model utama 80–100 epoch. Ini bukan
perbandingan apple-to-apple. Untuk reviewer:
- latih minimal satu baseline dengan budget epoch yang sama, atau
- laporkan tabel ini sebagai “short fine-tuning trend” dan jelaskan limitasinya.

Model utama (YOLO11s) tidak ditemukan saat evaluasi komparatif di notebook,
jadi tabel tersebut belum memuat model utama.

## 7. Rekomendasi resolusi

1. **Pilih satu jalur:**
   - **A (disarankan):** retrain pada **seluruh 19.087 citra unik** dengan split
     70/15/15 + external test; tidak memakai oversampling di run utama.
   - **B:** pertahankan run notebook (4.840 train / 960 val), tetapi hapus klaim
     19.087/3.529 dan jelaskan subset + oversampling.
2. Ganti “50 epoch” → “100 maksimum, early stop 80”.
3. Ganti hardware training → Kaggle Tesla T4 15.6 GB; sebut RTX 3050 hanya
   untuk deployment/lokal jika diuji.
4. Ganti klaim FP16 → FP32, atau ekspor ulang dan evaluasi.
5. Ganti “weighted loss” → “oversampling” (atau implementasi weighted loss).
6. Pisahkan metrik pixel-level dan instance-level.
7. Tambahkan independent test yang benar-benar tidak dipakai untuk training,
   early stopping, atau pemilihan threshold.
8. Untuk baseline, latih minimal satu model dengan epoch yang sama, atau
   nyatakan keterbatasannya secara eksplisit.

---

## 8. Kalimat aman untuk response letter

> “We audited the original notebook and found that the reported dataset counts
> referred to the full public corpora, whereas the training run used a
> computational subset (4,840 unique training images; 12,082 after
> oversampling) and 960 validation images. We have corrected the manuscript and,
> in the revision, retrained on the full 19,087-image corpus with a fixed
> 70/15/15 split and an independent test set. We also corrected the training
> hardware (Kaggle Tesla T4) and the FP16 claim (the original export failed; the
> deployed model is FP32).”
