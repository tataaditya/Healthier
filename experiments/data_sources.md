# Dataset sources & format notes

Semua dataset publik di bawah ini dapat diunduh ulang. **Catat versi, tanggal
akses, dan lisensi** di paper.

---

## 1. UECFoodPIXCOMPLETE

> Perintah `prepare_yolo_seg.py` di dokumen ini adalah jalur **lokal**.
> Jalur Kaggle sudah otomatis di `experiments/notebooks/revision_kaggle.ipynb`.

- Halaman: <https://mm.cs.uec.ac.jp/uecfoodpix/>
- Unduhan: `UECFOODPIXCOMPLETE.tar` (±759 MB)
- Lisensi: **non-commercial research only**.
- Struktur:

```text
UECFOODPIXCOMPLTE/data/UECFoodPIXCOMPLTE/
├── train/img/*.jpg
├── train/mask/*.png
├── test/img/*.jpg
├── test/mask/*.png
├── category.txt      # id + nama kelas
├── train.txt
└── test.txt
```

- Mask: label kelas ada di **channel R** (R=class id, G=B=0). Script memakai
  `--mask-channel auto` (RGB → red).
- `category.txt` berisi `id name`; buat `classes.txt` berisi **nama saja**
  (satu per baris) agar indeks baris = `id - 1`.
- Gunakan `train/` untuk train+val, dan `test/` sebagai test internal atau
  external source terpisah. Contoh:

```powershell
python experiments/scripts/local/prepare_yolo_seg.py `
  --source uec_train=raw/UECFoodPIXCOMPLTE/data/UECFoodPIXCOMPLTE/train/img=raw/UECFoodPIXCOMPLTE/data/UECFoodPIXCOMPLTE/train/mask=raw/uec_classes.txt `
  --source uec_test=raw/UECFoodPIXCOMPLTE/data/UECFoodPIXCOMPLTE/test/img=raw/UECFoodPIXCOMPLTE/data/UECFoodPIXCOMPLTE/test/mask=raw/uec_classes.txt `
  --out experiments/dataset --test-source uec_test
```

> Untuk split train/val yang reproducible di dalam `train/`, gunakan dua source
> terpisah atau pisahkan folder sebelum menjalankan script.

---

## 2. FoodSeg103

- Halaman: <https://xiongweiwu.github.io/foodseg103.html>
- Repo: <https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1>
- Unduhan: <https://research.larc.smu.edu.sg/downloads/datarepo/FoodSeg103.zip>
  (password: `LARCdataset9947`)
- Lisensi: Apache-2.0 (kode); cek lisensi dataset di halaman resmi.
- Struktur umum:

```text
FoodSeg103/
├── Images/img_dir/train/*.jpg
├── Images/img_dir/val/*.jpg
├── Annotations/ann_dir/train/*.png
├── Annotations/ann_dir/val/*.png
└── category_id.txt
```

- Mask berupa PNG semantic; cek apakah pixel 0 = background atau class 0.
  Sesuaikan `--mask-offset`.
- Gunakan `val` sebagai test internal, atau jadikan source `foodseg` sebagai
  external test dengan `--test-source foodseg`.
- Contoh:

```powershell
python experiments/scripts/local/prepare_yolo_seg.py `
  --source foodseg_train=raw/FoodSeg103/Images/img_dir/train=raw/FoodSeg103/Annotations/ann_dir/train=raw/foodseg_classes.txt `
  --source foodseg_val=raw/FoodSeg103/Images/img_dir/val=raw/FoodSeg103/Annotations/ann_dir/val=raw/foodseg_classes.txt `
  --out experiments/dataset --test-source foodseg_val
```

---

## 3. FruitSeg30

- Halaman: <https://data.mendeley.com/datasets/vkht8pfsp3/3>
- DOI: `10.17632/vkht8pfsp3.3`
- Lisensi: **CC BY 4.0**.
- Struktur: 30 folder kelas, masing-masing berisi `Images/` dan `Mask/`
  (512×512, mask PNG).
- Karena kelas ditentukan oleh **nama folder**, gunakan:

```powershell
python experiments/scripts/local/prepare_yolo_seg.py `
  --source fruit=raw/FruitSeg30/Images=raw/FruitSeg30/Mask=raw/fruit_classes.txt `
  --out experiments/dataset --recursive --folder-class-source fruit `
  --test-frac 0.15
```

`fruit_classes.txt` bisa berisi daftar nama kelas (untuk dokumentasi); mode
`--folder-class-source` memakai nama folder sebagai kelas dan heuristic
kelompok gizi.

> Jika FruitSeg30 dijadikan **external test** (`--test-source fruit`), pastikan
> kelas buah tetap ada di train dari source lain. Jika tidak, gunakan split
> 85:15 seperti contoh di atas dan buat external test terpisah.

---

## 4. Dataset OCR private

Tidak ada di repo. Format yang diharapkan:

```text
experiments/ocr_data/test/
├── product_001.jpg
├── product_001_gt.json
├── product_002.jpg
└── product_002_gt.json
```

`product_001_gt.json`:

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

Aturan anotasi yang harus ditulis di paper:
- Ambil nilai **per sajian** (bukan per 100 g), kecuali field yang memang hanya
  tersedia per 100 g; tulis aturan konversinya.
- Normalisasi satuan: kcal, g, mg, mcg.
- Jika field tidak ada di kemasan, **kosongkan** (jangan isi 0).
- Untuk field desimal, gunakan titik.
- Dua annotator untuk minimal 20% data; hitung Cohen's kappa.
- Simpan foto asli; jangan crop manual sebelum evaluasi (pipeline yang crop).

---

## 5. Sitasi dataset

Tambahkan ke daftar referensi:

- Okamoto, K., & Yanai, K. (2021). UEC-FoodPIX Complete: A Large-scale Food
  Image Segmentation Dataset. ICPR Workshop MADiMa.
- Wu, X., Fu, X., Liu, Y., Lim, E.-P., Hoi, S. C. H., & Sun, Q. (2021). A
  Large-Scale Benchmark for Food Image Segmentation. ACM Multimedia.
- Shamrat, F. M. J. M., Shakil, R., Idris, M. Y. I., Akter, B., & Zhou, X.
  (2024). FruitSeg30_Segmentation Dataset & Mask Annotations. Mendeley Data,
  V3. https://doi.org/10.17632/vkht8pfsp3.3
