# Menjalankan NutriLabel v3 OCR (FastAPI + PaddleOCR)

## 1. Aktifkan server OCR

```powershell
Set-Location "D:\Labolatorium Anti Gravity\NutrilLabel v3 - Copy"
.\nutrilabel_env\Scripts\python.exe nutrilabel_v3_ppocr.py --serve --host 127.0.0.1 --port 8000
```

Tunggu sampai muncul: `Uvicorn running on http://127.0.0.1:8000`

Request pertama bisa 15–30 detik (model loading).

## 2. Buka aplikasi web

Gunakan folder **NutrilLabel v3 - Copy** (bukan v2.5), misalnya dengan Live Server / `python -m http.server 5500`.

Pastikan `app.js` memanggil `http://127.0.0.1:8000/api/ocr`.

## 3. Uji satu foto (CLI)

```powershell
.\nutrilabel_env\Scripts\python.exe nutrilabel_v3_ppocr.py -i termudah_ocrtest.jpg -o hasil.json
```

## 4. Evaluasi batch (skripsi)

Siapkan `nama_foto_gt.json` di folder yang sama dengan foto (lihat `Dataset/GT_TEMPLATE.json`).

```powershell
# Hanya foto yang sudah punya ground truth (cepat)
.\nutrilabel_env\Scripts\python.exe evaluasi_batch.py -f Dataset -r --gt-only

# Satu folder uji
.\nutrilabel_env\Scripts\python.exe evaluasi_batch.py -f Dataset\test --gt-only
```

Hasil detail: `evaluasi_hasil.json` di folder target.

**Penting:** `*_gt.json` harus dipasangkan dengan foto **panel Informasi Nilai Gizi**, bukan foto komposisi/bahan.
