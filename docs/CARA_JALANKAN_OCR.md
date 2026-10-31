# Menjalankan Healthier v3.5 Pro — OCR (FastAPI + PaddleOCR)

## ⚠️ Prasyarat: Environment Sudah Diperbaiki

Versi library yang diperlukan (sudah terinstall setelah perbaikan):

| Library | Versi |
|---------|-------|
| Paddle | 3.3.1 |
| PaddleOCR | 3.7.0 |
| PaddleX | 3.7.1 |
| **PyTorch** | **2.6.0+cu126** _(stabil, bukan nightly)_ |
| NumPy | 2.3.5 _(kompatibel paddlex <2.4)_ |
| Torchvision | 0.21.0 |

---

## 1. Aktifkan Server OCR

```powershell
Set-Location "D:\Labolatorium Anti Gravity\NutrilLabel v3.5 Pro"
.\healthier_env\Scripts\python.exe backend\healthier_v3_ppocr.py --serve --host 127.0.0.1 --port 8000
```

Tunggu sampai muncul:
```
Uvicorn running on http://127.0.0.1:8000
```

> ⏱️ **Request pertama bisa 15–30 detik** (model loading PP-OCRv4 + SLANet).

---

## 2. Buka Aplikasi Web

Gunakan folder **NutrilLabel v3.5 Pro**, buka dengan Live Server / `python -m http.server 5500`.

`app.js` sudah memanggil `http://127.0.0.1:8000/api/ocr` secara otomatis.

---

## 3. Uji Satu Foto (CLI)

```powershell
.\healthier_env\Scripts\python.exe backend\healthier_v3_ppocr.py -i assets\img\foto_kemasan.jpg -o hasil.json
```

---

## 4. Cek Health Server

```powershell
curl http://127.0.0.1:8000/api/health
```

Respons OK:
```json
{"status": "ok", "model": "PP-OCRv4", "gpu": false}
```

---

## 5. Evaluasi Batch (Skripsi)

```powershell
# Hanya foto yang sudah punya ground truth (cepat)
.\healthier_env\Scripts\python.exe backend\evaluasi_batch.py -f Dataset -r --gt-only

# Satu folder uji
.\healthier_env\Scripts\python.exe backend\evaluasi_batch.py -f Dataset\test --gt-only
```

Hasil detail: `evaluasi_hasil.json` di folder target.

---

## Troubleshooting

### ❌ Error `WinError 127 / shm.dll`

PyTorch versi nightly/dev yang bermasalah. Jalankan:

```powershell
.\healthier_env\Scripts\pip.exe uninstall torch torchvision -y
.\healthier_env\Scripts\pip.exe install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
```

### ❌ Error `numpy incompatible`

```powershell
.\healthier_env\Scripts\pip.exe install "numpy>=1.24,<2.4" --force-reinstall
```

### ❌ CORS Error di browser

Pastikan server sudah berjalan di port 8000, dan buka web via Live Server (bukan file://).

---

**Penting:** `*_gt.json` harus dipasangkan dengan foto **panel Informasi Nilai Gizi**, bukan foto komposisi/bahan.
