# Cara Menjalankan Healthier

Langkah ini ditujukan untuk Windows menggunakan PowerShell. Gunakan Python 3.12 jika tersedia.

## 1. Instal dependensi

Buka PowerShell di folder proyek:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Jika PowerShell menolak aktivasi environment, jalankan Python melalui `.\.venv\Scripts\python.exe` sebagai pengganti perintah `python`.

## 2. Jalankan backend

Di terminal pertama, dari folder proyek:

```powershell
python backend\healthier_v3_ppocr.py --serve --host 127.0.0.1 --port 8000
```

Backend melayani API di `http://127.0.0.1:8000`. Pemrosesan pertama dapat memerlukan waktu untuk mengunduh atau memuat model OCR. Untuk segmentasi makanan melalui backend, file `models\best.pt` harus tersedia.

## 3. Jalankan frontend

Buka terminal PowerShell kedua di folder proyek dan aktifkan environment yang sama jika belum aktif. Jalankan:

```powershell
python -m http.server 5500 --bind 127.0.0.1
```

Buka <http://127.0.0.1:5500> di browser. Segmentasi offline di browser memerlukan `models\best_float16.tflite`.

Untuk informasi arsitektur, batasan model, dan dokumentasi teknis, lihat [README.md](README.md) serta folder [docs](docs/).
