# Menjalankan OCR NutriLabel v3 (FastAPI + PaddleOCR)

Semua perintah dijalankan dari **root proyek**.

## 1. Aktifkan server OCR

```powershell
# Aktifkan virtual environment bila memakai salah satunya
# .\nutrilabel_env\Scripts\Activate.ps1

python -m backend.app --serve --host 127.0.0.1 --port 8000
```

Tunggu sampai muncul `Uvicorn running on http://127.0.0.1:8000`.
Request pertama bisa memakan waktu 15–30 detik karena model dimuat ke memori.

## 2. Buka aplikasi web

```powershell
python -m http.server 5500 --directory frontend
```

Buka `http://127.0.0.1:5500`. Frontend memanggil API di
`http://127.0.0.1:8000` secara default. Untuk mengarahkan ke host lain, buka:

```text
http://127.0.0.1:5500/?api=http://192.168.1.10:8000
```

## 3. Uji satu foto (CLI)

```powershell
python -m backend.app -i data/samples/label-sample.jpg -o data/samples/label-sample.result.json
```

## 4. Evaluasi batch (riset)

Siapkan pasangan `nama_foto.jpg` + `nama_foto_gt.json` di folder dataset,
lalu jalankan:

```powershell
# Hanya foto yang sudah punya ground truth (cepat)
python scripts/evaluate.py -f dataset -r --gt-only

# Satu folder uji
python scripts/evaluate.py -f dataset/test --gt-only
```

Hasil detail disimpan sebagai `evaluasi_hasil.json` di folder target.

> **Penting:** `*_gt.json` harus dipasangkan dengan foto **panel Informasi
> Nilai Gizi**, bukan foto komposisi/bahan.
