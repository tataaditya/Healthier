# healthier (NutriLabel v3.5) 🥗

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-yellow.svg)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![Framework](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![AI-Engine](https://img.shields.io/badge/AI%20Engine-PaddleOCR%20%7C%20YOLOv11-orange.svg)](https://github.com/PaddlePaddle/PaddleOCR)

**healthier (NutriLabel v3.5)** adalah sistem pemantauan gizi harian personal yang komprehensif berbasis Kecerdasan Buatan (AI) untuk mendeteksi kandungan nutrisi kemasan makanan dan mengestimasi porsi makan secara *real-world*. 

Sistem ini mengintegrasikan pemrosesan citra tingkat lanjut (*Computer Vision*), sistem aturan klinis terpersonalisasi (*Clinical Health Rules Engine*), dan perhitungan kualitas nutrisi tingkat lanjut (*Extended Nutri-Score*) untuk membantu pengguna memantau kesehatan harian mereka sesuai dengan standar medis nasional dan internasional.

---

## 📸 Demo Tampilan Sistem

*(Silakan tambahkan tangkapan layar atau GIF antarmuka aplikasi Anda di sini untuk meningkatkan daya tarik visual)*
> **Catatan UX**: Antarmuka dirancang dengan tema **Sleek Dark Mode** menggunakan efek *glassmorphism*, indikator progres visual SVG, grafik analitik asupan 7 hari berbasis Chart.js, serta feedback real-time saat pengambilan foto.

---

## 🛠️ Analisis Sistem: Kelebihan & Kekurangan (Evaluasi Jujur & Objektif)

Untuk memberikan gambaran yang transparan bagi akademisi dan mentor, berikut adalah analisis kelebihan teknis serta keterbatasan performa riil (ground truth vs real-world deployment) dari sistem **healthier**:

### 🌟 Kelebihan (Strengths)

| Kategori | Kelebihan Teknis & Fungsional | Deskripsi & Implementasi Klinis |
| :--- | :--- | :--- |
| **Keandalan Offline** | **Dual-Mode Inference & Fallback** | Jika server lokal FastAPI mati, browser secara otomatis berpindah ke **Offline Fallback Mode** menggunakan model `@tensorflow/tfjs-tflite` (WASM) untuk segmentasi piring makan. |
| **Desain Mitigasi** | **Human-in-the-Loop Architecture** | Menyediakan **Segmentation Editor** dan **Form Koreksi Nilai Gizi** sebagai solusi mitigasi utama. Desain ini memastikan ketidakakuratan deteksi AI (baik OCR maupun YOLO) dapat dikoreksi langsung oleh pengguna sebelum masuk ke Clinical Rules Engine. |
| **Koreksi Presisi** | **Segmentation Editor & Canvas** | Pengguna dapat mengoreksi hasil segmentasi otomatis AI secara langsung lewat kuas warna (*brush paint*) bertransparansi 60% dan tools navigasi (Zoom, Pan, Minimap, Undo/Redo). |
| **Logika Medis** | **Clinical Rules Engine Terpersonalisasi** | Menerapkan parameter batas harian berdasarkan profil fisik & riwayat penyakit pengguna sesuai **PERKENI 2024** (untuk Diabetes) dan **JNC 8** (untuk Hipertensi). |
| **Skor Gizi Kaya** | **Extended Nutri-Score (FSA/Ofcom)** | Perluasan skor gizi tradisional dengan bonus mikronutrien (Vit C, Kalsium, Zat Besi, dll.), penyesuaian porsi riil (*Serving Fraction*), dan *Powder Reconstitution Mode*. |
| **Evaluasi Valid** | **Nutrient-Centric Batch Evaluator** | Mengabaikan metrik CER/WER tradisional yang kurang relevan secara klinis dan memprioritaskan *Field-Level Accuracy* untuk menjaga validitas angka gizi yang diekstrak. |

---

### ⚠️ Kekurangan & Batasan Akurasi (Limitations)

| Kategori | Batasan Sistem Saat Ini | Detail Teknis & Rencana Mitigasi Mendatang |
| :--- | :--- | :--- |
| **Akurasi Segmentasi** | **Bias Dataset YOLOv11-Seg (Isi Piringku)** | Meskipun dilatih menggunakan 3 folder dataset besar, dataset dasar `isipiringku_v2` masih didominasi oleh makanan bergaya Barat/luar negeri. Akibatnya, model **kurang optimal dan sering salah mengklasifikasikan makanan lokal khas Indonesia** (seperti nasi uduk, lauk bersaus/bumbu kacang, sayur berkuah, gorengan, dll.) karena perbedaan tekstur dan cara penyajian. |
| **Akurasi Deteksi Teks** | **Keterbatasan Riil Engine PaddleOCR** | Pada kemasan dunia nyata, PaddleOCR tidak selalu menghasilkan deteksi teks yang sempurna. Meskipun telah diakomodasi oleh filter preprocessing (*deglare*, *deskew*), PaddleOCR masih **sering mengalami kegagalan deteksi (false negative) atau salah baca** pada kemasan yang melengkung (seperti botol/kaleng), memiliki kontras warna teks-latar yang rendah, atau menggunakan fon/huruf dekoratif (*stylized fonts*). |
| **Beban Klien** | **WASM Model Loading Overhead** | Memuat pustaka TFJS-TFLite dan model `best_float16.tflite` secara lokal di browser membutuhkan memori RAM cukup tinggi pada perangkat seluler berspesifikasi rendah. |
| **Dependensi Server** | **PaddleOCR Python Dependency** | Backend OCR membutuhkan instalasi modul Python yang cukup besar (seperti `paddlepaddle` dan `paddleocr`), membutuhkan waktu setup awal yang relatif lama di lokal. |
| **Estimasi Dimensi** | **Perspektif 2D untuk Porsi** | Estimasi porsi makan didasarkan pada perbandingan jumlah piksel segmentasi secara 2D pada canvas, sehingga belum mampu mengukur ketebalan (kedalaman volume 3D) makanan secara presisi. |
| **Variasi Bahasa** | **Sensitivitas Struktur Kamus** | Rekonstruksi tabel gizi spasial masih sangat bergantung pada `KAMUS_NUTRISI` (Bahasa Indonesia & Inggris). Variasi struktur bahasa lain dapat menurunkan akurasi pemetaan kolom data gizi. |

---

## 📐 Arsitektur Alur Kerja Sistem

```mermaid
graph TD
    A[Image Input] --> B{Pilih Alur Analisis}
    
    %% Alur Kemasan
    B -->|Makanan Kemasan| C[Quality Check Kamera]
    C --> D[Preprocessing & ROI Crop]
    D --> E[PaddleOCR PP-OCRv4 Engine]
    E --> F[Semantic OCR Filtering]
    F --> G[Spatial Layout Reconstruction]
    G --> H[Koreksi & Normalisasi Unit]
    H --> I[Form Koreksi User]
    
    %% Alur Piring
    B -->|Isi Piringku| J{Cek Koneksi Server}
    J -->|Online| K[YOLOv11-Seg @ FastAPI]
    J -->|Offline Fallback| L[tfjs-tflite browser: Mobile Food Segmenter V1]
    K --> M[Estimasi Proporsi Piksel Makanan]
    L --> M
    M --> N[Interactive Paint Canvas / SegmentationEditor]
    N --> I
    
    %% Mesin Analisis
    I --> O[Extended Nutri-Score Engine]
    I --> P[Clinical Health Rules Engine]
    
    O --> Q[Modal Hasil & Saran Smart Tips]
    P --> Q
    
    Q -->|Simpan Makan| R[Dashboard & 7-Day Analytics]
    R --> S[Local Storage State]
```

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Prasyarat Sistem
* Python 3.9 atau lebih tinggi
* Node.js (opsional, untuk Live Server) atau Extensi VS Code Live Server
* Git

### 2. Konfigurasi Backend (FastAPI + PaddleOCR + YOLOv11)
Buka terminal Anda di folder utama proyek dan jalankan perintah berikut:

```powershell
# 1. Masuk ke lingkungan virtual (contoh untuk Windows)
.\nutrilabel_env\Scripts\activate

# 2. Jalankan Server API backend
python nutrilabel_v3_ppocr.py --serve --host 127.0.0.1 --port 8000
```
> **Catatan**: Tunggu hingga muncul log `Uvicorn running on http://127.0.0.1:8000`. Request analisis pertama mungkin memerlukan waktu 15-30 detik untuk memuat model ke memori.

### 3. Konfigurasi Frontend (Web Dashboard)
1. Buka file `app.js` dan pastikan konfigurasi API mengarah ke alamat backend lokal Anda:
   ```javascript
   const API_BASE_URL = "http://127.0.0.1:8000";
   ```
2. Jalankan server lokal untuk frontend (misalnya menggunakan Live Server pada VS Code, atau via python):
   ```bash
   python -m http.server 5500
   ```
3. Akses aplikasi melalui browser di alamat `http://127.0.0.1:5500`.

---

## 🧪 Evaluasi Model & Justifikasi Riset

Akurasi ekstraksi tabel nutrisi diuji secara batch menggunakan script `evaluasi_batch.py`. 

> [!IMPORTANT]
> **Mengapa Kami Tidak Menggunakan CER/WER Sebagai Metrik Utama?**
> Di dunia nyata, foto kemasan makanan mengandung banyak teks non-gizi (slogan promosi, tanggal kadaluwarsa, dll.) yang sengaja kami saring keluar menggunakan *Semantic Filter*. Membandingkan teks terfilter dengan transkrip asli akan merusak nilai CER/WER secara semu. Oleh karena itu, riset ini berfokus pada **Field-Level Accuracy** (seberapa akurat sistem mengekstrak angka kritis seperti gula, lemak, dan natrium) yang memiliki signifikansi klinis sesungguhnya.

Untuk menjalankan pengujian evaluasi batch:
```powershell
# Menguji hanya pada gambar yang memiliki data Ground Truth (*_gt.json)
python evaluasi_batch.py -f Dataset -r --gt-only
```
Hasil evaluasi komprehensif akan disimpan secara otomatis di file `evaluasi_hasil.json` dalam folder target.

---

## ⚕️ Dasaran Klinis (Clinical Health Rules)
- **Diabetes Melitus Tipe 2**: Disesuaikan dengan konsensus nasional **PERKENI 2024**.
- **Hipertensi**: Disesuaikan dengan rekomendasi global diet rendah garam **JNC 8** dan **DASH Diet**.

Sistem secara otomatis memberikan status keamanan makanan (`SAFE`, `WARNING`, `DANGER`) yang adaptif sesuai profil riwayat penyakit pengguna.

---

*Dikembangkan untuk keperluan Penelitian Riset AI dan Kesehatan Personal Terintegrasi.*
