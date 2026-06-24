# Analisis Model YOLO - Dataset "Isi Piringku"

Analisis ini membandingkan 5 folder pelatihan model **YOLO11s-seg** (Instance Segmentation) untuk mendeteksi dan mensegmentasi komponen makanan berdasarkan pedoman **"Isi Piringku"** (Makanan Pokok, Lauk Pauk, Sayuran, Buah-buahan, dan Lainnya).

---

## 📊 Tabel Perbandingan Performa Model

Berikut adalah ringkasan performa dan konfigurasi dari setiap iterasi pelatihan model:

| Nama Run / Folder | Status Pelatihan | Epoch Selesai / Konfigurasi | Patience | Best Box mAP50 (Epoch) | Best Box mAP50-95 | Best Mask mAP50 (Epoch) | Best Mask mAP50-95 | Box Precision / Recall | Mask Precision / Recall | Augmentasi Data (Hyperparameters) |
|---|---|---|---|---|---|---|---|---|---|---|
| **isipiringku_v1** | Gagal / Interrupted | 0 / 100 | - | - | - | - | - | - | Tidak ada weights yang tersimpan |
| **isipiringku_v1-2** | Dihentikan Awal | 2 / 100 | 20 | 0.4102 (Ep 2) | 0.2964 | 0.3862 (Ep 2) | 0.2668 | 0.4817 / 0.4105 | 0.4809 / 0.3905 | Tinggi (Mosaic: 0.5, Rotasi: 10°, Scale: 0.5, FlipLR: 0.5, HSV) |
| **isipiringku_v1-3** | Dihentikan Awal | 2 / 100 | 15 | 0.4299 (Ep 2) | 0.3103 | 0.4080 (Ep 2) | 0.2784 | 0.5064 / 0.4220 | 0.4987 / 0.4093 | Sedang (Mosaic: 0.3, Rotasi: 0°, Scale: 0.3, FlipLR: 0.5, HSV) |
| **isipiringku_v1-4** | Selesai (Early Stop) | 33 / 100 | 10 | 0.6350 (Ep 24) | 0.5205 | 0.6255 (Ep 22) | 0.4818 | 0.6692 / 0.5822 | 0.6693 / 0.5747 | Sangat Rendah (Mosaic: 0.1, Rotasi: 0°, Scale: 0.1, FlipLR: 0.0) |
| **isipiringku_v2** | **Selesai (Early Stop)** | **80 / 100** | **20** | **0.6806 (Ep 62)** | **0.5632** | **0.6728 (Ep 44)** | **0.5288** | **0.7147 / 0.6185** | **0.7122 / 0.6039** | **Tinggi (Mosaic: 0.5, Rotasi: 10°, Scale: 0.5, FlipLR: 0.5, HSV)** |

---

## 🔍 Analisis Mendalam Tiap Model

### 1. `isipiringku_v1`, `v1-2`, dan `v1-3` (Tidak Direkomendasikan)
* **Status**: Tidak layak digunakan.
* **Analisis**: 
  * `v1` tidak menyelesaikan epoch pertama dan folder `weights` kosong.
  * `v1-2` dan `v1-3` hanya berjalan selama **2 epoch**. Model berada dalam kondisi *underfitted* ekstrim dengan mAP yang sangat rendah (~40-42%).
  * **Kesimpulan**: Singkirkan ketiga folder ini dari kandidat deployment.

### 2. `isipiringku_v1-4` (Kandidat Menengah)
* **Status**: Cukup baik, tetapi kurang optimal untuk skenario dunia nyata.
* **Kelebihan**: Berhasil dilatih hingga 33 epoch dan mencapai mAP50 Box **63.5%** dan Mask **62.5%**.
* **Kekurangan**: 
  * Dihentikan terlalu cepat karena nilai `patience: 10` yang terlalu sensitif.
  * Dilatih dengan **hampir tanpa augmentasi data** (tidak ada rotasi, tidak ada horizontal flip, jitter warna nol, dan mosaic hanya 0.1).
  * **Risiko Skenario Nyata**: Karena kurangnya augmentasi warna, rotasi, dan skala, model ini akan sangat rentan terhadap variasi pencahayaan ruangan, sudut pengambilan foto makanan (angulasi kamera HP), serta jarak piring ke kamera (*overfitting* pada orientasi dan warna foto dataset asli).

### 3. `isipiringku_v2` (🏆 Rekomendasi Utama / Model Terbaik)
* **Status**: **Sangat Direkomendasikan untuk Real Use Case.**
* **Analisis**:
  * **Performa Tertinggi**: Mencapai akurasi deteksi lokalisasi (Box mAP50: **68.06%**) dan akurasi segmentasi bentuk makanan (Mask mAP50: **67.28%**). Ini adalah peningkatan akurasi absolut sebesar **~4.5% - 4.7%** dibandingkan `v1-4`.
  * **Generalisasi Kuat**: Dilatih dengan augmentasi data yang lengkap (Rotasi 10 derajat, Skala 0.5, Horizontal Flip 0.5, dan Color Jitter HSV). Hal ini membuat model jauh lebih tangguh terhadap variasi warna makanan, bayangan, rotasi piring, dan ukuran piring di dunia nyata.
  * **Pelatihan Optimal**: Pengaturan `patience: 20` memungkinkan model mengeksplorasi ruang pencarian bobot lebih lama (mencapai 80 epoch) sehingga konvergensi tercapai secara optimal sebelum dihentikan otomatis.

---

## 📱 Relevansi untuk Real Use Case (Aplikasi Dunia Nyata)

Model terbaik di dalam folder **`isipiringku_v2`** sangat siap untuk diintegrasikan ke aplikasi produksi (misalnya aplikasi mobile Android/iOS pemantau gizi).

### 📁 Format Deployment yang Tersedia di `isipiringku_v2/weights/`:

1. **`best.pt`** (PyTorch - 20.5 MB)
   * **Penggunaan**: Untuk backend server (Python/FastAPI/Flask) atau pengujian lokal. Sangat fleksibel tetapi membutuhkan runtime PyTorch.
2. **`best.onnx`** (ONNX - 40.6 MB)
   * **Penggunaan**: Standar industri untuk integrasi lintas platform. Dapat dijalankan dengan ONNX Runtime di NodeJS, C++, C#, atau web browser (ONNX Runtime Web) dengan performa CPU/GPU yang dioptimalkan.
3. **`best_saved_model/best_float32.tflite`** (TFLite FP32 - 40.6 MB) & **`best_float16.tflite`** (TFLite FP16 - 20.4 MB)
   * **Penggunaan**: **Sangat krusial untuk aplikasi mobile (Android/iOS)**.
   * Versi **FP16** memotong ukuran model hingga setengahnya (hanya 20.4 MB) tanpa mengorbankan akurasi secara signifikan. Ini membuat aplikasi mobile lebih ringan diunduh dan mempercepat inferensi langsung di perangkat (*on-device AI*) memanfaatkan NPU/GPU HP pengguna.

### 🍽️ Kelas yang Didukung
Model ini mendeteksi 5 kategori utama yang sesuai dengan pedoman piring makan sehat Indonesia:
1. **`Makanan Pokok`**: Nasi, kentang, roti, mie, jagung.
2. **`Lauk Pauk`**: Daging, ayam, ikan, telur, tempe, tahu (sumber protein).
3. **`Sayuran`**: Berbagai jenis sayur matang/mentah.
4. **`Buah-buahan`**: Pisang, pepaya, semangka, apel, dll.
5. **`Lainnya`**: Komponen non-kategori utama (saus, kerupuk, piring kosong, dll.).

---

## 📌 Kesimpulan & Rekomendasi
Gunakan model **`isipiringku_v2\weights\best.pt`** (atau versi `.tflite` / `.onnx` di folder tersebut tergantung arsitektur aplikasi Anda). Model ini memiliki akurasi segmentasi tertinggi, ketahanan visual terbaik berkat augmentasi data, dan telah berhasil diekspor ke berbagai format siap pakai untuk aplikasi mobile maupun web API.
