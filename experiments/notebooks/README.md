# Notebooks

- `revision_kaggle.ipynb` — **notebook utama revisi** (Kaggle T4). Self-contained:
  membangun dataset, split train/val + 3 external test, screening, training utama,
  baseline, evaluasi instance/pixel, dan evaluasi 10–20 foto Indonesia.

  **Ini satu-satunya file yang perlu kamu buka/run di Kaggle.** Script di
  `experiments/scripts/` (OCR, tabel) dijalankan terpisah di lokal setelah
  hasil Kaggle diunduh.
- `training_model.ipynb` — notebook Kaggle asli (bukti audit). Menyimpan output
  (termasuk gambar), sehingga ukurannya ±10 MB.

Gunakan sebagai **bukti audit**, bukan sebagai sumber angka final:
- output menunjukkan dataset aktual yang dipakai (4.840 train / 960 val,
  lalu oversampling menjadi 12.082),
- output menunjukkan export FP16 gagal,
- output menunjukkan hardware training (Tesla T4),
- output berisi baseline 5-epoch untuk 4 model YOLO Medium.

Pipeline revisi yang reproducible ada di `experiments/scripts/`. Jangan
menjalankan ulang notebook ini untuk angka final; gunakan script tersebut agar
split, seed, dan metrik tercatat.
