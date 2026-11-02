# Models

Artefak training YOLO11-Seg untuk dataset **Isi Piringku**.

```text
isipiringku_v2/
├── args.yaml                 # konfigurasi training
├── results.csv / *.png       # metrik & kurva
├── confusion_matrix*.png
├── val_batch*_*.jpg          # contoh prediksi validasi
└── weights/
    ├── last.pt               # checkpoint PyTorch terakhir
    ├── best.onnx             # ekspor ONNX
    └── best_saved_model/     # ekspor TensorFlow SavedModel
```

Bobot TFLite untuk fallback browser berada di
`frontend/assets/models/best_float32.tflite`.

Backend memilih bobot dengan urutan `best.pt` → `last.pt` → `best.onnx`.
Letakkan `best.pt` di `weights/` (atau set `HEALTHIER_YOLO_MODEL`) untuk memakai
checkpoint terbaik.
