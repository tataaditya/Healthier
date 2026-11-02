# Model fallback browser

`best_float32.tflite` adalah ekspor TFLite dari model YOLO11-Seg Isi Piringku.
File ini dipakai `frontend/js/seefood-segmenter.js` melalui
`@tensorflow/tfjs-tflite` (WASM) ketika API backend tidak tersedia.

- Ukuran: ±38,7 MB
- Input: `1 × 640 × 640 × 3` (float32, 0–1)
- Output: tensor YOLO segmentation (bounding box + mask coefficients)
- Path dapat diubah lewat `window.HEALTHIER_CONFIG.TFLITE_MODEL_URL`
  (lihat `frontend/js/config.js`).

Jangan memindahkan file ini tanpa memperbarui `TFLITE_MODEL_URL`.
