# NutriLabel v3.5 - Evaluation Data & Metrics Report

Laporan ini berisi rangkuman seluruh file evaluasi, metrik training YOLO, dan hasil pengujian OCR dari proyek NutriLabel v3.5, sebagaimana diminta untuk dianalisis lebih lanjut (misalnya oleh AI lain seperti Claude).

---

## TASK 1 — OCR Evaluation Files

Berdasarkan pencarian rekursif di folder `Dataset/`, saat ini hanya ditemukan **satu** pasang data evaluasi dengan Ground Truth (GT) yang berada di folder `test/`. Dataset nyata lainnya di folder seperti `Makanan Kaleng`, `Frozen Food`, dll belum memiliki file `_gt.json`.

### 1. `Dataset/test/termudah_ocrtest_gt.json`
Berisi label manual (ground truth) dari takaran saji dan nutrisi.
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
**Pasangan Gambar:** `Dataset/test/termudah_ocrtest.jpg`

### 2. `Dataset/test/evaluasi_hasil.json`
Berisi hasil dari pengujian OCR otomatis menggunakan `evaluasi_batch.py` (unit testing pada data test).
```json
{
  "ringkasan": {
    "total": 1,
    "skipped_no_gt": 0,
    "avg_field_accuracy": 1.0,
    "avg_nutrition_precision": 0.8,
    "avg_nutrition_recall": 1.0,
    "avg_detection_rate": 1.0,
    "serving_size_accuracy": 1.0,
    "target_field_accuracy_85": true
  },
  "detail": [
    {
      "file": "termudah_ocrtest.jpg",
      "status": "success",
      "quality": {
        "quality_score": 0.802,
        "usable": true,
        "blur_laplacian": 134.97,
        "brightness": 179.57,
        "contrast": 54.16,
        "glare_ratio": 0.0,
        "dark_ratio": 0.0002,
        "edge_ratio": 0.0466,
        "warnings": [],
        "image_size": { "width": 2499, "height": 1597 }
      },
      "roi": {
        "detected": true,
        "method": "horizontal_lines"
      },
      "semantic_filter": {
        "enabled": true,
        "reason": "nutrition_semantic_filter",
        "anchors": 17,
        "kept": 51,
        "removed": 3
      },
      "nutrition_metrics": {
        "field_accuracy": 1.0,
        "exact_or_close_fields": 8,
        "total_fields": 8,
        "nutrition_precision": 0.8,
        "nutrition_recall": 1.0,
        "nutrition_detection_rate": 1.0,
        "serving_size_correct": true,
        "missing_fields": [],
        "wrong_fields": [],
        "per_field": {
          "kalori": { "gt": 100, "pred": 100.0, "score": 1.0, "status": "exact" },
          "lemak_total": { "gt": 4, "pred": 4.0, "score": 1.0, "status": "exact" },
          "lemak_jenuh": { "gt": 2, "pred": 2.0, "score": 1.0, "status": "exact" },
          "protein": { "gt": 1, "pred": 1.0, "score": 1.0, "status": "exact" },
          "karbohidrat": { "gt": 15, "pred": 15.0, "score": 1.0, "status": "exact" },
          "serat": { "gt": 1, "pred": 1.0, "score": 1.0, "status": "exact" },
          "gula": { "gt": 8, "pred": 8.0, "score": 1.0, "status": "exact" },
          "natrium": { "gt": 110, "pred": 110.0, "score": 1.0, "status": "exact" }
        }
      }
    }
  ]
}
```

---

## TASK 2 — YOLO Training Metrics

Model YOLOv11 dilatih selama 80 epoch dengan konfigurasi berikut (`isipiringku_v2/args.yaml`):

### `args.yaml`
```yaml
task: segment
mode: train
model: yolo11s-seg.pt
epochs: 100
batch: 16
imgsz: 640
optimizer: AdamW
lr0: 0.001
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
```

### Grafik dan Kurva Pelatihan
*(Catatan: Anda dapat membuka file ini secara lokal untuk melihat kurva secara visual)*
- **Confusion Matrix Normalized:** `d:\Labolatorium Anti Gravity\NutrilLabel v3.5\isipiringku_v2\confusion_matrix_normalized.png`
- **Box PR Curve:** `d:\Labolatorium Anti Gravity\NutrilLabel v3.5\isipiringku_v2\BoxPR_curve.png`
- **Mask PR Curve:** `d:\Labolatorium Anti Gravity\NutrilLabel v3.5\isipiringku_v2\MaskPR_curve.png`
- **Box P/R Curves:** `d:\Labolatorium Anti Gravity\NutrilLabel v3.5\isipiringku_v2\BoxP_curve.png`, `BoxR_curve.png`
- **Mask P/R Curves:** `d:\Labolatorium Anti Gravity\NutrilLabel v3.5\isipiringku_v2\MaskP_curve.png`, `MaskR_curve.png`

### `results.csv` (Epoch terakhir)
Berikut adalah metrik pada epoch terakhir (Epoch 80). Performa mAP Box mencapai ~67.8% dan mAP Mask ~67.1%.
```csv
epoch,time,train/box_loss,train/seg_loss,train/cls_loss,train/dfl_loss,train/sem_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),metrics/precision(M),metrics/recall(M),metrics/mAP50(M),metrics/mAP50-95(M),val/box_loss,val/seg_loss,val/cls_loss,val/dfl_loss,val/sem_loss,lr/pg0,lr/pg1,lr/pg2
...
78,31093,0.55662,1.14656,0.64941,1.00686,0,0.72456,0.6111,0.67901,0.56268,0.72295,0.60991,0.67146,0.52861,0.66649,1.51736,1.0842,1.08232,0,0.0002377,0.0002377,0.0002377
79,31489.7,0.55697,1.13703,0.64631,1.00648,0,0.72948,0.6079,0.67869,0.56263,0.72838,0.60715,0.67135,0.52873,0.66622,1.51783,1.08481,1.08212,0,0.0002278,0.0002278,0.0002278
80,31886.4,0.55149,1.12998,0.63756,1.00623,0,0.72877,0.60792,0.67816,0.56234,0.72778,0.60724,0.67136,0.52876,0.66654,1.51876,1.08612,1.08289,0,0.0002179,0.0002179,0.0002179
```

---

## TASK 3 — Per-Class Metrics

YOLO `results.csv` secara default hanya menyimpan metrik gabungan (seluruh kelas). Namun metrik per-kelas (seperti *Makanan Pokok, Lauk Pauk, Sayuran, Buah-buahan*) dapat dilihat langsung pada visualisasi:
1. `confusion_matrix_normalized.png` (Diagonal merepresentasikan Recall/Akurasi per-kelas).
2. `BoxPR_curve.png` / `MaskPR_curve.png` (Terdapat kurva berbeda untuk setiap kelas beserta skor mAP50 individualnya di bagian legenda).

*Catatan: Tidak ada file JSON terpisah (`val_results.json`) yang dihasilkan untuk breakdown per kelas karena tidak diatur secara default saat training Ultralytics.*

---

## TASK 4 — Sample Results (Visual)

Berikut adalah beberapa gambar hasil prediksi (overlay) untuk keperluan analisis:

### 1. Contoh Segmentasi Makanan (YOLO)
Gambar-gambar berikut adalah hasil validasi YOLO (`val_batch`) yang menimpa mask prediksi di atas gambar makanan.
- **Prediksi Batch 0:** `d:\Labolatorium Anti Gravity\NutrilLabel v3.5\isipiringku_v2\val_batch0_pred.jpg`
- **Prediksi Batch 1:** `d:\Labolatorium Anti Gravity\NutrilLabel v3.5\isipiringku_v2\val_batch1_pred.jpg`

![YOLO Prediction Sample 1](file:///d:/Labolatorium%20Anti%20Gravity/NutrilLabel%20v3.5/isipiringku_v2/val_batch0_pred.jpg)

### 2. Contoh OCR Bounding Box Overlay
Gambar berikut adalah hasil debug OCR yang menampilkan bounding box di atas teks label nutrisi:
- **OCR Overlay:** `d:\Labolatorium Anti Gravity\NutrilLabel v3.5\Dataset\test\sandwich_debug\02_ocr_overlay.jpg`

![OCR Overlay Sample](file:///d:/Labolatorium%20Anti%20Gravity/NutrilLabel%20v3.5/Dataset/test/sandwich_debug/02_ocr_overlay.jpg)
