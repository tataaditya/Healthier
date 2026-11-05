# Figures — Healthier / JRI 2026

Gambar untuk naskah `experiments/paper/manuscript_jri.md` (`manuscript_jri.docx`).
**Semua file sudah terisi** dan digenerate otomatis oleh
`experiments/scripts/make_paper_figures.py` dari hasil eksperimen yang tersimpan
(di `experiments/results/` dan `experiments/runs/`).

| File | Caption (sesuai naskah) | Asal / status |
| :--- | :--- | :--- |
| `figure 1.png` | Healthier system architecture and experimental workflow. | Diagram teknis (digambar): dua menara bergaya Transformer — backbone/neck/head YOLO11s-seg dengan grup berulang `× 3`, pipeline PP-OCRv4, dan lapisan keputusan HITL → Clinical Rules Engine → SAFE/WARNING/DANGER, plus anotasi hasil eksperimen |
| `figure 2.png` | Human-in-the-Loop interfaces: segmentation editor and nutrient value verification form. | **Skema (digambar)** — ganti dengan screenshot asli bila ada |
| `figure 3.png` | Training and validation loss curves for the YOLO11s-seg model. | Dari `runs/seg_main/results.csv` |
| `figure 4.png` | Normalised confusion matrix on the internal validation split. | Salinan `runs/seg_main/confusion_matrix_normalized.png` |
| `figure 5.png` | Qualitative comparison of ground truth (left) and prediction (right) on the internal validation split. | Gabungan `runs/seg_main/val_batch0_{labels,pred}.jpg` |
| `figure 6.1.png` | Normalised confusion matrix — FoodSeg103 (validation). | `results/segmentation/foodseg_val_instance/confusion_matrix_normalized.png` |
| `figure 6.2.png` | Normalised confusion matrix — UECFoodPIXCOMPLETE (test). | `results/segmentation/uec_test_instance/confusion_matrix_normalized.png` |
| `figure 6.3.png` | Normalised confusion matrix — FruitSeg30. | `results/segmentation/fruitseg_instance/confusion_matrix_normalized.png` |
| `figure 7.png` | Mask precision-recall and F1 curves for each external source. | Gabungan `*_instance/MaskPR_curve.png` + `MaskF1_curve.png` |
| `figure 8.png` | Per-class external mask AP50 by source and class. | Dari `results/segmentation/per_class_all.csv` |
| `figure 9.png` | Nutrition-label extraction pipeline: preprocessing, recognition, and layout reconstruction. | Diagram (digambar) |
| `figure 10.png` | Qualitative example of label extraction. | Foto `20241108_132045` + tabel hasil grid parser vs ground truth |

## Regenerasi semua gambar

```powershell
python experiments/scripts/make_paper_figures.py
```

Skrip menulis ulang `figures/figure 1..10.png` (plus `figure 6.1/6.2/6.3.png`).
Gambar 1, 2, dan 9 adalah diagram yang digambar dari kode; sisanya disalin/digabung
dari hasil eksperimen.

## Menyisipkan ke docx

```powershell
python experiments/paper/build_jri_docx.py --embed-figures
```

Perintah itu mengganti penanda `[[INSERT FIGURE N HERE ...]]` di naskah dengan
`figure N.png/.jpg/.jpeg`. Tanpa flag itu, docx dibangun tanpa gambar.

**Figure 6 sengaja dipecah menjadi tiga file** (`figure 6.1.png`, `figure 6.2.png`,
`figure 6.3.png`) supaya tiap confusion matrix tampil utuh dan jelas. Karena itu
`--embed-figures` tidak menyisipkan Figure 6 otomatis (penanda tetap berupa teks);
sisipkan `figure 6.1/6.2/6.3` secara manual di bawah caption Figure 6.

## Catatan
- Resolusi 300 DPI; format PNG.
- Figure 6 = tiga file terpisah: 6.1 FoodSeg103 (validation), 6.2 UECFoodPIXCOMPLETE
  (test), 6.3 FruitSeg30. Penomoran figure lain (7-10) tidak berubah.
- Figure 1, 2, dan 9 (diagram yang **digambar**) dibuat **hitam-putih / abu-abu**
  seperti diagram blok referensi. Figure 3-8 dan 10 berasal dari hasil eksperimen
  (kurva, confusion matrix, tabel) sehingga tampil apa adanya.
- Figure 2 masih berupa **skema**, bukan screenshot asli. Jika Anda punya tangkapan
  layar editor segmentasi dan form verifikasi, timpa `figure 2.png` dengan screenshot asli.
- Caption resmi sudah ada di naskah (di bawah penanda `[[INSERT FIGURE N HERE ...]]`),
  jadi tidak perlu ditulis ulang di dalam gambar.
