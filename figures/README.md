# Figure slots — Healthier / JRI 2026

Folder ini berisi **slot gambar** untuk naskah `experiments/paper/manuscript_jri.md`
(`manuscript_jri.docx`). Setiap file adalah placeholder.

**Cara pakai:** timpa file placeholder dengan gambar final Anda, **nama file tetap sama**.
Boleh tetap `.png`, atau ganti ke `.jpg` selama namanya sama (mis. `figure 1.jpg`).

| File | Caption (sesuai naskah) | Saran ukuran |
| :--- | :--- | :--- |
| `figure 1.png` | Healthier system architecture and experimental workflow. | skema lebar, ~1600×900 px |
| `figure 2.png` | Human-in-the-Loop interfaces: segmentation editor and nutrient value verification form. | 2 screenshot berdampingan, ~1600×900 px |
| `figure 3.png` | Training and validation loss curves for the YOLO11s-seg model. | grafik garis, ~1400×1000 px |
| `figure 4.png` | Normalised confusion matrix on the internal validation split. | heatmap 5×5, ~1200×1200 px |
| `figure 5.png` | Qualitative comparison of ground truth (left) and prediction (right) on the internal validation split. | grid citra berpasangan, ~1600×1000 px |
| `figure 6.png` | Normalised confusion matrices on the three external test sources. | 3 panel berdampingan, ~1800×800 px |
| `figure 7.png` | Mask precision-recall and F1 curves for each external source. | multi-panel, ~1600×1200 px |
| `figure 8.png` | Per-class external mask AP50 by source and class. | bar chart, ~1600×900 px |
| `figure 9.png` | Nutrition-label extraction pipeline: preprocessing, recognition, and layout reconstruction. | diagram pipeline lebar, ~1800×800 px |
| `figure 10.png` | Qualitative example of label extraction. | input + ROI + tabel, ~1600×900 px |

## Catatan
- Resolusi minimal **300 DPI** (atau sisi terpanjang ≥ 1600 px) agar tidak pecah saat dicetak.
- Format aman untuk Word: **PNG**. Untuk foto bisa JPG kualitas tinggi.
- Di naskah, posisi tiap gambar ditandai `[[INSERT FIGURE N HERE ...]]`; caption resmi sudah
  tertulis di bawah penanda tersebut dan **tidak perlu** Anda tulis ulang di dalam gambar.
- Setelah gambar final dimasukkan, jangan lupa hapus kalimat placeholder di naskah bila ada,
  dan commit foldernya ke git agar tersimpan.

Placeholder ini digenerate otomatis; file gambar asli Anda akan menimpanya.
