# Revisi Mayor Ronde 2 — Matriks Tanggapan (8 poin)

> Sumber: rekap penulis 5 Nov 2026 (No. 310), Reviewer A (4 poin) + Reviewer B
> (13 komentar). Dokumen yang dicek:
> `Application of Deep Learning ... - Tata Aditya Pamungkas.docx` (101 paragraf,
> 16 tabel). Matrix ini memetakan **setiap poin → bukti yang sudah ada → yang
> masih kurang → tindakan**. Tidak ada angka baru yang diklaim di sini.

## Ringkasan status

| # | Poin reviewer | Status di repo | Bukti | Masih kurang | Perkiraan usaha |
| :-- | :--- | :--- | :--- | :--- | :--- |
| 1 | Dataset OCR ≥100 citra brand-stratified + prosedur GT (A3/B6) | **Setengah siap**: split 131 citra test terkunci, viewer verifikasi siap; 67/131 draf terisi | `ocr_eval/split_lock.json`, `annotation/annotation_guideline.md`, `annotation/viewer.html` | GT 131 citra belum diverifikasi manusia; kolom **merek** belum ada; tabel stratifikasi belum dibuat | User verifikasi + 0,5 hari analisis |
| 2 | Baseline segmentasi full-budget sebanding (A4) | **Belum ada**; yang ada hanya screening 5 epoch | `final_numbers.md` §82–87 | Latih 1 arsitektur alternatif dengan 100 epoch, split & seed sama | 6–10 jam GPU (RTX 3050) |
| 3 | Validasi proporsi area poligon vs GT massa/porsi; pisahkan area visual dari estimasi gizi; ketidakpastian tanpa depth (B3/B11) | **Belum ada**; naskah hanya menyebut "2D area ratios" sebagai limitasi | `_docx_5nov.txt` ¶99 | Data massa/porsi GT (timbangan) **atau** analisis sensitivitas/ketidakpastian eksplisit | Protokol + capture user + 1 hari analisis |
| 4 | Bukti kuantitatif HITL: frekuensi koreksi, akurasi sebelum–sesudah (B5) | **Bisa dihitung otomatis** setelah GT verified: prediksi pipeline vs GT | `ocr_eval/predictions/*.jsonl` + `annotation/drafts.json` | GT verified; skrip metrik HITL | 1–2 jam setelah GT |
| 5 | Deployment edge: latency end-to-end, memori, ukuran model (B10) | **Belum diukur**; klaim 168 ms sudah dihapus | `models/isipiringku_v2/weights/` | Skrip benchmark + jalankan di device target | 1–2 jam (run) |
| 6 | Audit anotasi/remapping kelas + class imbalance (A3/B1) | **Selesai untuk remapping & imbalance** | `segmentation/dataset_audit.md` (+`.json`) | IAA/κ tidak ada (hanya 1 anotator) — laporkan sebagai limitasi; audit GT OCR setelah verifikasi | 0,5 hari |
| 7 | Referensi PERKENI 2024, James et al., YOLO11 (Khanam & Hussain); reference implementation aturan klinis (B4/B7) | **Sebagian selesai**: ketiga sitasi ada di `manuscript_jri.md` REFERENCES | `manuscript_jri.md` §REFERENCES; `frontend/js/clinical-rules.js`; `results/clinical_rules/verification.md` | Pastikan masuk juga di docx; tambahkan rujukan eksplisit ke implementasi aturan (supplementary/repo) | 0,5 jam |
| 8 | Inkonsistensi minor: evaluator 4 vs 2, penomoran tabel/gambar, PP-OCRv5→v4 belum dijelaskan; surat tanggapan point-by-point (B?) | **Belum diperbaiki**; ¶91 menulis "four evaluators" lalu "two-evaluator panel" | `_docx_5nov.txt` ¶89–94 | Perbaiki teks + nomor + tambahkan paragraf penjelasan v5→v4; lampirkan surat | 2–3 jam |

---

## Detail per poin

### 1 — Dataset OCR ≥100 citra + prosedur GT

**Yang sudah ada**

- Split OCR terkunci: 438 citra → **test 131 / dev 307**, seed 42, sha256
  `a0c7a494…`, test∩dev = ∅ (`ocr_eval/split_lock.json`,
  `split_manifest.csv`).
- Prosedur anotasi tertulis (`annotation/annotation_guideline.md`): kolom
  **Per Sajian**, bukan %AKG / per 100 g; baris tak tercetak → `absent`;
  verifikasi manusia wajib.
- Viewer verifikasi + ekspor `ground_truth_verified.json`.
- Evaluasi 5 konfigurasi (A–E) + ablation kumulatif + LOO pada 131 citra
  (prediksi A/B valid; C–E menunggu rerun).

**Yang masih kurang**

1. **GT 131 citra belum diverifikasi manusia** — ini prasyarat angka final.
2. **Stratifikasi merek** belum ada. Folder dataset memberi stratifikasi
   kategori (Frozen Food, Makanan Kaleng, Makanan Ringan, Minuman Sachet,
   Saos & Perbumbuan), tetapi **merek tidak tersimpan sebagai field**.
3. Tabel "n per kategori × jumlah merek" belum dibuat.

**Tindakan**

- User menyelesaikan verifikasi di `viewer.html` → ekspor.
- Tambah input **Merek (opsional)** di viewer agar stratifikasi merek
  benar-benar terverifikasi, bukan ditebak dari nama file.
- Skrip stratifikasi: hitung n per kategori, n merek unik, dan sebaran
  field terisi per kategori.

**Draf kalimat tanggapan (English, isi angka setelah GT)**

> We expanded the OCR evaluation from six images to a locked test split of
> **131 images** drawn from five product categories, with a documented
> annotation protocol (per-serving column, explicit `absent` coding) and
> mandatory human verification. Table X reports category and brand coverage;
> ground truth was verified by [author], and a second spot-check of [n] images
> was performed. The six-image pilot is retained only as a historical
> screening result and is no longer used for any headline claim.

---

### 2 — Baseline segmentasi full-budget

**Yang sudah ada:** screening 5 epoch (YOLOv8m-seg 54.67, YOLO11m-seg 52.93,
YOLO26m-seg 53.45 mask mAP50) yang **tidak sebanding**.

**Tindakan (pilih satu):**

- **Opsi A (rekomendasi):** latih **YOLO11n-seg** dengan protokol identik
  (100 epoch, patience 20, seed 42, split sama, 640 px) di RTX 3050 6 GB.
  Batch mungkin perlu diturunkan ke 8; **catat** perbedaan batch dan
  konsekuensinya.
- **Opsi B:** latih **YOLOv8s-seg** (arsitektur pembanding yang lebih
  setara ukuran dengan YOLO11s-seg). Waktu lebih lama.

**Yang wajib dilaporkan:** jumlah epoch berhenti, checkpoint terbaik, mAP
internal + eksternal, dan pernyataan bahwa hardware berbeda (T4 vs RTX 3050)
tetapi *budget* (epoch/split/seed) sama.

**Draf tanggapan**

> We trained [architecture] with the same 100-epoch budget, early-stopping
> patience (20), split, and seed as the main model. It reached mask mAP50 of
> [x]% on the internal validation split (Table X). The earlier five-epoch
> screening is now explicitly labelled as non-comparable and is not used for
> any superiority claim.

---

### 3 — Proporsi area vs massa/porsi + ketidakpastian tanpa depth

**Kondisi jujur:** tidak ada dataset massa/porsi berlabel di repo. Klaim
"area proporsi = porsi" **tidak boleh** dibuat.

**Dua lapis yang harus ditulis:**

1. **Pemisahan konsep (wajib, bisa sekarang):** area poligon = *visual area
   proportion*, bukan estimasi energi/massa. Gizi hanya dihitung setelah
   pengguna memverifikasi/mengoreksi porsi. Ini harus menjadi kalimat tegas
   di Methods + Discussion.
2. **Validasi empiris (butuh data baru):**
   - **Opsi A (cepat, realistis):** mini-study terkontrol — 20–30 piring,
     3 komponen makanan, tiap komponen ditimbang (gram) sebelum difoto;
     bandingkan proporsi area mask vs proporsi massa. Laporkan MAE, bias,
     korelasi, dan Bland–Altman. Butuh timbangan dapur + user memotret.
   - **Opsi B (tanpa capture):** analisis sensitivitas/ketidakpastian —
     perturbasi mask (dilasi/erosi), variasi asumsi tinggi & densitas,
     laporkan rentang estimasi. Ini **bukan** validasi, hanya kuantifikasi
     ketidakpastian; harus diberi label demikian.

**Draf tanggapan**

> We now distinguish explicitly between *visual area proportion* and
> *quantitative nutrient estimation*. Polygon area is never converted to grams
> without a user-confirmed portion. Because the system has no depth sensor, we
> quantify the resulting uncertainty through [controlled mass validation /
> sensitivity analysis] and report it as a range rather than a point estimate
> (Table X). This limitation is stated in the Abstract, Methods, and Discussion.

---

### 4 — Bukti kuantitatif HITL

**Bisa dihitung tanpa eksperimen baru**, setelah GT verified:

- **Correction frequency:** % field yang harus diubah user dari keluaran
  pipeline; per field dan per kategori.
- **Accuracy before vs after:** exact-match pipeline vs GT (sebelum) dan
  setelah koreksi (= 100% by construction); yang lebih informatif:
  akurasi **setelah koreksi hanya pada field yang ditandai low-confidence**,
  serta **jumlah edit per citra** (beban koreksi).
- **Decision impact:** berapa keputusan SAFE/WARNING/DANGER berubah sebelum
  vs sesudah koreksi (memakai `clinical_rules/e2e_propagation`).

**Metrik yang dilaporkan:** edit rate (%), field-level accuracy before/after,
decision-change rate, median edit per image.

---

### 5 — Deployment edge

**Ukur apa adanya, di device yang jelas:**

| Metrik | Cara |
| :--- | :--- |
| Ukuran model | `best.pt`, `best.onnx` (MB) |
| Latency segmentasi | 30 gambar, warm-up 3, median + P95, CPU & GPU |
| Latency OCR | pipeline OCR penuh, median + P95 |
| End-to-end | segmentasi + OCR + aturan klinis |
| Memori | peak RSS (CPU) dan peak VRAM (GPU) |

**Penting:** sebutkan device target (laptop RTX 3050 6 GB dan/atau CPU-only),
versi library, resolusi input. Jangan mengklaim "edge" untuk perangkat yang
tidak diukur.

---

### 6 — Audit anotasi/remapping + imbalance

**Sudah dibuat:** `experiments/results/segmentation/dataset_audit.md`
(instance & image imbalance, rasio max/min, kelas absen per split, peta
FoodSeg → 5 grup, risiko catch-all `Others`).

**Angka kunci:** imbalance instance train **3,04×**, image **4,12×**;
UEC tanpa `Fruits`; FruitSeg hanya `Fruits` → macro-average eksternal tidak
sebanding.

**Masih kurang:** IAA/κ — tidak ada anotator kedua. Laporkan sebagai
keterbatasan; jangan mengarang κ.

---

### 7 — Referensi

Ketiga sitasi **sudah ada** di `manuscript_jri.md` REFERENCES:
PERKENI (2024), James et al. (2014), Khanam & Hussain (2024) — dan dikutip
di Methods. Yang perlu:

1. Pastikan docx final memuat ketiganya (docx yang diunggah belum bisa
   diverifikasi bagian REFERENCES-nya dari ekstraksi teks).
2. Tambahkan rujukan eksplisit ke **implementasi aturan klinis**
   (`frontend/js/clinical-rules.js`) sebagai supplementary material /
   repositori, bukan sebagai sitasi akademik.
3. Tambahkan kalimat bahwa batas klinis adalah *functional verification*,
   bukan validasi klinis.

---

### 8 — Inkonsistensi minor + surat tanggapan

**Daftar perbaikan konkret:**

| Lokasi | Masalah | Perbaikan |
| :--- | :--- | :--- |
| ¶91 | "four evaluators" lalu "two-evaluator panel" | Tulis satu skema: 4 evaluator untuk heuristic walkthrough; 2 evaluator untuk severity rating — atau seragamkan jadi 2 dan jelaskan |
| ¶92 | "Table 13 Usubality evaluator Assesment" (typo) | Perbaiki judul tabel + nomor |
| ¶85–89 | Rujukan "Figures 10 and 11", "Table 10/11/12" tidak konsisten dengan penomoran naskah | Samakan dengan daftar tabel/gambar final |
| ¶98 | "PP-OCRv4" tanpa penjelasan perubahan dari v5 | Tambah satu kalimat: versi pustaka `paddleocr` vs nama model; v4 dipilih karena didukung & reproducible |
| ¶99 | "2D area ratios" disebut limitasi tanpa kuantifikasi | Kaitkan ke poin 3 |

**Surat tanggapan:** file ini + `response_to_reviewers.md` bisa dijadikan
lampiran point-by-point. Perlu versi final berbahasa Inggris setelah angka
GT masuk.

---

## Urutan eksekusi yang disarankan

1. **GT 131 citra** (user) → metrik final OCR (poin 1 & 4).
2. **Rerun 18 konfigurasi Paddle** (±1–1,5 jam) → angka C/D/E + ablation.
3. **Benchmark edge** (poin 5, 1–2 jam).
4. **Audit dokumen + surat tanggapan** (poin 6–8, bisa paralel).
5. **Baseline full-budget** (poin 2, 6–10 jam) — keputusan terpisah.
6. **Mini-study massa/porsi** (poin 3) — butuh capture user.

> Catatan: poin 2 dan 3 tidak bisa diselesaikan murni dari data yang ada;
> keduanya butuh run pelatihan / capture baru. Jangan menulis klaim yang
> belum diukur.
