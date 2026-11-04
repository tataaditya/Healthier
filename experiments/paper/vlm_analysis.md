# VLM analysis — future-work comparison (bukan bagian inti paper)

> **Status: analysis / future work.** Harness ini alat bantu perbandingan, **bukan**
> bagian dari pipeline yang dipublikasikan. Pipeline inti tetap PP-OCRv4 +
> grid parser + aturan klinis deterministik. Angka di sini **tidak** masuk
> `final_numbers.md` dan tidak boleh diklaim sebagai hasil utama.
>
> Konteks keputusan: klaim C1/C2 skripsi adalah *OCR edge-based tanpa LLM cloud +
> aturan klinis deterministik*. Menjadikan VLM sebagai jalur produksi akan
> menabrak klaim itu, jadi VLM ditempatkan sebagai bahan pertimbangan future work.

## Setup

- Data: `experiments/ocr_data/makanan_ringan/` — 6 foto label kemasan + GT manual.
- GT dan metrik **identik** dengan evaluasi edge (`experiments/scripts/eval_ocr.py`):
  field dianggap benar bila sama persis atau dalam toleransi 5%; CI = cluster
  bootstrap per gambar.
- Harness: `experiments/scripts/eval_vlm.py` (kirim gambar via API Ollama,
  prompt meminta JSON per sajian, guard anti-halusinasi "jangan mengarang").
- Gambar dikecilkan ke sisi maksimum 1600 px, suhu 0.
- Model cloud: `gemma4:cloud` (vision). Model `minimax-m3:cloud` dan
  `deepseek-v4.1-flash:cloud` menolak dengan **HTTP 402** (butuh plan berbayar),
  jadi hanya `gemma4:cloud` yang tersedia.

## Hasil

| Pipeline | Field accuracy | 95% CI | Exact | Wrong | Missing |
| :--- | ---: | :--- | ---: | ---: | ---: |
| Edge baseline (PP-OCRv4) | 14.89% | [4.65%, 22.92%] | 14.89% | 17.02% | 68.09% |
| Edge grid parser (revisi) | 44.68% | [24.49%, 66.67%] | 44.68% | 19.15% | 36.17% |
| **VLM `gemma4:cloud`** | **68.09%** | [44.90%, 91.11%] | 68.09% | 25.53% | **6.38%** |

Per gambar (`gemma4:cloud`):

| Gambar | Benar/Total |
| :--- | :--- |
| 20241108_132045 | 7/7 |
| 20241108_133546 | 7/7 |
| 20241108_131859 | 7/10 |
| 20241108_132115 | 6/7 |
| 20241108_132308 | 3/8 |
| 20241108_134337 | 2/8 |

## Profil error — inilah temuan pentingnya

Beban error **berpindah jenis**, bukan sekadar mengecil:

| | Edge grid | VLM gemma4 |
| :--- | ---: | ---: |
| Missing (tidak terbaca) | 36.17% (17/47) | **6.38% (3/47)** |
| Wrong (salah nilai) | 19.15% (9/47) | **25.53% (12/47)** |

- **Recall/coverage naik drastis.** VLM nyaris selalu menemukan field-nya
  (`missing` dari 36% → 6%). Mode kegagalan utama pipeline edge — detektor
  menggabungkan grid angka & baris tak terdeteksi — praktis hilang.
- **Presisi turun sedikit.** VLM justru lebih sering *salah angka* daripada edge.

Rincian 12 error `gemma4:cloud`, hampir semuanya **meleset tipis**:

| Gambar | Field | GT | VLM | Rel. error |
| :--- | :--- | ---: | ---: | ---: |
| 131859 | lemak_jenuh | 1.5 | 1.0 | 33% |
| 131859 | gula | 10 | 3.0 | 70% |
| 131859 | natrium | 100 | 110 | 10% |
| 132115 | natrium | 65 | 85 | 31% |
| 132308 | kalori | 110 | 119 | 8% |
| 132308 | lemak_jenuh | 3 | 0 | 100% |
| 132308 | protein | 1 | 2 | 100% |
| 132308 | gula | 9 | 10 | 11% |
| 134337 | lemak_total | 2.5 | 2.0 | 20% |
| 134337 | karbohidrat | 17.9 | 17 | 5.03% |
| 134337 | gula | 5.2 | 9 | 73% |
| 134337 | natrium | 40 | 60 | 50% |

Catatan:
- Tiga error (`natrium 110 vs 100`, `kalori 119 vs 110`, `karbohidrat 17 vs 17.9`)
  hanya terlewat **sedikit di atas** toleransi 5%. Dengan toleransi 10%, akurasi
  naik jadi **35/47 = 74.5%**.
- `131859` gula 3 vs 10 dan `134337` gula 5.2 vs 9 → VLM **salah baris** (mirip
  penyebab #3 di pipeline edge).
- `132308` dan `134337` tetap paling sulit untuk **kedua** metode. Artinya untuk
  dua gambar ini bottleneck-nya adalah **keterbacaan citra/format**, bukan
  metode ekstraksi — VLM pun tidak menyelamatkannya.
- `takaran_saji_g` yang diminta VLM sering benar (20, 15, 25, 10 g) — field
  tambahan yang berguna dan tidak dipunyai pipeline edge saat ini.

## Risiko yang harus dicatat (kalau kelak dipakai)

1. **Halusinasi digit.** Error VLM bukan "tidak terbaca" tapi "salah tapi yakin".
   Di konteks gizi/klinis ini **lebih berbahaya** dari missing, karena aturan
   deterministik di hilir tidak bisa mendeteksinya. Butuh lapisan verifikasi
   (silang-cek digit dengan recognizer, atau HITL wajib untuk field sensitif
   natrium/gula).
2. **Non-determinisme & reproducibility.** Output cloud tidak dijamin identik
   antar-run/versi; bertentangan dengan klaim C2.
3. **Biaya & ketersediaan.** Dua dari tiga model cloud menolak (HTTP 402).
   Bergantung kuota/paket.
4. **Klaim edge.** Pemakaian cloud menabrak C1 ("tanpa LLM cloud").
5. **n = 6.** Terlalu kecil untuk klaim apa pun; ini baru indikasi arah.

## Implikasi untuk future work

1. **VLM sangat menjanjikan untuk *recall*.** Arah paling masuk akal: **hybrid** —
   VLM untuk menetapkan **struktur/kolom/baris** (yang jadi kelemahan fatal edge),
   OCR deterministik untuk **membaca digit persisnya** (yang jadi kelemahan VLM).
2. **On-device VLM belum diuji serius.** Model vision lokal kecil (mis.
   `qwen3.5:2b`) di RTX 3050 6 GB sempat dicoba, tetapi **di luar scope
   perbandingan ini** dan tidak dilaporkan. Trade-off akurasi vs kecepatan vs
   determinisme masih terbuka.
3. **Lapisan verifikasi wajib.** Apa pun modelnya, field sensitif harus
   divalidasi silang sebelum masuk aturan klinis.

## Reproduksi

```powershell
python experiments/scripts/eval_vlm.py `
  --data experiments/ocr_data/makanan_ringan `
  --model gemma4:cloud `
  --out experiments/results/ocr_vlm/gemma4_cloud `
  --label "Makanan Ringan pilot (VLM gemma4:cloud)" --bootstrap 2000
```

Harness juga mendukung model lokal Ollama (`--model qwen3.5:2b`), tetapi tidak
dipakai untuk perbandingan ini.
