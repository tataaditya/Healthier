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

- Data: `experiments/ocr_data/makanan_ringan/` — 6 foto label kemasan + GT manual
  (**sudah dikoreksi lewat audit GT**, lihat bagian akhir; 46 field).
- GT dan metrik **identik** dengan evaluasi edge (`experiments/scripts/eval_ocr.py`):
  field dianggap benar bila sama persis atau dalam toleransi 5%; CI = cluster
  bootstrap per gambar.
- Harness: `experiments/scripts/eval_vlm.py` (prompt meminta JSON per sajian,
  guard anti-halusinasi "jangan mengarang", suhu 0, gambar ≤1600 px).
- Dua jalur:
  - `--provider ollama` → Ollama lokal/cloud (`gemma4:cloud`).
  - `--provider openai` → endpoint OpenAI-compatible (Helipod `https://ai.helipod.io/v1`).
- Model Helipod yang benar-benar vision (terverifikasi membaca gambar):
  `deepseek-v4.1-flash`, `netra/deepseek-v4.1-flash`, `kimi-k3`, `minimax-m3`,
  `glm-5.3-flash`, `glm-5.3-flashx`. Model tanpa vision atau error 400
  (`hy3`, `deepseek-v4-pro`, `glm-5.3`, `glm-5.2`, `glm-5.1`) tidak diuji.

## Hasil (seluruh model, GT terkoreksi)

| Pipeline | Akurasi | 95% CI | Exact | Wrong | Missing | Latensi/gambar |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| Edge baseline (PP-OCRv4) | 21.74% | [16.67%, 25.58%] | 21.74% | 13.04% | 65.22% | ±2 s (GPU) |
| Edge grid parser (revisi) | 47.83% | [29.17%, 66.67%] | 47.83% | 17.39% | 34.78% | ±2 s (GPU) |
| VLM `gemma4:cloud` | 78.26% | [64.44%, 92.86%] | 78.26% | 17.39% | 4.35% | 4.3 s |
| VLM `deepseek-v4.1-flash` | **100%** | [100%, 100%] | 100% | 0% | 0% | 8.6 s |
| VLM `netra/deepseek-v4.1-flash` | **100%** | [100%, 100%] | 100% | 0% | 0% | 4.7 s |
| VLM `kimi-k3` | **100%** | [100%, 100%] | 100% | 0% | 0% | 31.2 s |
| VLM `minimax-m3` | 97.83% | [93.02%, 100%] | 97.83% | 2.17% | 0% | 9.5 s |
| VLM `glm-5.3-flash` | 95.65% | [86.05%, 100%] | 95.65% | 4.35% | 0% | 15.0 s |
| VLM `glm-5.3-flashx` | 95.65% | [86.05%, 100%] | 95.65% | 4.35% | 0% | 7.5 s |

> **Jangan klaim 100% sebagai "VLM sempurna".** Skor itu berasal dari 46 field
> (6 gambar) dan plafonnya naik setelah koreksi GT. Ini indikasi arah, bukan
> hasil generalisasi. Tetap laporkan sebagai pilot eksploratif.

Per gambar:

| Gambar | gemma4 | deepseek | minimax | glm-flash |
| :--- | ---: | ---: | ---: | ---: |
| 20241108_132045 | 7/7 | 7/7 | 7/7 | 7/7 |
| 20241108_133546 | 7/7 | 7/7 | 7/7 | 7/7 |
| 20241108_131859 | 7/10 | 10/10 | 10/10 | 10/10 |
| 20241108_132115 | 6/7 | 7/7 | 6/7 | 5/7 |
| 20241108_132308 | 4/8 | 8/8 | 8/8 | 8/8 |
| 20241108_134337 | 5/7 | 7/7 | 7/7 | 7/7 |

## Profil error

Beban error **berpindah jenis**, bukan sekadar mengecil:

| | Edge grid | gemma4 | deepseek |
| :--- | ---: | ---: | ---: |
| Missing (tidak terbaca) | 34.78% (16/46) | 4.35% (2/46) | 0% (0/46) |
| Wrong (salah nilai) | 17.39% (8/46) | 17.39% (8/46) | 0% (0/46) |

- **Recall/coverage naik drastis.** Semua VLM nyaris selalu menemukan field-nya
  (`missing` 35% → 0–4%). Mode kegagalan khas pipeline edge — detektor
  menggabungkan grid angka & baris tak terdeteksi — praktis hilang.
- **`gemma4` menukar missing dengan wrong.** Missing-nya kecil, tetapi salah
  nilainya paling banyak — inilah "salah tapi yakin".
- **Model frontier Helipod hampir bebas error.** Sisa error mereka sangat sedikit
  dan khas per-model (bukan sistematis): `minimax` salah `natrium` (85 vs 65)
  pada `132115`; `glm-5.3-flash`/`flashx` salah `lemak_jenuh` & `protein`
  (1.1 vs 1) pada `132115`.

### `gemma4:cloud` — 10 error yang tersisa

| Gambar | Field | GT | Prediksi | Rel. error |
| :--- | :--- | ---: | ---: | ---: |
| 131859 | lemak_jenuh | 1.5 | 1.0 | 33% |
| 131859 | gula | 10 | 3 | 70% |
| 131859 | natrium | 100 | 110 | 10% |
| 132115 | natrium | 65 | 85 | 31% |
| 132308 | kalori | 110 | 119 | 8% |
| 132308 | lemak_trans | 0 | (tidak ada) | missing |
| 132308 | lemak_jenuh | 3 | 0 | 100% |
| 132308 | protein | 1 | 2 | 100% |
| 134337 | serat | 1 | (tidak ada) | missing |
| 134337 | gula | 5 | 9 | 80% |

Ini error model yang asli (bukan GT): salah baca baris/angka, dengan beberapa
hanya terlewat tipis di atas toleransi 5% (`kalori 119 vs 110`, `natrium 110 vs
100`). `131859` gula 3 vs 10 dan `134337` gula 9 vs 5 = **salah baris**.

## Audit GT (sudah diterapkan)

Pemeriksaan ulang visual menemukan **kesalahan transkripsi GT** pada 2 dari 6
gambar; GT sudah diperbaiki dan semua pipeline di-skor ulang:

- `20241108_134337`: label asli `Lemak Total 2 g`, `Karbohidrat 17 g`,
  `Natrium 60 mg`, **tanpa** baris lemak jenuh; GT lama 2.5 / 17.9 / 40 / 0.5.
- `20241108_132308`: label asli `Gula 10 g`; GT lama 9.

Detail + bukti crop: `experiments/paper/audit/gt_audit.md`; patch:
`experiments/paper/audit/gt_patch_makanan_ringan.json`; skrip re-score offline:
`experiments/scripts/audit_gt.py`.

Sebelum koreksi, mayoritas "wrong" VLM sebenarnya kesalahan GT. Setelah koreksi,
`deepseek`/`netra`/`kimi` = 100% dan `gemma4` naik 68% → 78%.

## Risiko yang harus dicatat (kalau kelak dipakai)

1. **Halusinasi digit.** Error VLM bukan "tidak terbaca" tapi "salah tapi yakin"
   (lihat `gemma4`: gula 3 vs 10, lemak_jenuh 0 vs 3). Di konteks gizi/klinis ini
   **lebih berbahaya** dari missing, karena aturan deterministik di hilir tidak
   bisa mendeteksinya. Butuh lapisan verifikasi (silang-cek digit dengan
   recognizer, atau HITL wajib untuk field sensitif natrium/gula).
2. **Non-determinisme & reproducibility.** Output cloud tidak dijamin identik
   antar-run/versi; bertentangan dengan klaim C2.
3. **Biaya & ketersediaan.** Di akun Ollama gratis hanya `gemma4` yang vision;
   model vision kuat butuh kredit. Perbandingan di atas memakai gateway Helipod.
4. **Klaim edge.** Pemakaian cloud menabrak C1 ("tanpa LLM cloud").
5. **n = 6 (46 field).** Terlalu kecil untuk klaim apa pun; ini baru indikasi arah.

## Implikasi untuk future work

1. **VLM sangat menjanjikan untuk *recall*.** Arah paling masuk akal: **hybrid** —
   VLM untuk menetapkan **struktur/kolom/baris** (kelemahan fatal edge),
   OCR deterministik untuk **membaca digit persisnya** (kelemahan VLM), lalu
   lapisan verifikasi silang untuk field sensitif (natrium/gula).
2. **On-device VLM belum diuji serius.** Model vision lokal kecil di RTX 3050 6 GB
   sempat dicoba, tetapi **di luar scope** dan tidak dilaporkan. Trade-off
   akurasi vs kecepatan vs determinisme masih terbuka.
3. **Perbesar sampel dulu.** Naikkan n (≥100 gambar) sebelum klaim apa pun.
   Tanpa itu, perbandingan di sini hanya indikatif.

## Reproduksi

```powershell
# Jalur Ollama (gemma4 cloud)
python experiments/scripts/eval_vlm.py `
  --provider ollama --model gemma4:cloud `
  --data experiments/ocr_data/makanan_ringan `
  --out experiments/results/ocr_vlm/gemma4_cloud `
  --label "Makanan Ringan pilot (VLM gemma4:cloud)" --bootstrap 2000

# Jalur OpenAI-compatible (Helipod). API key dibaca dari .env (API_KEY=hel_ai_...).
python experiments/scripts/eval_vlm.py `
  --provider openai --base-url https://ai.helipod.io/v1 `
  --model deepseek-v4.1-flash `
  --data experiments/ocr_data/makanan_ringan `
  --out experiments/results/ocr_vlm/helipod_deepseek_v4_1_flash `
  --label "VLM cloud deepseek-v4.1-flash" --bootstrap 2000

# Skor ulang offline (tanpa panggil model lagi) setelah GT berubah
python experiments/scripts/eval_vlm.py --rescore `
  --provider openai --model deepseek-v4.1-flash `
  --data experiments/ocr_data/makanan_ringan `
  --out experiments/results/ocr_vlm/helipod_deepseek_v4_1_flash `
  --label "VLM cloud deepseek-v4.1-flash" --bootstrap 2000
```

Model lokal Ollama (`--model qwen3.5:2b`) juga didukung harness, tetapi tidak
dipakai untuk perbandingan ini.
