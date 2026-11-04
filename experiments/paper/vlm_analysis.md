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
- Harness: `experiments/scripts/eval_vlm.py` (prompt meminta JSON per sajian,
  guard anti-halusinasi "jangan mengarang", suhu 0, gambar ≤1600 px).
- Dua jalur:
  - `--provider ollama` → Ollama lokal/cloud (`gemma4:cloud`).
  - `--provider openai` → endpoint OpenAI-compatible (Helipod `https://ai.helipod.io/v1`).
- Model Helipod yang benar-benar vision (terverifikasi membaca gambar):
  `deepseek-v4.1-flash`, `netra/deepseek-v4.1-flash`, `kimi-k3`, `minimax-m3`,
  `glm-5.3-flash`, `glm-5.3-flashx`. Model tanpa vision atau error 400
  (`hy3`, `deepseek-v4-pro`, `glm-5.3`, `glm-5.2`, `glm-5.1`) tidak diuji.

## Hasil (seluruh model)

Angka di bawah memakai GT apa adanya (lihat catatan audit GT di bagian akhir).

| Pipeline | Akurasi | 95% CI | Exact | Close | Wrong | Missing | Latensi/gambar |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| Edge baseline (PP-OCRv4) | 14.89% | [4.65%, 22.92%] | — | — | 17.02% | 68.09% | ±2 s (GPU) |
| Edge grid parser (revisi) | 44.68% | [24.49%, 66.67%] | — | — | 19.15% | 36.17% | ±2 s (GPU) |
| VLM `gemma4:cloud` | 68.09% | [44.90%, 91.11%] | 68.09% | 0% | 25.53% | 6.38% | 4.3 s |
| VLM `deepseek-v4.1-flash` | 89.36% | [71.74%, 100%] | 87.23% | 2.13% | 8.51% | 2.13% | 8.6 s |
| VLM `netra/deepseek-v4.1-flash` | 89.36% | [71.74%, 100%] | 87.23% | 2.13% | 8.51% | 2.13% | 4.7 s |
| VLM `kimi-k3` | 89.36% | [71.74%, 100%] | 87.23% | 2.13% | 8.51% | 2.13% | 31.2 s |
| VLM `minimax-m3` | 87.23% | [69.57%, 98.04%] | 85.11% | 2.13% | 10.64% | 2.13% | 9.5 s |
| VLM `glm-5.3-flash` | 85.11% | [67.39%, 97.96%] | 82.98% | 2.13% | 12.77% | 2.13% | 15.0 s |
| VLM `glm-5.3-flashx` | 85.11% | [67.39%, 97.96%] | 82.98% | 2.13% | 12.77% | 2.13% | 7.5 s |

Per gambar — separuh sampel praktis sudah beres untuk semua VLM:

| Gambar | gemma4 | deepseek | minimax | glm-flash |
| :--- | ---: | ---: | ---: | ---: |
| 20241108_132045 | 7/7 | 7/7 | 7/7 | 7/7 |
| 20241108_133546 | 7/7 | 7/7 | 7/7 | 7/7 |
| 20241108_131859 | 7/10 | 10/10 | 10/10 | 10/10 |
| 20241108_132115 | 6/7 | 7/7 | 6/7 | 5/7 |
| 20241108_132308 | 3/8 | 7/8 | 7/8 | 7/8 |
| 20241108_134337 | 2/8 | 4/8 | 4/8 | 4/8 |

## Profil error — temuan pentingnya

Beban error **berpindah jenis**, bukan sekadar mengecil:

| | Edge grid | gemma4 | deepseek |
| :--- | ---: | ---: | ---: |
| Missing (tidak terbaca) | 36.17% (17/47) | 6.38% (3/47) | 2.13% (1/47) |
| Wrong (salah nilai) | 19.15% (9/47) | 25.53% (12/47) | 8.51% (4/47) |

- **Recall/coverage naik drastis.** Semua VLM nyaris selalu menemukan field-nya
  (`missing` 36% → 2–6%). Mode kegagalan khas pipeline edge — detektor
  menggabungkan grid angka & baris tak terdeteksi — praktis hilang.
- **`gemma4` justru menukar missing dengan wrong** (missing turun tapi wrong
  naik). Ini model vision terlemah di set ini.
- **Model frontier Helipod menekan keduanya**: `wrong` 9–13% dan `missing` 2%.

### Error yang tersisa hampir semuanya sama antar-model

Error `deepseek`, `netra`, dan `kimi` **identik** dan hanya 5 baris:

| Gambar | Field | GT | Prediksi |
| :--- | :--- | ---: | ---: |
| 132308 | gula | 9 | 10 |
| 134337 | lemak_total | 2.5 | 2.0 |
| 134337 | lemak_jenuh | 0.5 | (tidak ada) |
| 134337 | karbohidrat | 17.9 | 17 |
| 134337 | natrium | 40 | 60 |

Empat dari lima error itu **sama untuk keenam model Helipod**. Pola seperti ini
menandakan bottleneck-nya bukan model, melainkan **keterbacaan citra/format —
atau GT-nya sendiri** (lihat bagian berikutnya).

`gemma4` punya error yang lebih khas model (mis. `131859` gula 3 vs 10, `132308`
lemak_jenuh 0 vs 3, `natrium` 110 vs 100) — ini salah baca/baris, bukan sinyal
sistematis.

## Catatan audit GT (penting untuk kejujuran angka)

Pemeriksaan ulang visual menemukan **kesalahan transkripsi GT** pada 2 dari 6
gambar (detail + bukti crop di `experiments/paper/gt_audit.md`):

- `20241108_134337`: label aslinya `Lemak Total 2 g`, `Karbohidrat 17 g`,
  `Natrium 60 mg`, tanpa baris lemak jenuh; GT menulis 2.5 / 17.9 / 40 / 0.5.
- `20241108_132308`: label aslinya `Gula 10 g`; GT menulis 9.

Konsekuensinya, sebagian besar "wrong" VLM di atas **sebenarnya adalah kesalahan
GT, bukan kesalahan model**. Kalau GT dikoreksi, `deepseek`/`netra`/`kimi` jadi
46/46 (100%) dan `gemma4` 36/46 (78.3%). Skor 100% ini **tidak** boleh diklaim:
n terlalu kecil, dan koreksi GT menaikkan plafon.

> **Status: menunggu konfirmasi pemilik kemasan.** Angka resmi tetap memakai GT
> apa adanya sampai dikonfirmasi.

## Risiko yang harus dicatat (kalau kelak dipakai)

1. **Halusinasi digit.** Error VLM bukan "tidak terbaca" tapi "salah tapi yakin".
   Di konteks gizi/klinis ini **lebih berbahaya** dari missing, karena aturan
   deterministik di hilir tidak bisa mendeteksinya. Butuh lapisan verifikasi
   (silang-cek digit dengan recognizer, atau HITL wajib untuk field sensitif).
2. **Non-determinisme & reproducibility.** Output cloud tidak dijamin identik
   antar-run/versi; bertentangan dengan klaim C2.
3. **Biaya & ketersediaan.** Di akun Ollama gratis hanya `gemma4` yang vision;
   dua model vision kuat (`minimax-m3`, `deepseek-v4.1-flash`) butuh kredit.
   Perbandingan di atas memakai gateway Helipod.
4. **Klaim edge.** Pemakaian cloud menabrak C1 ("tanpa LLM cloud").
5. **n = 6 (47 field).** Terlalu kecil untuk klaim apa pun; ini baru indikasi arah.

## Implikasi untuk future work

1. **VLM sangat menjanjikan untuk *recall*.** Arah paling masuk akal: **hybrid** —
    VLM untuk menetapkan **struktur/kolom/baris** (kelemahan fatal edge),
    OCR deterministik untuk **membaca digit persisnya** (kelemahan VLM), lalu
    lapisan verifikasi silang untuk field sensitif (natrium/gula).
2. **On-device VLM belum diuji serius.** Model vision lokal kecil di RTX 3050 6 GB
    sempat dicoba, tetapi **di luar scope** dan tidak dilaporkan. Trade-off
    akurasi vs kecepatan vs determinisme masih terbuka.
3. **Audit GT & perbesar sampel dulu.** Sebelum klaim apa pun, perbaiki GT dan
    naikkan n (≥100 gambar). Tanpa itu, perbandingan apa pun di sini hanya
    indikatif.

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
```

Model lokal Ollama (`--model qwen3.5:2b`) juga didukung harness, tetapi tidak
dipakai untuk perbandingan ini.
