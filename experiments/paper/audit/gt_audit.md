# Audit ground truth — pilot Makanan Ringan (n=6)

> **Status: sudah diterapkan.** Hasil pemeriksaan ulang visual GT
> `experiments/ocr_data/makanan_ringan/*_gt.json` terhadap foto labelnya, dan
> koreksinya **sudah dipakai** untuk semua angka pilot (OCR + VLM). Ini memo
> internal, bukan klaim hasil.

## Latar belakang

Saat membandingkan 6 model VLM cloud pada pilot n=6, muncul pola mencurigakan:
beberapa "error" muncul **identik di semua model** yang berbeda arsitektur.
Contoh: keenam model Helipod sama-sama membaca `natrium 60`, `karbohidrat 17`,
dan `lemak_total 2` pada gambar `20241108_134337`, sedangkan GT lama menulis
`40`, `17.9`, dan `2.5`.

Karena 6 model independen berkata sama, hipotesisnya bukan "semua model salah
dengan cara yang sama", melainkan **GT-nya yang salah transkripsi**. Saya
verifikasi dengan memperbesar tabel dari foto asli.

> Crop bukti disimpan **lokal** di `experiments/results/ocr_vlm/gt_audit/`
> (`134337_kolom_nilai.png`, `134337_lemak_total.png`, `132308_gula.png`).
> Folder `experiments/results/` di-`.gitignore`, jadi crop tidak ikut ter-commit
> — ini disengaja karena citra label bersifat privat.

## Temuan (per gambar)

### `20241108_134337.jpg` (Milna Bubur Bayi)

| Field | Label (foto) | GT lama | Tindakan | Bukti |
| :--- | ---: | ---: | :--- | :--- |
| kalori | 100 | 100 | — | — |
| lemak_total | **2 g** | 2.5 | → **2** | `134337_lemak_total.png` |
| lemak_jenuh | **tidak ada barisnya** | 0.5 | **dibuang** | `134337_kolom_nilai.png` |
| protein | 3 | 3 | — | — |
| karbohidrat | **17 g** | 17.9 | → **17** | `134337_kolom_nilai.png` |
| serat | 1 | 1 | — | — |
| gula | **5 g** (Gula Total) | 5.2 | → **5** | `134337_kolom_nilai.png` |
| natrium | **60 mg** | 40 | → **60** | `134337_kolom_nilai.png` |

Baris yang benar-benar tercetak di label: *Lemak Total 2 g · Asam α-Linolenat/
Omega 3 50 mg · Asam Linoleat/Omega 6 418 mg · Lemak Trans 0 g · Protein 3 g ·
Karbohidrat Total 17 g · Serat Pangan 1 g · Gula Total 5 g · Gula (Sukrosa) 2 g ·
Laktosa 3 g · Garam (Natrium) 60 mg*. Tidak ada baris **lemak jenuh**.

### `20241108_132308.jpg` (Choco Pie)

| Field | Label (foto) | GT lama | Tindakan | Bukti |
| :--- | ---: | ---: | :--- | :--- |
| gula | **10 g** | 9 | → **10** | `132308_gula.png` |

Karena `lemak_jenuh` dibuang dari GT `134337`, total field pilot menjadi **46**
(dari 47).

## Dampak ke angka

Sensitivitas di bawah dihitung **offline** oleh `experiments/scripts/audit_gt.py`
(dari `predictions.jsonl`, tanpa memanggil model lagi). Kolom "GT terkoreksi"
adalah angka yang sekarang dipakai.

| Pipeline | GT lama | GT terkoreksi (dipakai) |
| :--- | ---: | ---: |
| OCR baseline | 7/47 (14.9%) | 10/46 (21.7%) |
| OCR grid (revisi) | 21/47 (44.7%) | 22/46 (47.8%) |
| VLM `gemma4:cloud` | 32/47 (68.1%) | 36/46 (78.3%) |
| VLM `deepseek-v4.1-flash` | 42/47 (89.4%) | 46/46 (100%) |
| VLM `netra/deepseek-v4.1-flash` | 42/47 (89.4%) | 46/46 (100%) |
| VLM `kimi-k3` | 42/47 (89.4%) | 46/46 (100%) |
| VLM `minimax-m3` | 41/47 (87.2%) | 45/46 (97.8%) |
| VLM `glm-5.3-flash` | 40/47 (85.1%) | 44/46 (95.7%) |
| VLM `glm-5.3-flashx` | 40/47 (85.1%) | 44/46 (95.7%) |

Catatan: angka OCR "GT lama" (14.9%) berasal dari run pra-koreksi **sebelum**
perbaikan semantic filter; run ulang kode terkini dengan GT lama tidak lagi bisa
direproduksi. Angka resmi ada di `final_numbers.md` dan `table_ocr.md`.

## Yang sudah dilakukan

1. Memperbaiki `*_gt.json` (`134337`, `132308`) sesuai label.
2. Menjalankan ulang `eval_ocr.py` (GPU) untuk `ocr_full` (baseline) dan
   `ocr_full_grid` (revisi) — beda hanya flag `HEALTHIER_OCR_TABLE_GRID`.
3. Men-skor ulang seluruh VLM **offline** lewat `eval_vlm.py --rescore`
   (prediksi tersimpan, tanpa biaya API).
4. Meregenerasi `final_numbers.md` + `table_ocr.md`.

## Reproduksi

```powershell
# 1) Audit / sensitivitas offline (pakai predictions.jsonl yang ada)
python experiments/scripts/audit_gt.py `
  --data experiments/ocr_data/makanan_ringan `
  --patch experiments/paper/audit/gt_patch_makanan_ringan.json

# 2) OCR ulang (baseline vs grid) dengan GT baru
$env:NUTRILABEL_OCR_DEVICE = "gpu:0"
$env:NUTRILABEL_USE_TABLE_STRUCTURE = "0"
$env:HEALTHIER_OCR_TABLE_GRID = "0"
python experiments/scripts/eval_ocr.py --data experiments/ocr_data/makanan_ringan `
  --out experiments/results/ocr_full --bootstrap 2000 --save-raw
$env:HEALTHIER_OCR_TABLE_GRID = "1"
python experiments/scripts/eval_ocr.py --data experiments/ocr_data/makanan_ringan `
  --out experiments/results/ocr_full_grid --bootstrap 2000 --save-raw

# 3) VLM: cukup skor ulang offline
python experiments/scripts/eval_vlm.py --rescore --provider openai `
  --model deepseek-v4.1-flash --data experiments/ocr_data/makanan_ringan `
  --out experiments/results/ocr_vlm/helipod_deepseek_v4_1_flash

# 4) Regenerasi angka
python experiments/scripts/make_final_numbers.py
```

## Catatan

- `takaran_saji_g` tidak ikut diskor (metrik hanya menilai `per_sajian`).
- Skor 100% pada 3 model bukan berarti "sempurna": 46 field dari 6 gambar
  terlalu kecil untuk klaim apa pun, dan koreksi GT menaikkan plafon.
- Pelajaran metodologis: perbedaan antar-model yang sangat besar vs GT sebaiknya
  memicu audit GT sebelum disimpulkan sebagai "model lemah".
