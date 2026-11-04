# Audit ground truth — pilot Makanan Ringan (n=6)

> **Status: temuan, menunggu konfirmasi pemilik kemasan.** Dokumen ini mencatat
> hasil pemeriksaan ulang visual GT `experiments/ocr_data/makanan_ringan/*_gt.json`
> terhadap foto labelnya. Ini memo internal, bukan klaim hasil.

## Latar belakang

Saat membandingkan 6 model VLM cloud pada pilot n=6, muncul pola mencurigakan:
beberapa "error" muncul **identik di semua model** yang berbeda arsitektur.
Contoh: keenam model Helipod sama-sama membaca `natrium 60`, `karbohidrat 17`,
dan `lemak_total 2` pada gambar `20241108_134337`, sedangkan GT menulis
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

Nilai pada label (dibaca dari crop `134337_kolom_nilai.png` + `134337_lemak_total.png`):

| Field | Label (foto) | GT saat ini | Selisih | Bukti |
| :--- | ---: | ---: | :--- | :--- |
| kalori | 100 | 100 | cocok | — |
| lemak_total | **2 g** | 2.5 | GT salah | `134337_lemak_total.png` |
| lemak_jenuh | **tidak ada barisnya** | 0.5 | GT mengada-ada | `134337_kolom_nilai.png` |
| protein | 3 | 3 | cocok | — |
| karbohidrat | **17 g** | 17.9 | GT salah | `134337_kolom_nilai.png` |
| serat | 1 | 1 | cocok | — |
| gula | **5 g** (Gula Total) | 5.2 | GT salah | `134337_kolom_nilai.png` |
| natrium | **60 mg** | 40 | GT salah | `134337_kolom_nilai.png` |

Baris yang benar-benar tercetak di label: *Lemak Total 2 g · Asam α-Linolenat/
Omega 3 50 mg · Asam Linoleat/Omega 6 418 mg · Lemak Trans 0 g · Protein 3 g ·
Karbohidrat Total 17 g · Serat Pangan 1 g · Gula Total 5 g · Gula (Sukrosa) 2 g ·
Laktosa 3 g · Garam (Natrium) 60 mg*. Tidak ada baris **lemak jenuh**.

### `20241108_132308.jpg` (Choco Pie)

| Field | Label (foto) | GT saat ini | Selisih | Bukti |
| :--- | ---: | ---: | :--- | :--- |
| gula | **10 g** | 9 | GT salah | `132308_gula.png` |
| (lain-lain) | — | — | cocok | — |

## Dampak ke angka (sensitivitas)

Saya hitung ulang **semua** pipeline (OCR + VLM) memakai GT asli vs GT terkoreksi.
Skor dihitung dengan cara yang sama seperti `eval_ocr.py`/`eval_vlm.py`
benar = sama persis **atau** dalam toleransi 5%. Karena `lemak_jenuh` tidak ada
di label, field itu dikeluarkan → total field jadi 46 (dari 47).

| Pipeline | GT asli | GT terkoreksi |
| :--- | ---: | ---: |
| OCR baseline | 7/47 (14.9%) | 7/46 (15.2%) |
| OCR grid (revisi) | 21/47 (44.7%) | 22/46 (47.8%) |
| VLM `gemma4:cloud` | 32/47 (68.1%) | 36/46 (78.3%) |
| VLM `deepseek-v4.1-flash` | 42/47 (89.4%) | 46/46 (100%) |
| VLM `netra/deepseek-v4.1-flash` | 42/47 (89.4%) | 46/46 (100%) |
| VLM `kimi-k3` | 42/47 (89.4%) | 46/46 (100%) |
| VLM `minimax-m3` | 41/47 (87.2%) | 45/46 (97.8%) |
| VLM `glm-5.3-flash` | 40/47 (85.1%) | 44/46 (95.7%) |
| VLM `glm-5.3-flashx` | 40/47 (85.1%) | 44/46 (95.7%) |

## Rekomendasi

1. **Konfirmasi dulu** dengan kemasan fisik. Kalau nilai label memang seperti
   pada crop, GT `134337` dan `132308` perlu diperbaiki.
2. Kalau dikonfirmasi: perbaiki GT, jalankan ulang `eval_ocr.py` (GPU, ±2 menit),
   dan **skor ulang VLM secara offline** dari `predictions.jsonl` yang sudah ada
   (tanpa panggilan API lagi), lalu regenerasi `final_numbers.md`.
3. Kalau **tidak** diperbaiki: cantumkan audit ini sebagai limitation eksplisit —
   angka pilot memakai GT apa adanya, dan sebagian "error" VLM sebenarnya
   kesalahan GT.

## Reproduksi

Tabel sensitivitas di atas dihasilkan `experiments/scripts/audit_gt.py` (offline,
tanpa panggil model lagi) memakai patch kandidat
`experiments/paper/gt_patch_makanan_ringan.json`:

```powershell
python experiments/scripts/audit_gt.py `
  --data experiments/ocr_data/makanan_ringan `
  --patch experiments/paper/gt_patch_makanan_ringan.json
```

Skrip membaca `predictions.jsonl` dari `experiments/results/ocr_full*` dan
`experiments/results/ocr_vlm/*`. Untuk menerapkan koreksi secara resmi, ubah
`*_gt.json` lalu jalankan ulang `eval_ocr.py`; VLM cukup di-skor ulang offline
(karena prediksi tersimpan).

## Catatan

- `takaran_saji_g` tidak ikut diskor (metrik hanya menilai `per_sajian`).
- Grid parser edge (44.7%) hanya naik tipis (→47.8%) karena keunggulannya bukan
  di gambar yang GT-nya bermasalah.
- Skor 100% pada 3 model bukan berarti "sempurna": n=46 field dari 6 gambar
  terlalu kecil untuk klaim apa pun, dan koreksi GT menaikkan plafon.
