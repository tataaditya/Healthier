# Catatan kerja (audit) — Healthier / JRI 2026

Folder ini berisi jejak kerja penulisan dan audit revisi. **Bukan bagian dari naskah
yang dikirim ke jurnal.** Naskah final ada di folder induk (`../manuscript_jri.md` dan
`../manuscript_jri.docx`).

## Draf lama (digantikan `../manuscript_jri.md`)

| File | Isi |
| :--- | :--- |
| `manuscript_full.md` | Versi anotasi internal lengkap |
| `manuscript_intro_methods.md` | Draf lama: Introduction & Methods |
| `manuscript_results_conclusions.md` | Draf lama: Abstract, Results, Conclusions |
| `_publishing_docx.md` | Versi awal dari template lama "Publishing (1).docx" |

## Audit & verifikasi

| File | Isi |
| :--- | :--- |
| `notebook-audit.md` | Audit notebook training vs klaim manuscript |
| `gt_audit.md` + `gt_patch_makanan_ringan.json` | Audit ground truth pilot OCR (n=6) |
| `claims.csv` | Daftar setiap klaim + status verifikasi/aksi |
| `analysis_handoff.md` | Rangkuman lengkap seluruh analisis revisi |

## Perencanaan & panduan

| File | Isi |
| :--- | :--- |
| `experimental-design.md` | Desain eksperimen menjawab 4 komentar reviewer |
| `4-day-plan.md` | Rencana kerja deadline |
| `writing_guide.md` | Panduan menulis per bagian template |
| `response_to_reviewers.md` | Template jawaban reviewer |

## Pendukung

| File | Isi |
| :--- | :--- |
| `_referensi_index.txt` | Indeks isi PDF di `referensi/` |
| `vlm_analysis.md` | Analisis VLM multi-model (future work, bukan inti paper) |

> Catatan: `gt_patch_makanan_ringan.json` adalah default `--patch` untuk
> `experiments/scripts/audit_gt.py`.
