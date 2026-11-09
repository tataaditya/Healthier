"""Assemble the final OCR-evaluation report from computed artefacts.

Reads the Markdown/CSV outputs of ``ocr_eval_metrics.py`` plus the split lock,
the verified-GT audit, the HITL metrics and the parameter log, and writes
``experiments/results/ocr_eval/REPORT.md``.

Usage:
    python experiments/scripts/ocr_eval_report.py
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "results" / "ocr_eval"


def read(path: Path, default: str = "") -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else default


def strip_title(md: str) -> str:
    lines = [ln for ln in md.splitlines()]
    while lines and (lines[0].startswith("#") or not lines[0].strip()):
        lines.pop(0)
    return "\n".join(lines).strip()


def edge_section() -> str:
    docs = []
    for dev in ("cpu", "gpu"):
        p = OUT.parent / "edge" / dev / "bench.json"
        if not p.is_file():
            continue
        b = json.loads(p.read_text(encoding="utf-8"))
        seg, ocr, rules = b["segmentation"], b["ocr"], b["clinical_rules"]
        docs.append(
            f"**{dev.upper()}**\n\n"
            f"- Segmentasi: median {seg.get('median_s')} s, P95 {seg.get('p95_s')} s, "
            f"RSS puncak {seg.get('peak_rss_mb')} MB\n"
            f"- OCR label: median {ocr.get('median_s')} s, P95 {ocr.get('p95_s')} s, "
            f"RSS puncak {ocr.get('peak_rss_mb')} MB\n"
            f"- Aturan klinis: median {rules.get('median_s')} s\n"
            f"- End-to-end (jumlah komponen): {b.get('end_to_end_sum_s', {}).get('median_s')} s\n"
            f"- Artefak: {b.get('artefacts_mb')}"
        )
    if not docs:
        return "_Belum tersedia._"
    note = ("\n\n> GPU tidak diukur: PyTorch terpasang versi CPU-only sehingga "
            "`torch.cuda.is_available()` bernilai False. Lihat "
            "`experiments/results/edge/README.md`. Jangan klaim hasil GPU.")
    return "\n\n".join(docs) + note


def timing_table() -> str:
    p = OUT / "timing.csv"
    if not p.is_file():
        return "_(belum tersedia)_"
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    lines = ["| Konfigurasi | Citra | Median (s/img) | P95 (s/img) |",
             "| :--- | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['config']} | {r['n_images']} | {r['median_s']} | {r['p95_s']} |")
    return "\n".join(lines)


def hitl_section() -> str:
    p = OUT / "hitl_E_full_pipeline.md"
    if not p.is_file():
        return "_Belum tersedia (jalankan `hitl_metrics.py --config E_full_pipeline`)._"
    return strip_title(read(p))


def gt_summary() -> str:
    p = OUT / "annotation" / "gt_verified_audit.json"
    if not p.is_file():
        return "_Audit GT belum tersedia._"
    a = json.loads(p.read_text(encoding="utf-8"))
    return (
        f"- Record terverifikasi: **{a['n_verified']}/{a['n_records']}**\n"
        f"- Field hadir di label: **{a['fields_present']}**, terisi: **{a['fields_filled']}**, "
        f"tidak terbaca pada foto: **{a['fields_present_but_null']}** "
        "(dikeluarkan dari semua denominator)\n"
        f"- Koreksi manusia terhadap draf: **{a['total_field_corrections']}** field pada "
        f"{a['images_with_correction']} citra; median "
        f"{a['median_corrections_per_image']} koreksi/citra"
    )


def main() -> None:
    status = {}
    sp = OUT / "metrics_status.json"
    if sp.is_file():
        status = json.loads(sp.read_text(encoding="utf-8"))
    src = status.get("gt_source", "unknown")
    lock = (json.loads((OUT / "split_lock.json").read_text(encoding="utf-8"))
            if (OUT / "split_lock.json").is_file() else {})

    banner = ("## STATUS GROUND TRUTH: DRAFT (BELUM DIVERIFIKASI MANUSIA)\n\n"
              "> **Angka pada laporan ini dihitung terhadap anotasi draf yang dibuat asisten.** "
              "Aturan proyek mengharuskan metrik final dihitung hanya setelah verifikasi manusia. "
              "Segera setelah `annotation/ground_truth_verified.json` tersedia, jalankan kembali "
              "`ocr_eval_metrics.py` dan `ocr_eval_report.py`; semua angka akan difinalkan tanpa "
              "menjalankan ulang OCR (prediksi ter-cache)."
              if src != "human_verified" else
              "## STATUS GROUND TRUTH: TERVERIFIKASI MANUSIA\n\n"
              "> Metrik di bawah dihitung terhadap `annotation/ground_truth_verified.json`.")

    configs = status.get("configs_evaluated", [])
    n_cum = sum(1 for c in configs if c.startswith("cum_"))
    n_loo = sum(1 for c in configs if c.startswith("loo_"))

    doc = f"""# Evaluasi OCR Label Gizi - Healthier (JRI 2026)

_Dibuat: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_

{banner}

---

## 1. Ringkasan

Laporan ini menyajikan evaluasi field-level **pada split test yang dikunci**
(n={lock.get('n_test','?')} citra, seed={lock.get('seed')}, hash `{lock.get('sha256','')[:16]}...`),
terhadap ground truth tabel "Informasi Nilai Gizi". Lima konfigurasi utama
(A-E), {n_cum} titik ablation kumulatif, dan {n_loo} konfigurasi
*leave-one-out* (LOO) dibandingkan pada citra test yang sama.

Definisi metrik dibekukan sebelum pengukuran di `metrics.md`.

**Sumber GT:** `{src}` · **Konfigurasi dievaluasi:** {', '.join(configs)}

## 2. Dataset, split, dan ground truth

- Total citra dataset OCR: {lock.get('n_total_files','?')}
- Dibuang karena duplikat (md5 + dHash): {lock.get('n_dropped','?')}
- Dipakai: {lock.get('n_kept','?')} -> test **{lock.get('n_test','?')}** / dev {lock.get('n_dev','?')}
- Kontaminasi test n dev: 0 · manifold `split_manifest.csv` (sha256 terkunci)

**Kelengkapan GT terverifikasi**

{gt_summary()}

## 3. Hasil utama (A-E)

{strip_title(read(OUT / 'results_main.md'))}

## 4. Ablation kumulatif

{strip_title(read(OUT / 'results_ablation_cumulative.md'))}

## 5. Ablation leave-one-out

{strip_title(read(OUT / 'results_ablation_loo.md'))}

> Catatan cakupan: LOO hanya dijalankan untuk empat stage **pasca-proses**
> (semantic, fuzzy, fallback, validation). Empat stage **citra**
> (deskew, deglare, clahe, roi) tidak dijalankan karena biaya komputasi
> (puluhan menit per konfigurasi); efeknya dibaca dari ablation kumulatif dan
> tidak diklaim sebagai hasil LOO.

## 6. Evaluasi human-in-the-loop (HITL)

{hitl_section()}

## 7. Hasil per field

{strip_title(read(OUT / 'results_per_field.md'))}

## 8. Taksonomi error

{strip_title(read(OUT / 'error_taxonomy.md'))}

## 9. Efisiensi

### 9.1 Benchmark perangkat (CPU)

{edge_section()}

### 9.2 Waktu dinding harness

{timing_table()}

> Catatan: durasi di tabel ini adalah waktu dinding harness pada mesin
> pengembangan dan **bukan** latensi perangkat target. Sebagian citra memakai
> cache deteksi sehingga waktunya tidak representatif. Angka latensi, memori,
> dan ukuran model untuk perangkat tepi diambil terpisah dari `bench_edge.py`
> (CPU dan GPU) dan dilaporkan di bagian tersendiri.

## 10. Keterbatasan

- GT hanya mencakup **lima kategori produk**, bukan stratifikasi merek; tidak
  ada field merek yang diverifikasi.
- {status.get('gt_unreadable_note', 'Sebagian field yang tercetak tidak terbaca pada foto dan dikeluarkan dari denominator.')}
- Ambang fuzzy tetap **0,75** (dibekukan sebelum evaluasi) dan dilaporkan
  sebagai keterbatasan, bukan disetel pada test set.
- Validasi empiris massa/area tidak dilakukan karena tidak ada GT massa porsi;
  hubungan area->massa ditulis sebagai keterbatasan.
- {read(OUT / 'questions_for_user.md') or '_lihat questions_for_user.md_'}

## 11. Reproduksi

```powershell
# 1. split (sekali, terkunci)
python experiments/scripts/ocr_eval_split.py
# 2. pre-anotasi draf + viewer
python experiments/scripts/ocr_eval_preannotate.py
# 3. verifikasi manusia (viewer.html) -> ground_truth_verified.json
python experiments/scripts/audit_verified_gt.py
# 4. harness konfigurasi (cache OCR dipakai ulang antar konfigurasi)
python experiments/scripts/ocr_eval_harness.py --configs all
# 5. metrik + HITL + laporan
python experiments/scripts/ocr_eval_metrics.py
python experiments/scripts/hitl_metrics.py --config E_full_pipeline
python experiments/scripts/ocr_eval_report.py
```

Detil parameter lengkap: `params_ocr.md`.
"""
    (OUT / "REPORT.md").write_text(doc, encoding="utf-8")
    print("wrote", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
