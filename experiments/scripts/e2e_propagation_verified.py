"""Decision propagation on the verified 131-image OCR test set.

Supersedes the six-product pilot analysis in ``e2e_propagation.py``. The
clinical status (SAFE / WARNING / DANGER) is computed twice for every test
image and every clinical profile: once from the verified ground truth and once
from a pipeline's extracted fields. The mismatch rate is the share of decisions
that label-extraction error would change.

No OCR is run here, so this is a pure, cheap re-analysis of artefacts that
already exist. The rule model is the same specification model that was checked
against ``frontend/js/clinical-rules.js`` (36/36 limits, 30/30 decisions).

Usage:
    python experiments/scripts/e2e_propagation_verified.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from e2e_propagation import PROFILES, amount_from, status

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "experiments" / "results" / "ocr_eval" / "annotation"
PRED = ROOT / "experiments" / "results" / "ocr_eval" / "predictions"
OUT = ROOT / "experiments" / "results" / "ocr_eval"
CONFIGS = ["C_paddle_raw", "D_baseline_parser", "E_full_pipeline"]
FIELDS = ["kalori", "lemak_total", "lemak_jenuh", "lemak_trans", "kolesterol",
          "natrium", "karbohidrat", "serat", "gula", "protein"]


def _num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_preds(name: str) -> dict[str, dict]:
    out = {}
    with (PRED / f"{name}.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[r["file"].replace("\\", "/")] = r
    return out


def main() -> None:
    gt_rows = json.loads((ANN / "ground_truth_verified.json").read_text(encoding="utf-8-sig"))
    preds = {c: load_preds(c) for c in CONFIGS}

    rows = []
    agree = Counter()
    total = 0
    total_with_table = 0
    agree_with_table = Counter()
    for g in gt_rows:
        f = g["file"].replace("\\", "/")
        ab = g.get("field_absent_on_label") or {}
        nut = g.get("nutrisi") or {}
        serving = _num(nut.get("takaran_saji_g") or g.get("takaran_saji_g"), 100.0)
        has_table = any(not ab.get(k) for k in FIELDS)
        gt_amount = amount_from(nut, serving)
        for pname, profile in PROFILES.items():
            ref = status(profile, gt_amount)
            total += 1
            if has_table:
                total_with_table += 1
            row = {"file": f, "has_table": has_table, "profile": pname, "gt": ref}
            for cfg in CONFIGS:
                pnut = (preds[cfg].get(f) or {}).get("nutrisi") or {}
                st = status(profile, amount_from(pnut, serving))
                row[cfg] = st
                if st == ref:
                    agree[cfg] += 1
                    if has_table:
                        agree_with_table[cfg] += 1
            rows.append(row)

    summary = {
        "n_images": len(gt_rows),
        "n_images_with_table": sum(1 for g in gt_rows
                                   if any(not (g.get("field_absent_on_label") or {}).get(k)
                                          for k in FIELDS)),
        "n_profiles": len(PROFILES),
        "n_decisions": total,
        "n_decisions_with_table": total_with_table,
        "agreement": {
            cfg: {
                "all": agree[cfg] / total if total else 0.0,
                "all_n": agree[cfg],
                "with_table": (agree_with_table[cfg] / total_with_table) if total_with_table else 0.0,
                "with_table_n": agree_with_table[cfg],
            }
            for cfg in CONFIGS
        },
    }
    (OUT / "e2e_propagation_verified.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    lines = [
        "# Propagasi keputusan klinis (131 citra terverifikasi)",
        "",
        "> Dihasilkan oleh `experiments/scripts/e2e_propagation_verified.py`. "
        "Tidak ada OCR yang dijalankan ulang; analisis memakai GT dan prediksi yang sudah ada.",
        "",
        f"- Citra: **{summary['n_images']}** "
        f"({summary['n_images_with_table']} dengan tabel gizi, "
        f"{summary['n_images'] - summary['n_images_with_table']} tanpa tabel)",
        f"- Profil klinis: **{summary['n_profiles']}**; total keputusan: **{total}**",
        "",
        "| Konfigurasi | Keputusan sama (semua) | Keputusan sama (citra bertabel) |",
        "| :--- | ---: | ---: |",
    ]
    for cfg in CONFIGS:
        a = summary["agreement"][cfg]
        lines.append(f"| {cfg} | {a['all_n']}/{total} ({a['all']:.1%}) | "
                     f"{a['with_table_n']}/{total_with_table} ({a['with_table']:.1%}) |")
    lines += [
        "",
        "## Catatan",
        "",
        "- `semua` mencakup 60 citra tanpa tabel gizi: nilai GT-nya nol, sehingga",
        "  setiap nilai yang dikeluarkan pipeline dihitung sebagai perubahan status.",
        "  Kolom `citra bertabel` memisahkan efek halusinasi dari galat pembacaan.",
        "- Model aturan sama dengan yang diverifikasi terhadap `clinical-rules.js`",
        "  (36/36 limit, 30/30 keputusan).",
        "- Status dihitung per sajian, sesuai nilai yang disimpan GT dan pipeline.",
        "",
    ]
    (OUT / "e2e_propagation_verified.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
