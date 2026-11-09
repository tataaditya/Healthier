"""Stratify the verified OCR test set by product category and export a clean CSV.

Reviewer point A3/B6 asks for a representative, stratified evaluation. This
script reports the distribution over the five dataset folders and the number of
usable fields per category, so the manuscript can state exactly what the 131
images cover. Brand-level stratification is *not* claimed (no verified brand
field exists).

Run:
    python experiments/scripts/ocr_eval_stratify.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "experiments" / "results" / "ocr_eval" / "annotation"
FIELDS = [
    "kalori", "lemak_total", "lemak_jenuh", "lemak_trans", "kolesterol",
    "natrium", "karbohidrat", "serat", "gula", "protein",
]


def category(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    return parts[2] if len(parts) > 2 else parts[-2]


def main() -> None:
    rows = json.loads((ANN / "ground_truth_verified.json").read_text(encoding="utf-8-sig"))
    drafts = {r["file"]: r for r in json.loads((ANN / "drafts.json").read_text(encoding="utf-8"))}

    by_cat: dict[str, dict] = {}
    for r in rows:
        c = category(r["file"])
        s = by_cat.setdefault(c, {
            "images": 0, "present_fields": 0, "filled_fields": 0,
            "all_absent_images": 0, "images_with_draft": 0, "images_with_vision_draft": 0,
        })
        ab = r.get("field_absent_on_label") or {}
        nut = r.get("nutrisi") or {}
        s["images"] += 1
        present = [k for k in FIELDS if not ab.get(k)]
        s["present_fields"] += len(present)
        s["filled_fields"] += sum(1 for k in present if nut.get(k) is not None)
        if not present:
            s["all_absent_images"] += 1
        d = (drafts.get(r["file"], {}).get("draft") or {})
        if d:
            s["images_with_draft"] += 1
        if d.get("confidence"):
            s["images_with_vision_draft"] += 1

    total = {
        "images": len(rows),
        "present_fields": sum(v["present_fields"] for v in by_cat.values()),
        "filled_fields": sum(v["filled_fields"] for v in by_cat.values()),
        "all_absent_images": sum(v["all_absent_images"] for v in by_cat.values()),
        "categories": len(by_cat),
        "vision_draft_coverage": sum(v["images_with_vision_draft"] for v in by_cat.values()) / len(rows),
    }

    # clean CSV (one row per image, explicit empty string for unresolved values)
    cols = ["file", "category", "verified", "takaran_saji_g", *FIELDS,
            *[f"absent_{k}" for k in FIELDS]]
    with (ANN / "ground_truth_verified.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            ab = r.get("field_absent_on_label") or {}
            nut = r.get("nutrisi") or {}
            w.writerow([
                r["file"], category(r["file"]), r.get("verified", False),
                nut.get("takaran_saji_g", r.get("takaran_saji_g")),
                *["" if nut.get(k) is None else nut.get(k) for k in FIELDS],
                *[int(bool(ab.get(k))) for k in FIELDS],
            ])

    lines = [
        "# Stratifikasi test set OCR (terverifikasi)",
        "",
        "> Dihasilkan oleh `experiments/scripts/ocr_eval_stratify.py`.",
        "",
        f"- Citra test: **{total['images']}** dalam **{total['categories']} kategori**",
        f"- Field hadir: **{total['present_fields']}**, terisi: **{total['filled_fields']}** "
        f"({total['filled_fields']/total['present_fields']:.1%})",
        f"- Citra tanpa tabel (semua field `absent`): **{total['all_absent_images']}**",
        f"- Cakupan draf visi: **{total['vision_draft_coverage']:.1%}** "
        "(sisanya diisi/diverifikasi manual)",
        "",
        "| Kategori | Citra | Field hadir | Field terisi | Semua absent | Punya draf visi |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c, s in sorted(by_cat.items(), key=lambda kv: -kv[1]["images"]):
        lines.append(f"| {c} | {s['images']} | {s['present_fields']} | {s['filled_fields']} | "
                     f"{s['all_absent_images']} | {s['images_with_vision_draft']} |")
    lines += [
        "",
        "**Catatan penting untuk naskah:** stratifikasi ini adalah **kategori produk**,",
        "bukan merek. Tidak ada field merek yang diverifikasi, sehingga klaim",
        "\"brand-stratified\" tidak boleh dibuat; tulis \"five product categories\".",
        "",
    ]
    (ANN / "ocr_stratification.md").write_text("\n".join(lines), encoding="utf-8")
    (ANN / "ocr_stratification.json").write_text(
        json.dumps({"by_category": by_cat, "total": total}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(json.dumps({"by_category": by_cat, "total": total}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
