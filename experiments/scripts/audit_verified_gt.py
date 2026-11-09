"""Audit the human-verified OCR ground truth and quantify annotation effort.

Compares ``ground_truth_verified.json`` against (a) the locked test file list and
(b) the assistant's pre-annotation drafts. The draft-vs-verified delta is the
human correction burden (reviewer point B5) and doubles as a quality check on
the pre-annotation.

Run:
    python experiments/scripts/audit_verified_gt.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "experiments" / "results" / "ocr_eval" / "annotation"
SRC_CANDIDATES = [
    ANN / "ground_truth_verified.json",
    ROOT / "Ground_truth" / "ground_truth_verified.json",
]
FIELDS = [
    "kalori", "lemak_total", "lemak_jenuh", "lemak_trans", "kolesterol",
    "natrium", "karbohidrat", "serat", "gula", "protein",
]


def _num(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _same(a, b) -> bool:
    x, y = _num(a), _num(b)
    if x is None and y is None:
        return True
    if x is None or y is None:
        return False
    return abs(x - y) <= 1e-9


def load() -> tuple[list[dict], Path]:
    for p in SRC_CANDIDATES:
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8-sig")), p
    raise SystemExit("ground_truth_verified.json not found")


def main() -> None:
    rows, src = load()
    test = json.loads((ANN / "test_files.json").read_text(encoding="utf-8"))
    drafts = json.loads((ANN / "drafts.json").read_text(encoding="utf-8"))
    draft_by = {d["file"].replace("\\", "/"): d for d in drafts}

    files = [r["file"].replace("\\", "/") for r in rows]
    dup = [f for f, c in Counter(files).items() if c > 1]
    missing = sorted(set(test) - set(files))
    extra = sorted(set(files) - set(test))

    field_stats = {f: Counter() for f in FIELDS}
    corrections = Counter()
    per_image_corr: list[int] = []
    conf_corr: dict[str, list[int]] = defaultdict(list)
    detail: list[dict] = []

    for r in rows:
        f = r["file"].replace("\\", "/")
        absent = r.get("field_absent_on_label") or {}
        nut = r.get("nutrisi") or {}
        d = draft_by.get(f, {})
        dn = (d.get("draft") or {}).get("nutrisi") or {}
        img_corr = 0
        for k in FIELDS:
            gv = nut.get(k)
            is_abs = bool(absent.get(k))
            if is_abs:
                field_stats[k]["absent"] += 1
            else:
                field_stats[k]["present"] += 1
                if _num(gv) is None:
                    field_stats[k]["present_but_null"] += 1
                else:
                    field_stats[k]["filled"] += 1
            if not d:
                continue
            dv = dn.get(k)
            same = _same(dv, gv)
            change = "unchanged" if same else "changed"
            if is_abs and not same:
                change = "cleared_or_absent"
            elif _num(dv) is None and _num(gv) is not None:
                change = "filled_by_human"
            elif _num(dv) is not None and _num(gv) is None:
                change = "cleared_by_human"
            corrections[change] += 1
            if not same:
                img_corr += 1
                field_stats[k]["human_changes"] += 1
                detail.append({"file": f, "field": k, "draft": dv,
                               "verified": gv, "absent": is_abs, "kind": change})
            else:
                field_stats[k]["human_same"] += 1
        per_image_corr.append(img_corr)
        conf = (d.get("draft") or {}).get("confidence")
        if conf:
            conf_corr[conf].append(img_corr)

    n = len(rows)
    n_fields_present = sum(s["present"] for s in field_stats.values())
    n_filled = sum(s["filled"] for s in field_stats.values())
    total_corr = sum(per_image_corr)
    n_zero = sum(1 for c in per_image_corr if c == 0)
    per_image_corr_sorted = sorted(per_image_corr)

    out = {
        "source": str(src),
        "n_records": n,
        "n_verified": sum(1 for r in rows if r.get("verified")),
        "duplicates": dup,
        "missing_from_verified": missing,
        "extra_in_verified": extra,
        "fields_present": n_fields_present,
        "fields_filled": n_filled,
        "fields_present_but_null": n_fields_present - n_filled,
        "field_stats": {k: dict(v) for k, v in field_stats.items()},
        "draft_vs_verified": dict(corrections),
        "images_without_correction": n_zero,
        "images_with_correction": n - n_zero,
        "total_field_corrections": total_corr,
        "median_corrections_per_image": per_image_corr_sorted[len(per_image_corr_sorted) // 2],
        "max_corrections_per_image": max(per_image_corr) if per_image_corr else 0,
        "corrections_by_draft_confidence": {
            k: {"images": len(v), "total": sum(v),
                "mean_per_image": round(sum(v) / len(v), 2)}
            for k, v in conf_corr.items()
        },
        "detail": detail,
    }
    (ANN / "gt_verified_audit.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Audit GT terverifikasi (131 citra)",
        "",
        f"> Sumber: `{src.name}`. Dihasilkan oleh `experiments/scripts/audit_verified_gt.py`.",
        "",
        "## Kelengkapan",
        "",
        f"- Record: **{n}**, ditandai verified: **{out['n_verified']}**",
        f"- Duplikat: {len(dup)} · Hilang dari daftar test: {len(missing)} · "
        f"Ekstra: {len(extra)}",
        f"- Field hadir di label: **{n_fields_present}**, terisi: **{n_filled}**, "
        f"hadir-tapi-tak-terbaca: **{out['fields_present_but_null']}**",
        "",
        "## Per field",
        "",
        "| Field | Hadir | Absent | Terisi | Hadir tapi tak terbaca | Diubah manusia |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for k in FIELDS:
        s = field_stats[k]
        lines.append(f"| {k} | {s['present']} | {s['absent']} | {s['filled']} | "
                     f"{s['present_but_null']} | {s['human_changes']} |")

    lines += [
        "",
        "## Beban koreksi manusia vs draf",
        "",
        f"- Total koreksi field: **{total_corr}** dari {n * len(FIELDS)} sel "
        f"({total_corr / (n * len(FIELDS)):.1%})",
        f"- Citra tanpa koreksi: **{n_zero}/{n}** "
        f"({n_zero / n:.1%}); citra dengan ≥1 koreksi: {n - n_zero}",
        f"- Median koreksi per citra: **{out['median_corrections_per_image']}**, "
        f"maksimum: **{out['max_corrections_per_image']}**",
        "",
        "| Jenis | Jumlah |",
        "| :--- | ---: |",
    ]
    for k, v in sorted(corrections.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} |")

    lines += ["", "| Keyakinan draf | Citra | Total koreksi | Rata-rata/citra |",
              "| :--- | ---: | ---: | ---: |"]
    for k, v in sorted(conf_corr.items()):
        lines.append(f"| {k} | {len(v)} | {sum(v)} | {sum(v)/len(v):.2f} |")

    lines += [
        "",
        "## Catatan",
        "",
        "- Angka ini adalah **beban koreksi anotasi**, bukan bukti bahwa pengguna",
        "  aplikasi akan mengoreksi dengan pola yang sama.",
        "- `present_but_null` (38 field di 19 citra) berarti baris tercetak di label",
        "  tetapi nilainya **tidak terbaca pada foto** (bukan terlewat). Field ini",
        "  dikeluarkan dari denominator metrik dan dilaporkan terpisah sebagai",
        "  `unreadable`, sesuai keputusan pemverifikasi.",
        "- Koreksi terhadap draf dihitung eksak (tanpa toleransi); perubahan",
        "  pembulatan sekecil apa pun ikut terhitung.",
        "",
    ]
    (ANN / "gt_verified_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "detail"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
