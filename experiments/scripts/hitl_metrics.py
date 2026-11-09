"""Quantify the human-in-the-loop (HITL) correction burden.

Reviewer point B4 asks for quantitative HITL evidence. The honest question is:
*if a human starts from the pipeline's output, how many field edits are needed
before it matches the verified ground truth?* This script answers exactly that,
using the same locked test set and verified GT as the accuracy metrics.

For every test image it classifies each GT field into one of:

* ``accepted``  - pipeline value already matches the verified GT;
* ``edit_wrong`` - pipeline produced a value, but a different one;
* ``edit_missing`` - pipeline produced no value where the label has one;
* ``edit_spurious`` - pipeline produced a value where the label has none
  (human must clear it);
* ``unreadable`` - label row exists but the value is not legible in the photo;
  excluded from every denominator (see ``gt_verified_audit.md``).

Reported statistics: field accept rate, edits per image, share of images that
need at least one edit, and per-field/per-category breakdowns.

Usage:
    python experiments/scripts/hitl_metrics.py --config E_full_pipeline
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "experiments" / "results" / "ocr_eval" / "annotation"
PRED = ROOT / "experiments" / "results" / "ocr_eval" / "predictions"
OUT = ROOT / "experiments" / "results" / "ocr_eval"
FIELDS = [
    "kalori", "lemak_total", "lemak_jenuh", "lemak_trans", "kolesterol",
    "natrium", "karbohidrat", "serat", "gula", "protein",
]


def norm(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def category(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    return parts[2] if len(parts) > 2 else parts[-2]


def load_predictions(name: str) -> dict[str, dict]:
    p = PRED / f"{name}.jsonl"
    out = {}
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["file"].replace("\\", "/")] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="E_full_pipeline")
    args = ap.parse_args()

    gt_rows = json.loads((ANN / "ground_truth_verified.json").read_text(encoding="utf-8-sig"))
    preds = load_predictions(args.config)

    field_stats = {f: Counter() for f in FIELDS}
    by_cat: dict[str, Counter] = defaultdict(Counter)
    per_image: list[dict] = []
    missing_images: list[str] = []
    actions_total = 0
    fields_total = 0
    accepted_total = 0

    for g in gt_rows:
        f = g["file"].replace("\\", "/")
        if f not in preds:
            missing_images.append(f)
            continue
        ab = g.get("field_absent_on_label") or {}
        nut = g.get("nutrisi") or {}
        pred = preds[f].get("nutrisi") or {}
        cat = category(f)
        actions = 0
        n_fields = 0
        accepted = 0
        detail = []
        for k in FIELDS:
            pv = norm(pred.get(k))
            if ab.get(k):
                if pv is not None:
                    field_stats[k]["edit_spurious"] += 1
                    by_cat[cat]["edit_spurious"] += 1
                    actions += 1
                    detail.append((k, "edit_spurious"))
                else:
                    field_stats[k]["absent_ok"] += 1
                continue
            gv = norm(nut.get(k))
            if gv is None:
                field_stats[k]["unreadable"] += 1
                continue
            n_fields += 1
            if pv is None:
                kind = "edit_missing"
            elif abs(pv - gv) <= 1e-6:
                kind = "accepted"
            else:
                kind = "edit_wrong"
            field_stats[k][kind] += 1
            by_cat[cat][kind] += 1
            if kind == "accepted":
                accepted += 1
            else:
                actions += 1
                detail.append((k, kind))
        if n_fields == 0:
            # all-absent image: only spurious values can create work
            per_image.append({"file": f, "category": cat, "fields": 0,
                              "accepted": 0, "actions": actions, "detail": detail})
            actions_total += actions
            continue
        actions_total += actions
        fields_total += n_fields
        accepted_total += accepted
        per_image.append({"file": f, "category": cat, "fields": n_fields,
                          "accepted": accepted, "actions": actions, "detail": detail})

    n_eval = len(per_image)
    img_with_actions = sum(1 for r in per_image if r["actions"] > 0)
    action_counts = sorted(r["actions"] for r in per_image)
    median_actions = action_counts[len(action_counts) // 2] if action_counts else 0
    max_actions = max(action_counts) if action_counts else 0
    spurious_total = sum(s["edit_spurious"] for s in field_stats.values())
    decidable = fields_total + spurious_total

    result = {
        "config": args.config,
        "n_test_images": len(gt_rows),
        "n_images_with_predictions": n_eval,
        "missing_images": missing_images,
        "fields_evaluated": fields_total,
        "fields_accepted_unchanged": accepted_total,
        "field_accept_rate": accepted_total / fields_total if fields_total else 0.0,
        "total_edit_actions": actions_total,
        "spurious_values_to_clear": spurious_total,
        "edit_rate_per_decidable_field": actions_total / decidable if decidable else 0.0,
        "images_needing_edits": img_with_actions,
        "share_images_needing_edits": img_with_actions / n_eval if n_eval else 0.0,
        "median_edits_per_image": median_actions,
        "mean_edits_per_image": actions_total / n_eval if n_eval else 0.0,
        "max_edits_per_image": max_actions,
        "field_stats": {k: dict(v) for k, v in field_stats.items()},
        "by_category": {c: dict(v) for c, v in by_cat.items()},
        "per_image": per_image,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"hitl_{args.config}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        f"# Beban koreksi HITL - {args.config}",
        "",
        "> Dihasilkan oleh `experiments/scripts/hitl_metrics.py`. "
        "Skenario: manusia mulai dari keluaran pipeline, bukan dari nol.",
        "",
        "## Ringkasan",
        "",
        f"- Citra test: **{len(gt_rows)}**, dengan prediksi: **{n_eval}**"
        + (f", **hilang: {len(missing_images)}**" if missing_images else ""),
        f"- Field dievaluasi (GT terverifikasi): **{fields_total}**",
        f"- Field diterima tanpa perubahan: **{accepted_total}** "
        f"({result['field_accept_rate']:.1%})",
        f"- Total aksi edit: **{actions_total}** "
        f"({result['edit_rate_per_decidable_field']:.1%} dari "
        f"{decidable} field yang dapat diputuskan)",
        f"- Citra yang butuh >=1 edit: **{img_with_actions}/{n_eval}** "
        f"({result['share_images_needing_edits']:.1%})",
        f"- Edit per citra: median **{median_actions}**, rata-rata "
        f"**{result['mean_edits_per_image']:.2f}**, maksimum **{max_actions}**",
        "",
        "## Per field",
        "",
        "| Field | Diterima | Salah nilai | Nilai hilang | Nilai berlebih | Tak terbaca |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for k in FIELDS:
        s = field_stats[k]
        lines.append(f"| {k} | {s['accepted']} | {s['edit_wrong']} | "
                     f"{s['edit_missing']} | {s['edit_spurious']} | {s['unreadable']} |")

    lines += ["", "## Per kategori", "",
              "| Kategori | Diterima | Salah nilai | Nilai hilang | Nilai berlebih |",
              "| :--- | ---: | ---: | ---: | ---: |"]
    for c, s in sorted(by_cat.items()):
        lines.append(f"| {c} | {s['accepted']} | {s['edit_wrong']} | "
                     f"{s['edit_missing']} | {s['edit_spurious']} |")

    lines += [
        "",
        "## Catatan",
        "",
        "- Angka ini mengukur **usaha koreksi terhadap keluaran sistem**, bukan",
        "  waktu pengerjaan atau beban kognitif pengguna.",
        "- `edit_spurious` dihitung per field yang ditandai `absent` di GT tetapi",
        "  diisi nilai oleh pipeline; ini juga menambah usaha manusia.",
        "- Bila `n_images_with_predictions` < jumlah test, sisa citra belum",
        "  diprediksi dan tidak dihitung.",
        "",
    ]
    (OUT / f"hitl_{args.config}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("per_image", "field_stats", "by_category")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
