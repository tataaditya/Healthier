"""End-to-end error propagation: OCR output -> clinical rule decision.

For every pilot product the SAFE/WARNING/DANGER status is computed twice: once
from the ground-truth nutrition fields and once from the OCR-extracted fields
(baseline parser and grid parser). The mismatch rate measures how often a
label-extraction error changes the clinical decision.

The spec model reused here is the SAME one that `verify_clinical_rules.js`
checked against `frontend/js/clinical-rules.js` (36/36 limits, 30/30 status
decisions matched). The pilot has only 6 products, so this is reported as an
exploratory propagation analysis, not a population estimate.

Usage:
    python experiments/scripts/e2e_propagation.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OCR_DATA = ROOT / "experiments" / "ocr_data" / "makanan_ringan"
RESULTS = ROOT / "experiments" / "results"
OUT = RESULTS / "clinical_rules"

PROFILES = {
    "P1_general": dict(age=30, gender="male", weight_kg=70, height_cm=170,
                       isDiabetic=False, hasNephropathy=False,
                       hasDyslipidemia=False, isHypertensive=False),
    "P2_diabetes": dict(age=45, gender="female", weight_kg=65, height_cm=158,
                        isDiabetic=True, hasNephropathy=False,
                        hasDyslipidemia=False, isHypertensive=False),
    "P3_hypertension": dict(age=55, gender="male", weight_kg=75, height_cm=168,
                            isDiabetic=False, hasNephropathy=False,
                            hasDyslipidemia=False, isHypertensive=True),
    "P4_diabetes_dyslipidemia": dict(age=50, gender="female", weight_kg=68, height_cm=160,
                                     isDiabetic=True, hasNephropathy=False,
                                     hasDyslipidemia=True, isHypertensive=False),
    "P5_diabetes_nephropathy": dict(age=60, gender="male", weight_kg=60, height_cm=165,
                                    isDiabetic=True, hasNephropathy=True,
                                    hasDyslipidemia=False, isHypertensive=False),
    "P6_hypertension_dyslipidemia": dict(age=58, gender="female", weight_kg=72, height_cm=155,
                                         isDiabetic=False, hasNephropathy=False,
                                         hasDyslipidemia=True, isHypertensive=True),
}

# OCR / GT field name -> internal name
SUGAR, SODIUM, SATFAT, PROTEIN = "gula", "natrium", "lemak_jenuh", "protein"


def bmr(p):
    base = 10 * p["weight_kg"] + 6.25 * p["height_cm"] - 5 * p["age"]
    return base + 5 if p["gender"] == "male" else base - 161


def limits(p):
    tdee = bmr(p) * 1.2
    L = dict(sugar=tdee * 0.10 / 4, sodium=2300.0, satfat=tdee * 0.10 / 9,
             protein=None, sod_warn=800.0)
    if p["isDiabetic"]:
        L["sugar"] = tdee * 0.05 / 4
        if p["hasDyslipidemia"]:
            L["satfat"] = tdee * 0.07 / 9
        if p["hasNephropathy"]:
            L["protein"] = p["weight_kg"] * 0.8
    if p["isHypertensive"]:
        L["sodium"] = 1500.0
        L["sod_warn"] = 600.0
    return L


def status(p, amount):
    """`amount` holds per-serving values: sugar, sodium, satfat, protein, sodium_per100."""
    L = limits(p)
    st = "SAFE"

    def warn():
        nonlocal st
        if st != "DANGER":
            st = "WARNING"

    sugar = amount["sugar"]
    if sugar > L["sugar"]:
        st = "DANGER"
    elif sugar > L["sugar"] * 0.5:
        warn()

    if p["isDiabetic"] and p["hasNephropathy"] and L["protein"]:
        if amount["protein"] > L["protein"] * 0.30:
            warn()

    if p["isHypertensive"]:
        if amount["sodium_per100"] > L["sod_warn"]:
            warn()
        if amount["sodium"] > L["sodium"] * 0.40:
            warn()

    if p["hasDyslipidemia"] or p["isDiabetic"]:
        if amount["satfat"] > L["satfat"] * 0.5:
            warn()

    return st


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def amount_from(fields, serving_g):
    """`fields` are PER-SERVING values (as stored by both GT and OCR)."""
    sugar = _num(fields.get(SUGAR))
    sodium = _num(fields.get(SODIUM))
    satfat = _num(fields.get(SATFAT))
    protein = _num(fields.get(PROTEIN))
    sodium_per100 = (sodium / serving_g * 100.0) if serving_g else 0.0
    return dict(sugar=sugar, sodium=sodium, satfat=satfat,
                protein=protein, sodium_per100=sodium_per100)


def load_gt(stem):
    data = json.loads((OCR_DATA / f"{stem}_gt.json").read_text(encoding="utf-8"))
    return data.get("nutrisi", {}), _num(data.get("takaran_saji_g") or 100)


def load_predictions(name):
    out = {}
    path = RESULTS / name / "predictions.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        stem = Path(obj["file"]).stem
        fields = {}
        for key, val in (obj.get("data") or {}).items():
            if isinstance(val, dict) and val.get("per_sajian") is not None:
                fields[key] = _num(val["per_sajian"])
        out[stem] = fields
    return out


def main():
    stems = sorted(p.name.replace("_gt.json", "") for p in OCR_DATA.glob("*_gt.json"))
    gt = {s: load_gt(s) for s in stems}
    preds = {name: load_predictions(name) for name in ("ocr_full", "ocr_full_grid")}

    rows = []
    agree = {name: 0 for name in preds}
    total = 0
    for stem in stems:
        gtn, serving = gt[stem]
        gt_amount = amount_from(gtn, serving)
        for pname, p in PROFILES.items():
            total += 1
            ref = status(p, gt_amount)
            row = {"product": stem, "profile": pname, "gt": ref}
            for name in preds:
                ocr_amount = amount_from(preds[name].get(stem, {}), serving)
                st = status(p, ocr_amount)
                row[name] = st
                if st == ref:
                    agree[name] += 1
            rows.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e2e_propagation.json").write_text(
        json.dumps({"products": len(stems), "decisions": total, "agree": agree, "rows": rows},
                   indent=2), encoding="utf-8")

    lines = [
        "# End-to-end propagation: OCR error -> clinical decision",
        "",
        f"Pilot products: {len(stems)}; profile x product decisions: {total}.",
        "",
        "Reference status is computed from the hand-transcribed ground truth; the "
        "candidate statuses come from the OCR fields saved by the two parsers.",
        "",
        f"- Baseline parser: status unchanged vs GT in **{agree['ocr_full']}/{total}** decisions "
        f"({agree['ocr_full'] / total:.1%}).",
        f"- Grid parser: status unchanged vs GT in **{agree['ocr_full_grid']}/{total}** decisions "
        f"({agree['ocr_full_grid'] / total:.1%}).",
        "",
        "| Product | Profile | GT status | Baseline | Grid |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in rows:
        lines.append(f"| {r['product']} | {r['profile']} | {r['gt']} | {r['ocr_full']} | {r['ocr_full_grid']} |")
    (OUT / "e2e_propagation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:8]))


if __name__ == "__main__":
    main()
