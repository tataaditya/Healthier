"""Merge vision pre-annotation batches into drafts and rebuild the viewer.

Reads every ``batch*.json`` (arrays of per-image dicts produced by the vision
pre-annotation pass) under ``.../annotation/batches/`` and fills the draft of
each record in ``drafts.json``. Raw OCR text from the earlier pass is preserved.
Then rewrites ``drafts.json``, ``ground_truth_draft.csv`` and ``viewer.html``.

Run:
    python experiments/scripts/ocr_eval_merge_drafts.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "experiments" / "results" / "ocr_eval" / "annotation"
BATCHES = ANN / "batches"

NUTRI = ["kalori", "lemak_total", "lemak_jenuh", "lemak_trans", "kolesterol",
         "natrium", "karbohidrat", "serat", "gula", "protein",
         "takaran_saji_g"]


def load_batches() -> dict[str, dict]:
    merged: dict[str, dict] = {}
    if not BATCHES.is_dir():
        return merged
    for path in sorted(BATCHES.glob("batch*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("SKIP (invalid json):", path.name)
            continue
        if isinstance(data, dict) and "items" in data:
            files = data.get("files") or []
            for f, item in zip(files, data["items"]):
                if isinstance(item, dict):
                    merged[f.replace("\\", "/")] = item
        elif isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                f = item.get("file") or item.get("path")
                if f:
                    merged[f.replace("\\", "/")] = item
        elif isinstance(data, dict):
            f = data.get("file") or data.get("path")
            if f:
                merged[f.replace("\\", "/")] = data
    return merged


def main() -> None:
    records = json.loads((ANN / "drafts.json").read_text(encoding="utf-8"))
    by_file = {r["file"]: r for r in records}
    merged = load_batches()
    hit = 0
    for f, item in merged.items():
        rec = by_file.get(f)
        if rec is None:
            print("WARN not in split:", f)
            continue
        absent = set(item.get("absent") or [])
        nut = {k: item.get(k) for k in NUTRI}
        rec["draft"] = {
            "takaran_saji_g": nut.get("takaran_saji_g"),
            "nutrisi": nut,
            "field_absent_on_label": {k: (k in absent) for k in NUTRI},
            "confidence": item.get("confidence", ""),
            "note": item.get("note", ""),
        }
        hit += 1

    (ANN / "drafts.json").write_text(json.dumps(records, ensure_ascii=False, indent=1),
                                     encoding="utf-8")

    with (ANN / "ground_truth_draft.csv").open("w", newline="", encoding="utf-8") as fh:
        cols = (["file", "folder", "verified", "confidence"]
                + NUTRI + [f"absent_{k}" for k in NUTRI] + ["note"])
        w = csv.writer(fh)
        w.writerow(cols)
        for r in records:
            d = r.get("draft", {})
            nut = d.get("nutrisi", {})
            ab = d.get("field_absent_on_label", {})
            w.writerow([r["file"], r["folder"], "no", d.get("confidence", "")]
                       + [nut.get(k) if nut.get(k) is not None else "" for k in NUTRI]
                       + [1 if ab.get(k) else 0 for k in NUTRI]
                       + [d.get("note", "")])

    tpl = (ROOT / "experiments" / "scripts" / "annotation_viewer_template.html").read_text(encoding="utf-8")
    (ANN / "viewer.html").write_text(tpl.replace("/*__RECORDS__*/",
                                   json.dumps(records, ensure_ascii=False)), encoding="utf-8")

    filled = sum(1 for r in records
                 if any((r.get("draft", {}).get("nutrisi", {}) or {}).get(k) is not None
                        for k in NUTRI))
    print(f"batches merged      : {hit}/{len(records)} records")
    print(f"records with values : {filled}/{len(records)}")
    print("wrote", ANN / "drafts.json")
    print("wrote", ANN / "ground_truth_draft.csv")
    print("wrote", ANN / "viewer.html")


if __name__ == "__main__":
    main()
