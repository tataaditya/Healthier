"""Langkah 2 - pre-annotation drafts + HTML verification viewer.

For every image of a split (default: the test set) this runs *raw* PaddleOCR
(no tuned pipeline) and fills a DRAFT ground-truth record. The draft is only a
typing aid: the human annotator reads the label photo and confirms/corrects it.
No metric is ever computed from a draft.

Outputs (``experiments/results/ocr_eval/annotation/``):
    drafts.jsonl        one JSON record per image, written incrementally
    drafts.json         same records as a single array (viewer input)
    ground_truth_draft.csv   flat CSV template (one row per image)
    viewer.html         self-contained verification UI (image | editable fields)

The OCR used here is intentionally *not* the full pipeline and runs on a
downscaled copy, so it is fast; nothing here is reused for evaluation.

Run:
    python experiments/scripts/ocr_eval_preannotate.py --split test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "experiments" / "results" / "ocr_eval"
ANN = OUT / "annotation"

NUTRI = ["kalori", "lemak_total", "lemak_jenuh", "lemak_trans", "kolesterol",
         "natrium", "karbohidrat", "serat", "gula", "protein"]

# keyword -> canonical field, checked longest-first
KEYWORDS = [
    ("takaran_saji", ["takaran saji", "takaran penyajian", "serving size",
                      "sajian per", "per sajian", "takaran"]),
    ("kalori", ["total energi", "energi total", "energi", "kalori", "energy"]),
    ("lemak_total", ["lemak total", "total lemak", "lemak"]),
    ("lemak_jenuh", ["lemak jenuh", "saturated fat", "saturated"]),
    ("lemak_trans", ["lemak trans", "trans fat"]),
    ("kolesterol", ["kolesterol", "cholesterol"]),
    ("karbohidrat", ["karbohidrat total", "total karbohidrat", "karbohidrat",
                     "carbohydrate"]),
    ("serat", ["serat pangan", "dietary fiber", "serat", "fiber"]),
    ("gula", ["total gula", "gula", "sugar"]),
    ("protein", ["protein"]),
    ("natrium", ["natrium", "sodium", "garam"]),
]

NUM_RE = re.compile(r"(\d+(?:[.,]\d+)?)")


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def parse_number(token: str) -> float | None:
    token = token.replace(",", ".")
    m = NUM_RE.search(token)
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    return value


def simple_parse(lines: list[str]) -> dict:
    """Very small baseline: first number on the line that matches a keyword."""
    draft: dict[str, float] = {}
    for raw in lines:
        line = clean(raw)
        if not line:
            continue
        for field, keys in KEYWORDS:
            if field in draft:
                continue
            if any(k in line for k in keys):
                # ignore a pure "%AKG" match
                if re.fullmatch(r"[^0-9]*\d+(?:[.,]\d+)?\s*%[^0-9]*", line):
                    continue
                value = parse_number(line)
                if value is not None:
                    draft[field] = value
                break
    return draft


def read_split(split: str) -> list[dict]:
    path = OUT / "split_manifest.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return [r for r in rows if r["split"] == split]


def relative_image(path_from: Path, target: Path) -> str:
    try:
        rel = os.path.relpath(target, path_from)
    except ValueError:
        rel = str(target)
    return rel.replace("\\", "/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--device", default=os.getenv("NUTRILABEL_OCR_DEVICE", "gpu"))
    ap.add_argument("--max-side", type=int, default=1800,
                    help="downscale longest side for the fast draft OCR")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.environ.setdefault("NUTRILABEL_USE_TABLE_STRUCTURE", "0")
    os.environ["NUTRILABEL_OCR_DEVICE"] = args.device

    import cv2
    import numpy as np
    from backend.app import PPOCREngine

    ANN.mkdir(parents=True, exist_ok=True)
    rows = read_split(args.split)
    if args.limit:
        rows = rows[: args.limit]
    print(f"split={args.split}  images={len(rows)}  device={args.device}")

    engine = PPOCREngine()
    t0 = time.time()
    engine.muat()
    print(f"engine ready in {time.time() - t0:.1f}s")

    jsonl = (ANN / "drafts.jsonl").open("w", encoding="utf-8")
    records: list[dict] = []
    for i, row in enumerate(rows, 1):
        image_path = ROOT / row["file"]
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise RuntimeError("cv2 could not read image")
            h, w = img.shape[:2]
            side = max(h, w)
            if side > args.max_side:
                s = args.max_side / side
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            res = engine.deteksi(img)
            lines = [d["text"] for d in res.get("deteksi_list", []) if d.get("text")]
            draft = simple_parse(lines)
        except Exception as exc:  # keep going; draft stays empty
            lines, draft = [], {}
            print(f"  [{i}] ERROR {row['file']}: {exc}")

        rec = {
            "file": row["file"],
            "folder": row["folder"],
            "group_id": row["group_id"],
            "verified": False,
            "annotator": "",
            "notes": "",
            "draft": {
                "takaran_saji_g": draft.get("takaran_saji"),
                "nutrisi": {k: draft.get(k) for k in NUTRI},
                "field_absent_on_label": {k: False for k in NUTRI + ["takaran_saji_g"]},
            },
            "raw_text": lines,
        }
        records.append(rec)
        jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
        jsonl.flush()
        print(f"  [{i}/{len(rows)}] {Path(row['file']).name}  n_text={len(lines)}")
    jsonl.close()

    (ANN / "drafts.json").write_text(json.dumps(records, ensure_ascii=False, indent=1),
                                     encoding="utf-8")

    with (ANN / "ground_truth_draft.csv").open("w", newline="", encoding="utf-8") as fh:
        cols = (["file", "folder", "verified", "takaran_saji_g"]
                + [f"{k}" for k in NUTRI]
                + [f"absent_{k}" for k in NUTRI + ["takaran_saji_g"]]
                + ["notes"])
        w = csv.writer(fh)
        w.writerow(cols)
        for r in records:
            d = r["draft"]
            w.writerow([r["file"], r["folder"], "no", d["takaran_saji_g"] or ""]
                       + [d["nutrisi"][k] if d["nutrisi"][k] is not None else "" for k in NUTRI]
                       + ["0" for _ in NUTRI + ["takaran_saji_g"]]
                       + [""])

    viewer = (ROOT / "experiments" / "scripts" / "annotation_viewer_template.html").read_text(encoding="utf-8")
    viewer = viewer.replace("/*__RECORDS__*/", json.dumps(records, ensure_ascii=False))
    (ANN / "viewer.html").write_text(viewer, encoding="utf-8")

    print("\nwrote", ANN / "drafts.json")
    print("wrote", ANN / "ground_truth_draft.csv")
    print("wrote", ANN / "viewer.html")


if __name__ == "__main__":
    main()
