"""Run the OCR preprocessing ablation study.

Each variant is executed in a fresh subprocess so the PaddleOCR engine and
environment flags cannot leak between runs.

Example:
    python experiments/scripts/ablate_ocr.py \
        --data experiments/ocr_data/test \
        --out experiments/results/ocr_ablation \
        --bootstrap 2000

Then compare PP-OCRv4 and PP-OCRv5:
    python experiments/scripts/ablate_ocr.py --data ... --out .../v4 --ocr-version PP-OCRv4
    python experiments/scripts/ablate_ocr.py --data ... --out .../v5 --ocr-version PP-OCRv5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = Path(__file__).with_name("eval_ocr.py")

FLAG_NAMES = [
    "HEALTHIER_OCR_DESKEW",
    "HEALTHIER_OCR_DEGLARE",
    "HEALTHIER_OCR_CLAHE",
    "HEALTHIER_OCR_ROI",
    "HEALTHIER_OCR_SEMANTIC_FILTER",
    "HEALTHIER_OCR_FUZZY",
    "HEALTHIER_OCR_FALLBACK",
    "HEALTHIER_OCR_VALIDATION",
    "NUTRILABEL_USE_TABLE_STRUCTURE",
]

VARIANTS: dict[str, dict[str, str]] = {
    "raw": {},
    "+deskew": {"HEALTHIER_OCR_DESKEW": "1"},
    "+deglare": {"HEALTHIER_OCR_DESKEW": "1", "HEALTHIER_OCR_DEGLARE": "1"},
    "+clahe": {"HEALTHIER_OCR_DESKEW": "1", "HEALTHIER_OCR_DEGLARE": "1", "HEALTHIER_OCR_CLAHE": "1"},
    "+roi": {
        "HEALTHIER_OCR_DESKEW": "1", "HEALTHIER_OCR_DEGLARE": "1",
        "HEALTHIER_OCR_CLAHE": "1", "HEALTHIER_OCR_ROI": "1",
    },
    "+semantic": {
        "HEALTHIER_OCR_DESKEW": "1", "HEALTHIER_OCR_DEGLARE": "1",
        "HEALTHIER_OCR_CLAHE": "1", "HEALTHIER_OCR_ROI": "1",
        "HEALTHIER_OCR_SEMANTIC_FILTER": "1",
    },
    "+fuzzy": {
        "HEALTHIER_OCR_DESKEW": "1", "HEALTHIER_OCR_DEGLARE": "1",
        "HEALTHIER_OCR_CLAHE": "1", "HEALTHIER_OCR_ROI": "1",
        "HEALTHIER_OCR_SEMANTIC_FILTER": "1", "HEALTHIER_OCR_FUZZY": "1",
    },
    "+fallback": {
        "HEALTHIER_OCR_DESKEW": "1", "HEALTHIER_OCR_DEGLARE": "1",
        "HEALTHIER_OCR_CLAHE": "1", "HEALTHIER_OCR_ROI": "1",
        "HEALTHIER_OCR_SEMANTIC_FILTER": "1", "HEALTHIER_OCR_FUZZY": "1",
        "HEALTHIER_OCR_FALLBACK": "1",
    },
    "+validation": {
        "HEALTHIER_OCR_DESKEW": "1", "HEALTHIER_OCR_DEGLARE": "1",
        "HEALTHIER_OCR_CLAHE": "1", "HEALTHIER_OCR_ROI": "1",
        "HEALTHIER_OCR_SEMANTIC_FILTER": "1", "HEALTHIER_OCR_FUZZY": "1",
        "HEALTHIER_OCR_FALLBACK": "1", "HEALTHIER_OCR_VALIDATION": "1",
    },
    "full": {name: "1" for name in FLAG_NAMES},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR preprocessing ablation.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT_DIR / "experiments" / "results" / "ocr_ablation")
    parser.add_argument("--variants", default="all",
                        help="Comma-separated variant names, atau 'all'")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ocr-version", default=None)
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    if not args.data.is_dir():
        raise SystemExit(f"Folder data tidak ditemukan: {args.data}")
    if not EVAL_SCRIPT.is_file():
        raise SystemExit(f"eval_ocr.py tidak ditemukan: {EVAL_SCRIPT}")

    if args.variants.strip().lower() == "all":
        selected = list(VARIANTS)
    else:
        selected = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in selected if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"Variant tidak dikenal: {unknown}. Pilihan: {list(VARIANTS)}")

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for variant in selected:
        flags = {name: "0" for name in FLAG_NAMES}
        flags.update(VARIANTS[variant])
        env = {**os.environ, **flags}
        if args.ocr_version:
            env["NUTRILABEL_OCR_VERSION"] = args.ocr_version
        variant_dir = out_dir / variant.replace("+", "plus_")
        cmd = [
            sys.executable, str(EVAL_SCRIPT),
            "--data", str(args.data.resolve()),
            "--out", str(variant_dir),
            "--label", variant,
            "--bootstrap", str(args.bootstrap),
            "--seed", str(args.seed),
        ]
        if args.limit > 0:
            cmd += ["--limit", str(args.limit)]

        print("\n" + "=" * 70)
        print(f"Variant: {variant}")
        print(f"Flags: {json.dumps(flags)}")
        print("=" * 70)
        result = subprocess.run(cmd, env=env, cwd=ROOT_DIR)
        summary_path = variant_dir / "summary.json"
        if result.returncode != 0 or not summary_path.is_file():
            print(f"[warn] variant {variant} gagal (returncode={result.returncode})")
            if not args.keep_going:
                raise SystemExit(1)
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append({
            "variant": variant,
            "n_images": summary["n_images"],
            "n_fields": summary["n_fields"],
            "field_accuracy": summary["field_accuracy"],
            "ci_low": summary["field_accuracy_ci95"][0],
            "ci_high": summary["field_accuracy_ci95"][1],
            "image_level_accuracy": summary["image_level_accuracy"],
            "exact": summary["status_counts"].get("exact", 0),
            "close": summary["status_counts"].get("close", 0),
            "wrong": summary["status_counts"].get("wrong", 0),
            "missing": summary["status_counts"].get("missing", 0),
            "ocr_version": summary.get("ocr_version"),
        })

    if not rows:
        raise SystemExit("Tidak ada variant yang berhasil.")

    baseline = next((r["field_accuracy"] for r in rows if r["variant"] == "raw"), rows[0]["field_accuracy"])
    for row in rows:
        row["delta_vs_raw"] = round(row["field_accuracy"] - baseline, 4)

    fieldnames = list(rows[0].keys())
    with (out_dir / "ocr_ablation.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# OCR preprocessing ablation",
        "",
        "| Variant | Images | Fields | Field Acc. | 95% CI | Δ vs raw | Image-level | Exact | Close | Wrong | Missing |",
        "| :--- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        delta = f"{row['delta_vs_raw']:+.2%}"
        lines.append(
            f"| {row['variant']} | {row['n_images']} | {row['n_fields']} "
            f"| {row['field_accuracy']:.2%} | [{row['ci_low']:.2%}, {row['ci_high']:.2%}] "
            f"| {delta} | {row['image_level_accuracy']:.2%} "
            f"| {row['exact']} | {row['close']} | {row['wrong']} | {row['missing']} |"
        )
    (out_dir / "ocr_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved: {out_dir}")
    print(f"  ocr_ablation.csv, ocr_ablation.md")
    print("\n" + "\n".join(lines[2:]))


if __name__ == "__main__":
    main()
