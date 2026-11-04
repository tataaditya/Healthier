"""Evaluate the OCR pipeline at field level with clustered bootstrap CIs.

Dataset layout (same as ``scripts/evaluate.py``):
    ocr_data/
      product_001.jpg
      product_001_gt.json   # {"takaran_saji_g": 20, "nutrisi": {...}}
      product_002.jpg
      product_002_gt.json

Example:
    python experiments/scripts/eval_ocr.py \
        --data experiments/ocr_data/test \
        --out experiments/results/ocr_full \
        --bootstrap 2000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

CANONICAL_FIELDS = [
    "kalori", "lemak_total", "lemak_jenuh", "lemak_trans", "kolesterol",
    "natrium", "karbohidrat", "serat", "gula", "protein",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_gt(path: Path) -> dict | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "nutrisi" in data:
        return data
    return {"nutrisi": data}


def numeric(value):
    if isinstance(value, dict):
        return value.get("per_sajian")
    return value


def bootstrap_ci(rows: list[tuple[float, int]], iterations: int, seed: int) -> tuple[float, float]:
    """Cluster bootstrap over images; rows are (correct_fields, total_fields)."""
    if not rows:
        return 0.0, 0.0
    rng = random.Random(seed)
    values = []
    n = len(rows)
    for _ in range(iterations):
        correct = 0.0
        total = 0
        for _ in range(n):
            c, t = rows[rng.randrange(n)]
            correct += c
            total += t
        values.append(correct / total if total else 0.0)
    values.sort()
    low = values[int(0.025 * (len(values) - 1))]
    high = values[int(0.975 * (len(values) - 1))]
    return low, high


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for module_name in ("paddle", "paddleocr", "cv2", "numpy"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[module_name] = f"unavailable: {exc}"
    return versions


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Field-level OCR evaluation with bootstrap CI.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT_DIR / "experiments" / "results" / "ocr_full")
    parser.add_argument("--gt-suffix", default="_gt.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--ocr-version", default=None)
    parser.add_argument("--label", default="")
    parser.add_argument("--save-raw", action="store_true")
    args = parser.parse_args()

    if not args.data.is_dir():
        raise SystemExit(f"Folder data tidak ditemukan: {args.data}")
    if args.ocr_version:
        os.environ["NUTRILABEL_OCR_VERSION"] = args.ocr_version

    # Import after env is set: OCR_VERSION is read at module import.
    from backend.app import hitung_field_accuracy, jalankan_ocr  # noqa: E402

    images = sorted(
        p for p in args.data.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"Tidak ada gambar di {args.data}")

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_image_rows: list[dict] = []
    error_rows: list[dict] = []
    field_counts: dict[str, Counter] = defaultdict(Counter)
    field_abs_error: dict[str, list[float]] = defaultdict(list)
    field_rel_error: dict[str, list[float]] = defaultdict(list)
    bootstrap_rows: list[tuple[float, int]] = []
    missing_gt: list[str] = []
    predictions_fh = (out_dir / "predictions.jsonl").open("w", encoding="utf-8") if args.save_raw else None

    try:
        for index, image_path in enumerate(images, 1):
            gt_path = image_path.with_name(image_path.stem + args.gt_suffix)
            gt = load_gt(gt_path)
            if gt is None:
                missing_gt.append(image_path.name)
                continue
            gt_nutrition = {
                key: value for key, value in (gt.get("nutrisi") or {}).items()
                if key in CANONICAL_FIELDS and value is not None
            }
            if not gt_nutrition:
                missing_gt.append(image_path.name)
                continue

            try:
                result = jalankan_ocr(str(image_path))
            except Exception as exc:
                result = {"status": "error", "data": {}, "flags": [f"exception:{exc}"]}

            pred = result.get("data") or {}
            evaluation = hitung_field_accuracy(gt_nutrition, pred, toleransi_pct=args.tolerance)
            per_field = evaluation["per_field"]
            correct = 0.0
            for field, info in per_field.items():
                status = info["status"]
                field_counts[field][status] += 1
                gt_value = info.get("gt")
                pred_value = info.get("pred")
                if status in ("exact", "close"):
                    correct += 1.0
                if pred_value is not None and gt_value is not None:
                    abs_error = abs(float(pred_value) - float(gt_value))
                    rel_error = abs_error / max(abs(float(gt_value)), 0.001)
                    field_abs_error[field].append(abs_error)
                    field_rel_error[field].append(rel_error)
                    if status == "wrong":
                        error_rows.append({
                            "file": image_path.name,
                            "field": field,
                            "gt": gt_value,
                            "pred": pred_value,
                            "abs_error": round(abs_error, 4),
                            "rel_error": round(rel_error, 4),
                            "status": status,
                            "flags": ";".join(map(str, result.get("flags", []))),
                        })
                elif status == "missing":
                    error_rows.append({
                        "file": image_path.name,
                        "field": field,
                        "gt": gt_value,
                        "pred": None,
                        "abs_error": "",
                        "rel_error": "",
                        "status": status,
                        "flags": ";".join(map(str, result.get("flags", []))),
                    })

            total = evaluation["total_fields"]
            bootstrap_rows.append((correct, total))
            per_image_rows.append({
                "file": image_path.name,
                "status": result.get("status", "unknown"),
                "n_fields": total,
                "correct": correct,
                "field_accuracy": round(correct / total, 4) if total else 0.0,
                "all_correct": int(correct == total),
                "flags": len(result.get("flags", [])),
                "duration_s": result.get("durasi_detik", ""),
                "gt": json.dumps(gt_nutrition, ensure_ascii=False),
            })

            if predictions_fh:
                record = {
                    "file": image_path.name,
                    "status": result.get("status"),
                    "data": result.get("data"),
                    "flags": result.get("flags"),
                    "ablation": result.get("ablation"),
                    "raw_ocr": result.get("raw_ocr"),
                    "nutrition_ocr": result.get("nutrition_ocr"),
                }
                predictions_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

            if index % 10 == 0:
                print(f"  ... {index}/{len(images)}")
    finally:
        if predictions_fh:
            predictions_fh.close()

    if not per_image_rows:
        raise SystemExit("Tidak ada gambar dengan ground truth yang valid.")

    total_correct = sum(row["correct"] for row in per_image_rows)
    total_fields = sum(row["n_fields"] for row in per_image_rows)
    field_accuracy = total_correct / total_fields if total_fields else 0.0
    image_accuracy = sum(row["all_correct"] for row in per_image_rows) / len(per_image_rows)
    mean_image_accuracy = sum(row["field_accuracy"] for row in per_image_rows) / len(per_image_rows)
    ci_low, ci_high = bootstrap_ci(bootstrap_rows, args.bootstrap, args.seed)

    status_totals = Counter()
    for counts in field_counts.values():
        status_totals.update(counts)
    total_observed = sum(status_totals.values())
    summary = {
        "label": args.label,
        "data": str(args.data),
        "n_images": len(per_image_rows),
        "n_gt_images_skipped": len(missing_gt),
        "n_fields": total_fields,
        "field_accuracy": round(field_accuracy, 4),
        "field_accuracy_ci95": [round(ci_low, 4), round(ci_high, 4)],
        "image_level_accuracy": round(image_accuracy, 4),
        "mean_image_field_accuracy": round(mean_image_accuracy, 4),
        "status_counts": dict(status_totals),
        "status_rates": {
            key: round(value / total_observed, 4) if total_observed else 0.0
            for key, value in status_totals.items()
        },
        "tolerance": args.tolerance,
        "bootstrap_iterations": args.bootstrap,
        "ocr_version": os.getenv("NUTRILABEL_OCR_VERSION", "PP-OCRv4"),
        "ablation": {
            key: os.getenv(key)
            for key in (
                "HEALTHIER_OCR_DESKEW", "HEALTHIER_OCR_DEGLARE", "HEALTHIER_OCR_CLAHE",
                "HEALTHIER_OCR_ROI", "HEALTHIER_OCR_SEMANTIC_FILTER", "HEALTHIER_OCR_FUZZY",
                "HEALTHIER_OCR_FALLBACK", "HEALTHIER_OCR_VALIDATION",
                "NUTRILABEL_USE_TABLE_STRUCTURE",
            )
        },
        "packages": package_versions(),
        "git_commit": git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (out_dir / "per_image.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)

    per_field_rows = []
    for field in CANONICAL_FIELDS:
        counts = field_counts.get(field, Counter())
        n = sum(counts.values())
        if n == 0:
            continue
        correct = counts["exact"] + counts["close"]
        per_field_rows.append({
            "field": field,
            "n": n,
            "exact": counts["exact"],
            "close": counts["close"],
            "wrong": counts["wrong"],
            "missing": counts["missing"],
            "accuracy": round(correct / n, 4),
            "exact_rate": round(counts["exact"] / n, 4),
            "mean_abs_error": round(sum(field_abs_error[field]) / len(field_abs_error[field]), 4)
            if field_abs_error[field] else "",
            "mean_rel_error": round(sum(field_rel_error[field]) / len(field_rel_error[field]), 4)
            if field_rel_error[field] else "",
        })
    with (out_dir / "per_field.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_field_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_field_rows)

    with (out_dir / "errors.csv").open("w", encoding="utf-8", newline="") as fh:
        fieldnames = ["file", "field", "gt", "pred", "abs_error", "rel_error", "status", "flags"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(error_rows)

    lines = [
        f"# OCR evaluation{f' — {args.label}' if args.label else ''}",
        "",
        f"- Images evaluated: **{len(per_image_rows)}**",
        f"- GT images skipped: {len(missing_gt)}",
        f"- Fields evaluated: **{total_fields}**",
        f"- Field accuracy (exact + within {args.tolerance:.0%}): **{field_accuracy:.2%}**",
        f"- 95% CI (cluster bootstrap by image): **[{ci_low:.2%}, {ci_high:.2%}]**",
        f"- Image-level all-fields-correct: **{image_accuracy:.2%}**",
        f"- Mean per-image field accuracy: **{mean_image_accuracy:.2%}**",
        "",
        "| Status | Count | Rate |",
        "| :--- | ---: | ---: |",
    ]
    for key in ("exact", "close", "wrong", "missing"):
        count = status_totals.get(key, 0)
        lines.append(f"| {key} | {count} | {count / total_observed:.2%} |")
    lines += ["", "## Per field", "", "| Field | n | Accuracy | Exact | Close | Wrong | Missing |", "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in per_field_rows:
        lines.append(
            f"| {row['field']} | {row['n']} | {row['accuracy']:.2%} | {row['exact']} "
            f"| {row['close']} | {row['wrong']} | {row['missing']} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved: {out_dir}")
    print(f"  field_accuracy={field_accuracy:.4f} CI=[{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  images={len(per_image_rows)} fields={total_fields}")


if __name__ == "__main__":
    main()
