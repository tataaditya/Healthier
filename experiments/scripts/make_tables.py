"""Generate paper-ready tables from experiment result files.

Expected inputs (all optional; available ones are rendered):
    experiments/dataset/dataset_summary.json
    experiments/results/seg_test/metrics.json
    experiments/results/seg_test/per_class.csv
    experiments/results/ocr_full/summary.json
    experiments/results/ocr_full/per_field.csv
    experiments/results/ocr_ablation/ocr_ablation.csv

Example:
    python experiments/scripts/make_tables.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT_DIR / "experiments" / "results"
DEFAULT_DATASET = ROOT_DIR / "experiments" / "dataset"
DEFAULT_OUT = ROOT_DIR / "experiments" / "paper" / "tables"


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write(name: str, lines: list[str], out_dir: Path) -> None:
    path = out_dir / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper tables from results.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Generating tables...")

    dataset = read_json(args.dataset / "dataset_summary.json")
    if dataset:
        lines = [
            "# Dataset summary",
            "",
            "Cells are `images containing class / instances`.",
            "",
            "| Split | Staple | Side | Vegetables | Fruits | Others | Total images |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for split in ("train", "val", "test"):
            groups = dataset.get("per_group_present", {}).get(split, {})
            instances = dataset.get("instances", {}).get(split, {})
            total = dataset["splits"].get(split, 0)
            cells = [
                f"{groups.get(group, 0)} / {instances.get(group, 0)}"
                for group in ("makanan_pokok", "lauk_pauk", "sayur", "buah", "lainnya")
            ]
            lines.append(f"| {split} | " + " | ".join(cells) + f" | {total} |")
        lines += ["", f"Total images: **{dataset.get('records', 0)}**", ""]
        write("table_dataset.md", lines, args.out)

    seg_dir = args.results / "seg_test"
    metrics = read_json(seg_dir / "metrics.json")
    per_class = read_csv(seg_dir / "per_class.csv")
    if metrics:
        lines = [
            "# Segmentation test-set metrics",
            "",
            "| Metric | Box | Mask |",
            "| :--- | ---: | ---: |",
            f"| Precision | {metrics['box']['precision']:.4f} | {metrics['mask']['precision']:.4f} |",
            f"| Recall | {metrics['box']['recall']:.4f} | {metrics['mask']['recall']:.4f} |",
            f"| mAP50 | {metrics['box']['map50']:.4f} | {metrics['mask']['map50']:.4f} |",
            f"| mAP50-95 | {metrics['box']['map50_95']:.4f} | {metrics['mask']['map50_95']:.4f} |",
            "",
        ]
        if per_class:
            lines += [
                "| Class | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |",
                "| :--- | ---: | ---: | ---: | ---: |",
            ]
            for row in per_class:
                lines.append(
                    f"| {row['class_name']} | {float(row['mask_precision']):.4f} "
                    f"| {float(row['mask_recall']):.4f} | {float(row['mask_map50']):.4f} "
                    f"| {float(row['mask_map50_95']):.4f} |"
                )
        write("table_segmentation.md", lines, args.out)

    pixel_dir = args.results / "seg_pixel_test"
    pixel_summary = read_json(pixel_dir / "summary.json")
    pixel_rows = read_csv(pixel_dir / "pixel_metrics.csv")
    if pixel_summary and pixel_rows:
        lines = [
            "# Pixel-level segmentation (test)",
            "",
            f"- Macro: P={pixel_summary['macro']['precision']:.4f}, "
            f"R={pixel_summary['macro']['recall']:.4f}, F1={pixel_summary['macro']['f1']:.4f}",
            f"- Micro: P={pixel_summary['micro']['precision']:.4f}, "
            f"R={pixel_summary['micro']['recall']:.4f}, F1={pixel_summary['micro']['f1']:.4f}",
            "",
            "| Class | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in pixel_rows:
            lines.append(
                f"| {row['class_name']} | {float(row['accuracy']):.4f} "
                f"| {float(row['precision']):.4f} | {float(row['recall']):.4f} "
                f"| {float(row['f1']):.4f} | {row['tp']} | {row['fp']} | {row['fn']} | {row['tn']} |"
            )
        write("table_segmentation_pixel.md", lines, args.out)

    ocr_summary = read_json(args.results / "ocr_full" / "summary.json")
    if ocr_summary:
        lines = [
            "# OCR field-level results",
            "",
            f"- Images: **{ocr_summary['n_images']}**",
            f"- Fields: **{ocr_summary['n_fields']}**",
            f"- Field accuracy: **{ocr_summary['field_accuracy']:.2%}** "
            f"(95% CI {ocr_summary['field_accuracy_ci95'][0]:.2%}–{ocr_summary['field_accuracy_ci95'][1]:.2%})",
            f"- Image-level all-correct: **{ocr_summary['image_level_accuracy']:.2%}**",
            "",
        ]
        write("table_ocr_summary.md", lines, args.out)

    per_field = read_csv(args.results / "ocr_full" / "per_field.csv")
    if per_field:
        lines = [
            "# OCR per-field accuracy",
            "",
            "| Field | n | Accuracy | Exact | Close | Wrong | Missing |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in per_field:
            lines.append(
                f"| {row['field']} | {row['n']} | {float(row['accuracy']):.2%} "
                f"| {row['exact']} | {row['close']} | {row['wrong']} | {row['missing']} |"
            )
        write("table_ocr_per_field.md", lines, args.out)

    ablation = read_csv(args.results / "ocr_ablation" / "ocr_ablation.csv")
    if ablation:
        lines = [
            "# OCR preprocessing ablation",
            "",
            "| Variant | Field Acc. | 95% CI | Δ vs raw | Image-level |",
            "| :--- | ---: | :---: | ---: | ---: |",
        ]
        for row in ablation:
            lines.append(
                f"| {row['variant']} | {float(row['field_accuracy']):.2%} "
                f"| [{float(row['ci_low']):.2%}, {float(row['ci_high']):.2%}] "
                f"| {float(row['delta_vs_raw']):+.2%} | {float(row['image_level_accuracy']):.2%} |"
            )
        write("table_ocr_ablation.md", lines, args.out)

    # Baseline table: any JSON in results/baselines with method/metric keys.
    baseline_dir = args.results / "baselines"
    baseline_rows = []
    if baseline_dir.is_dir():
        for path in sorted(baseline_dir.glob("*.json")):
            data = read_json(path)
            if isinstance(data, dict):
                baseline_rows.append({"method": data.get("method", path.stem), **data})
    if baseline_rows:
        lines = [
            "# Baseline comparison",
            "",
            "| Method | Mask mAP50 | Mask mAP50-95 | OCR Field Acc. | Latency (ms) |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
        for row in baseline_rows:
            lines.append(
                f"| {row.get('method')} | {row.get('mask_map50', '-')} | {row.get('mask_map50_95', '-')} "
                f"| {row.get('ocr_field_accuracy', '-')} | {row.get('latency_ms', '-')} |"
            )
        write("table_baselines.md", lines, args.out)

    print(f"Done. Tables in {args.out}")


if __name__ == "__main__":
    main()
