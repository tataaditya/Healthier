"""Build a single source of truth for manuscript numbers.

Reads the dataset summary, the YOLO training log, and the external-test
metric files produced by the revision run, then writes paper-ready Markdown
tables to ``experiments/paper/tables`` and a summary to
``experiments/paper/final_numbers.md``.

Usage:
    python experiments/scripts/make_final_numbers.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
DATASET = EXP / "dataset" / "revision" / "dataset_summary.json"
TRAIN_CSV = EXP / "runs" / "seg_main" / "results.csv"
RESULTS = EXP / "results" / "segmentation"
OCR_BASELINE = EXP / "results" / "ocr_full"
OCR_GRID = EXP / "results" / "ocr_full_grid"
OCR_ABLATION = EXP / "results" / "ocr_ablation"
CLINICAL = EXP / "results" / "clinical_rules"
OUT = EXP / "paper" / "tables"
SUMMARY = EXP / "paper" / "final_numbers.md"

EXTERNAL = {
    "FoodSeg103 validation": "foodseg_val",
    "UECFoodPIXCOMPLETE test": "uec_test",
    "FruitSeg30": "fruitseg",
}
CLASSES = ["Staple Food", "Side Dishes", "Vegetables", "Fruits", "Others"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def best_row(rows: list[dict]) -> dict:
    def fitness(row: dict) -> float:
        return 0.1 * float(row["metrics/mAP50-95(M)"]) + 0.9 * float(row["metrics/mAP50(M)"])

    return max(rows, key=fitness)


def count_manifest(root: Path) -> dict[tuple[str, str], dict[str, int]]:
    """Count images and instances per source/split from the manifest + labels."""
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for entry in manifest:
        key = (entry["source"], entry["split"])
        row = counts.setdefault(key, {"images": 0, **{c: 0 for c in CLASSES}})
        row["images"] += 1
        label = root / entry["split"] / "labels" / f"{Path(entry['image']).stem}.txt"
        if not label.is_file():
            continue
        for line in label.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row[CLASSES[int(line.split()[0])]] += 1
    return counts


def dataset_lines(d: dict, counts: dict[tuple[str, str], dict[str, int]]) -> list[str]:
    source_labels = {
        "uec": "UECFoodPIXCOMPLETE",
        "foodseg": "FoodSeg103",
        "fruitseg": "FruitSeg30",
    }
    lines = [
        "# Dataset summary",
        "",
        "Candidate corpus: **19,087** images; **373 duplicates removed**; "
        "**18,714 unique** images used.",
        "",
        "| Split | Source | Images | Staple | Side | Vegetables | Fruits | Others |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in ("train", "val"):
        for source in ("uec", "foodseg"):
            key = (source, split)
            if key not in counts:
                continue
            row = counts[key]
            lines.append(
                f"| {'Train' if split == 'train' else 'Validation'} | "
                f"{source_labels[source]} | {row['images']:,} | "
                + " | ".join(f"{row[c]:,}" for c in CLASSES)
                + " |"
            )
    for label, key in EXTERNAL.items():
        ext = d["external"][key]
        lines.append(
            f"| External test | {label} | {ext['total_images']:,} | "
            + " | ".join(f"{ext['instances'][c]:,}" for c in CLASSES)
            + " |"
        )
    lines += [
        "",
        "Cells are **instance counts**. External test splits were never used for "
        "training or model selection. The validation split is 15% of the training "
        "pool (not the 20% stated in the original draft).",
        "",
    ]
    return lines


def internal_lines(rows: list[dict]) -> list[str]:
    r = best_row(rows)
    return [
        "# Internal validation metrics (best checkpoint)",
        "",
        f"- Best epoch: **{r['epoch']}** (early stopping; max 100, patience 20)",
        f"- Training time: **{float(r['time']) / 3600:.2f} h** on Tesla T4 15.6 GB",
        "",
        "| Metric | Box | Mask |",
        "| :--- | ---: | ---: |",
        f"| Precision | {pct(float(r['metrics/precision(B)']))} | {pct(float(r['metrics/precision(M)']))} |",
        f"| Recall | {pct(float(r['metrics/recall(B)']))} | {pct(float(r['metrics/recall(M)']))} |",
        f"| mAP50 | {pct(float(r['metrics/mAP50(B)']))} | {pct(float(r['metrics/mAP50(M)']))} |",
        f"| mAP50-95 | {pct(float(r['metrics/mAP50-95(B)']))} | {pct(float(r['metrics/mAP50-95(M)']))} |",
        "",
    ]


def external_lines() -> list[str]:
    lines = [
        "# External-test instance-level metrics",
        "",
        "| Dataset | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, key in EXTERNAL.items():
        path = RESULTS / f"{key}_instance" / "metrics.json"
        if not path.is_file():
            lines.append(f"| {label} | — | — | — | — | — | — | — | — |")
            continue
        m = read_json(path)
        lines.append(
            f"| {label} | {pct(m['box']['precision'])} | {pct(m['box']['recall'])} "
            f"| {pct(m['box']['map50'])} | {pct(m['box']['map50_95'])} "
            f"| {pct(m['mask']['precision'])} | {pct(m['mask']['recall'])} "
            f"| {pct(m['mask']['map50'])} | {pct(m['mask']['map50_95'])} |"
        )
    lines.append("")
    return lines


def pixel_lines() -> list[str]:
    lines = [
        "# External-test pixel-level metrics",
        "",
        "| Dataset | Class | Accuracy | Precision | Recall | F1 |",
        "| :--- | :--- | ---: | ---: | ---: | ---: |",
    ]
    for label, key in EXTERNAL.items():
        path = RESULTS / f"{key}_pixel" / "pixel_metrics.csv"
        if not path.is_file():
            continue
        for row in read_csv(path):
            lines.append(
                f"| {label} | {row['class_name']} | {pct(float(row['accuracy']))} "
                f"| {pct(float(row['precision']))} | {pct(float(row['recall']))} "
                f"| {pct(float(row['f1']))} |"
            )
    lines.append("")
    return lines


def per_class_lines() -> list[str]:
    path = RESULTS / "per_class_all.csv"
    if not path.is_file():
        return []
    labels = {
        "uec_test": "UECFoodPIXCOMPLETE test",
        "foodseg_val": "FoodSeg103 validation",
        "fruitseg": "FruitSeg30",
    }
    lines = [
        "# Per-class instance-level metrics",
        "",
        "| Dataset | Class | GT inst. | Mask AP50 | Mask AP50-95 | Box AP50 | Box AP50-95 |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in read_csv(path):
        lines.append(
            f"| {labels.get(row['dataset'], row['dataset'])} | {row['class']} "
            f"| {row['gt_instances']} | {_fmt(row['mask_ap50'])} | {_fmt(row['mask_ap50_95'])} "
            f"| {_fmt(row['box_ap50'])} | {_fmt(row['box_ap50_95'])} |"
        )
    lines += [
        "",
        "Computed with `experiments/scripts/per_class_metrics.py` (pycocotools, "
        "ground truth rebuilt from the YOLO polygon labels, no images required). "
        "Classes with no GT in a split are left empty.",
        "",
    ]
    return lines


def _fmt(value: str) -> str:
    return "-" if value in ("", None) else pct(float(value))


def _ocr_summary(path: Path) -> dict | None:
    summary = path / "summary.json"
    if not summary.is_file():
        return None
    return read_json(summary)


def ocr_lines() -> list[str]:
    base = _ocr_summary(OCR_BASELINE)
    grid = _ocr_summary(OCR_GRID)
    lines = [
        "# OCR field-level evaluation (pilot)",
        "",
        "Indonesian snack-label photos with hand-transcribed ground truth. "
        "Field accuracy = exact value or within 5% of the per-serving value. "
        "CI is a cluster bootstrap over images.",
        "",
        "| Pipeline | Images | Fields | Field accuracy | 95% CI | Exact | Wrong | Missing |",
        "| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: |",
    ]
    for label, s in (("Baseline parser", base), ("Grid parser (revised)", grid)):
        if not s:
            continue
        rates = s.get("status_rates", {})
        ci = s.get("field_accuracy_ci95", [0.0, 0.0])
        lines.append(
            f"| {label} | {s['n_images']} | {s['n_fields']} | "
            f"{100 * s['field_accuracy']:.2f}% | "
            f"[{100 * ci[0]:.2f}%, {100 * ci[1]:.2f}%] | "
            f"{100 * rates.get('exact', 0):.2f}% | {100 * rates.get('wrong', 0):.2f}% | "
            f"{100 * rates.get('missing', 0):.2f}% |"
        )
    lines += [
        "",
        "The revised parser adds a row-label x value-column grid and splits tall "
        "numeric boxes that PP-OCR merges across adjacent rows (e.g. `1725` = "
        "1, 7, 2, 5), which separates the Per Sajian and %AKG columns. Standalone "
        "unit tokens (`mg`, `g`) are kept through the semantic filter.",
        "",
        "This is a **pilot (n=6)** and must be reported as exploratory, not as a "
        "generalisation result: the CI is wide and the sample is not stratified by "
        "brand or store. Rerun on >=100 images before citing a headline accuracy.",
        "",
    ]
    return lines


def ocr_ablation_lines() -> list[str]:
    path = OCR_ABLATION / "ocr_ablation.csv"
    if not path.is_file():
        return []
    rows = read_csv(path)
    lines = [
        "# OCR per-stage preprocessing ablation (pilot)",
        "",
        "Cumulative preprocessing chain with the baseline (non-grid) parser, six "
        "images and 46 fields; each row adds one stage to the row above. The grid "
        "parser is measured separately in the field-level table.",
        "",
        "| Variant | Field accuracy | 95% CI | delta vs raw | Exact | Wrong | Missing |",
        "| :--- | ---: | :--- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {100 * float(r['field_accuracy']):.2f}% | "
            f"[{100 * float(r['ci_low']):.2f}%, {100 * float(r['ci_high']):.2f}%] | "
            f"{100 * float(r['delta_vs_raw']):+.2f}% | {r['exact']} | {r['wrong']} | {r['missing']} |"
        )
    lines.append("")
    return lines


def clinical_lines() -> list[str]:
    ver = CLINICAL / "verification.json"
    e2e = CLINICAL / "e2e_propagation.json"
    if not ver.is_file() and not e2e.is_file():
        return []
    lines = ["# Clinical Rules Engine verification and propagation", ""]
    if ver.is_file():
        v = read_json(ver)
        lines += [
            f"- Daily-limit values matched against the specification: "
            f"**{v['limitMatch']}/{v['limitTotal']}**",
            f"- Status decisions matched against the specification: "
            f"**{v['statusMatch']}/{v['statusTotal']}**",
            "",
        ]
    if e2e.is_file():
        e = read_json(e2e)
        total = e["decisions"]
        agree = e["agree"]
        lines += [
            f"- End-to-end decisions (products x profiles): **{total}**",
            f"- Status unchanged vs GT (baseline parser): "
            f"**{agree['ocr_full']}/{total}** ({100 * agree['ocr_full'] / total:.1f}%)",
            f"- Status unchanged vs GT (grid parser): "
            f"**{agree['ocr_full_grid']}/{total}** ({100 * agree['ocr_full_grid'] / total:.1f}%)",
            "",
        ]
    return lines


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = read_json(DATASET)
    counts = count_manifest(DATASET.parent)
    rows = read_csv(TRAIN_CSV)

    dataset_md = dataset_lines(d, counts)
    internal_md = internal_lines(rows)
    external_md = external_lines()
    pixel_md = pixel_lines()
    per_class_md = per_class_lines()
    ocr_md = ocr_lines()
    ablation_md = ocr_ablation_lines()
    clinical_md = clinical_lines()

    for name, lines in (
        ("table_dataset.md", dataset_md),
        ("table_segmentation_internal.md", internal_md),
        ("table_segmentation_external.md", external_md),
        ("table_segmentation_pixel.md", pixel_md),
        ("table_segmentation_per_class.md", per_class_md),
        ("table_ocr.md", ocr_md),
        ("table_ocr_ablation.md", ablation_md),
        ("table_clinical.md", clinical_md),
    ):
        if not lines:
            continue
        (OUT / name).write_text("\n".join(lines), encoding="utf-8")
        print(f"wrote {OUT / name}")

    best = best_row(rows)
    summary = [
        "# Final numbers (single source of truth)",
        "",
        "> Generated by `experiments/scripts/make_final_numbers.py`.",
        "> Every number in the manuscript must trace back to this file or to "
        "`experiments/results/`.",
        "",
        "## Run configuration",
        "",
        "| Item | Value |",
        "| :--- | :--- |",
        "| Model | YOLO11s-seg |",
        "| Training images | 11,684 |",
        "| Validation images | 2,017 |",
        "| External test images | 5,013 (UEC 948; FoodSeg 2,096; FruitSeg 1,969) |",
        "| Candidate corpus | 19,087 images; 373 duplicates removed; 18,714 unique |",
        "| Max epochs | 100 |",
        "| Early stopping | patience 20; stopped at epoch 67 |",
        f"| Best checkpoint | epoch {best['epoch']} |",
        "| Optimizer | AdamW (lr0 0.001, lrf 0.01, weight decay 0.0005) |",
        "| Batch / image size | 16 / 640 |",
        "| Seed | 42 (deterministic) |",
        "| Hardware | Kaggle Tesla T4, 15.6 GB VRAM |",
        "| Training time | 4.01 h (14,420.7 s) |",
        "| Ultralytics / Torch | 8.4.199 / 2.10.0+cu128 |",
        "| Python | 3.12.13 |",
        "",
        "## Headline numbers",
        "",
        "**Internal validation, best checkpoint**",
        "",
        f"- Mask mAP50: **{pct(float(best['metrics/mAP50(M)']))}%**",
        f"- Mask mAP50-95: **{pct(float(best['metrics/mAP50-95(M)']))}%**",
        f"- Mask precision / recall: **{pct(float(best['metrics/precision(M)']))}% / "
        f"{pct(float(best['metrics/recall(M)']))}%**",
        f"- Box mAP50 / mAP50-95: **{pct(float(best['metrics/mAP50(B)']))}% / "
        f"{pct(float(best['metrics/mAP50-95(B)']))}%**",
        "",
        "**External tests (mask, %) — domain shift is part of the result**",
        "",
    ]
    for label, key in EXTERNAL.items():
        path = RESULTS / f"{key}_instance" / "metrics.json"
        if path.is_file():
            m = read_json(path)
            summary.append(
                f"- {label}: mAP50 **{pct(m['mask']['map50'])}**, "
                f"mAP50-95 **{pct(m['mask']['map50_95'])}**, "
                f"P **{pct(m['mask']['precision'])}**, R **{pct(m['mask']['recall'])}**"
            )

    per_class_path = RESULTS / "per_class_all.csv"
    if per_class_path.is_file():
        rows_pc = read_csv(per_class_path)
        summary += ["", "**Per-class mask AP50 (external)**", ""]
        for label, key in EXTERNAL.items():
            present = [
                r for r in rows_pc
                if r["dataset"] == key and r["mask_ap50"] not in ("", None)
            ]
            if not present:
                continue
            best_pc = max(present, key=lambda r: float(r["mask_ap50"]))
            worst_pc = min(present, key=lambda r: float(r["mask_ap50"]))
            summary.append(
                f"- {label}: best **{best_pc['class']} {pct(float(best_pc['mask_ap50']))}**, "
                f"worst **{worst_pc['class']} {pct(float(worst_pc['mask_ap50']))}**"
            )
    base_ocr = _ocr_summary(OCR_BASELINE)
    grid_ocr = _ocr_summary(OCR_GRID)
    if base_ocr or grid_ocr:
        summary += ["", "**OCR field-level accuracy (pilot, n=6)**", ""]
        for label, s in (("baseline parser", base_ocr), ("grid parser (revised)", grid_ocr)):
            if not s:
                continue
            ci = s.get("field_accuracy_ci95", [0.0, 0.0])
            summary.append(
                f"- {label}: **{100 * s['field_accuracy']:.2f}%** "
                f"(95% CI [{100 * ci[0]:.2f}%, {100 * ci[1]:.2f}%], "
                f"n={s['n_images']}, fields={s['n_fields']})"
            )
        summary.append(
            "- Exploratory pilot only (see `table_ocr.md`); rerun on >=100 images "
            "before citing a headline OCR number."
        )

    ablation_path = OCR_ABLATION / "ocr_ablation.csv"
    if ablation_path.is_file():
        summary += ["", "**OCR per-stage preprocessing ablation (pilot, n=6)**", ""]
        for r in read_csv(ablation_path):
            summary.append(
                f"- {r['variant']}: **{100 * float(r['field_accuracy']):.2f}%** "
                f"({100 * float(r['delta_vs_raw']):+.2f}% vs raw)"
            )
    ver_path = CLINICAL / "verification.json"
    e2e_path = CLINICAL / "e2e_propagation.json"
    if ver_path.is_file() or e2e_path.is_file():
        summary += ["", "**Clinical rules: verification and end-to-end propagation**", ""]
        if ver_path.is_file():
            v = read_json(ver_path)
            summary.append(
                f"- Implementation vs specification: limits "
                f"**{v['limitMatch']}/{v['limitTotal']}**, status decisions "
                f"**{v['statusMatch']}/{v['statusTotal']}**"
            )
        if e2e_path.is_file():
            e = read_json(e2e_path)
            t = e["decisions"]
            summary.append(
                f"- Status unchanged vs GT: baseline parser "
                f"**{e['agree']['ocr_full']}/{t}**, grid parser "
                f"**{e['agree']['ocr_full_grid']}/{t}**"
            )

    summary += [
        "",
        "## Manuscript-ready tables",
        "",
        "- `table_dataset.md`",
        "- `table_segmentation_internal.md`",
        "- `table_segmentation_external.md`",
        "- `table_segmentation_pixel.md`",
        "- `table_segmentation_per_class.md`",
        "- `table_ocr.md`",
        "- `table_ocr_ablation.md`",
        "- `table_clinical.md`",
        "",
        "## Status of previously missing items",
        "",
        "1. **Internal per-class metrics** — still open. External per-class AP is "
        "available (`table_segmentation_per_class.md`); internal validation per-class "
        "is not computable here because only the validation labels were retained "
        "(no validation images). Report as a limitation.",
        "2. **OCR evaluation** — RESOLVED and reported as an exploratory pilot: "
        "6-image baseline vs grid parser (`table_ocr.md`) plus the per-stage "
        "preprocessing ablation (`table_ocr_ablation.md`). The old 83.8% field-level "
        "accuracy is **not** reproduced and is not cited. Expansion to >=100 "
        "brand-stratified images remains future work.",
        "3. **Latency** — still open; the old 168 ms/image claim is not made. Inference "
        "latency on the target device is future work.",
        "4. **Baseline comparison** — reported as a 5-epoch screening pilot only; a "
        "matched full-budget baseline could not be trained on the CPU-only local "
        "environment and remains future work.",
        "5. **Clinical thresholds** — RESOLVED. Table 4 matches "
        "`frontend/js/clinical-rules.js`; the engine matched an independent "
        "re-implementation on 36/36 limits and 30/30 status decisions "
        "(`table_clinical.md`).",
        "6. **TFLite precision** — resolved by removal; no FP16 claim is made.",
        "7. **Usability** — reported from the author's Nielsen heuristic evaluation "
        "(four evaluators; no severity-4 findings, remaining findings severity 2-3; "
        "two-evaluator severity panel). Reported as exploratory, not as an end-user "
        "study.",
        "",
        "## What the FruitSeg result means",
        "",
        "FruitSeg30 is a cross-dataset, fruit-only test. The new model scores "
        "mask mAP50 17.06% and pixel F1 4.19% for Fruits, while predicting many "
        "non-fruit pixels as Staple Food. This is a genuine domain-shift failure, "
        "not a typo. Report it as a limitation and as evidence for the "
        "generalization/failure-analysis contribution. Do not repeat the old "
        "'Fruits is the best class' claim without clarifying that it refers to "
        "internal validation.",
        "",
    ]
    SUMMARY.write_text("\n".join(summary), encoding="utf-8")
    print(f"wrote {SUMMARY}")


if __name__ == "__main__":
    main()
