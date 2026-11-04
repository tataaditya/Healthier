"""Per-class COCO metrics from saved predictions + YOLO labels (no images needed).

The revision run saved COCO-style ``predictions.json`` files for each external
test split, and the YOLO polygon labels are still on disk. This script rebuilds
the COCO ground truth from the labels and runs ``pycocotools`` evaluation, so
per-class AP can be recovered without re-running the model or re-downloading
the images.

Example:
    python experiments/scripts/per_class_metrics.py
    python experiments/scripts/per_class_metrics.py --only uec_test
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results" / "segmentation"
EXTERNAL = ROOT / "experiments" / "dataset" / "external"

CLASSES = ["Staple Food", "Side Dishes", "Vegetables", "Fruits", "Others"]
SPLITS = {
    "uec_test": "uec_test",
    "foodseg_val": "foodseg_val",
    "fruitseg": "fruitseg",
}


def load_predictions(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def image_sizes(predictions: list[dict]) -> dict[str, tuple[int, int]]:
    sizes: dict[str, tuple[int, int]] = {}
    for pred in predictions:
        seg = pred.get("segmentation")
        if not isinstance(seg, dict):
            continue
        height, width = seg["size"]
        sizes.setdefault(pred["image_id"], (int(height), int(width)))
    return sizes


def polygon_area(points: list[float]) -> float:
    xs, ys = points[0::2], points[1::2]
    return 0.5 * abs(sum(xs[i] * ys[i - 1] - xs[i - 1] * ys[i] for i in range(len(xs))))


def build_coco(labels_dir: Path, predictions: list[dict]) -> tuple[COCO, dict]:
    sizes = image_sizes(predictions)
    label_files = sorted(labels_dir.glob("*.txt"))
    images = []
    annotations = []
    skipped: list[str] = []
    ann_id = 1
    for label_file in label_files:
        stem = label_file.stem
        if stem not in sizes:
            skipped.append(stem)
            continue
        height, width = sizes[stem]
        images.append({"id": stem, "file_name": f"{stem}.jpg", "height": height, "width": width})
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 7:
                continue
            category = int(parts[0]) + 1
            coords = [float(v) for v in parts[1:]]
            xs = [coords[i] * width for i in range(0, len(coords), 2)]
            ys = [coords[i] * height for i in range(1, len(coords), 2)]
            polygon = [v for pair in zip(xs, ys) for v in pair]
            annotations.append({
                "id": ann_id,
                "image_id": stem,
                "category_id": category,
                "segmentation": [polygon],
                "area": polygon_area(polygon),
                "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                "iscrowd": 0,
            })
            ann_id += 1
    dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i + 1, "name": name} for i, name in enumerate(CLASSES)
        ],
    }
    return COCO(), {"dataset": dataset, "skipped": skipped}


def coco_from_dict(coco: COCO, dataset: dict) -> COCO:
    coco.dataset = dataset
    coco.createIndex()
    return coco


def evaluate(coco_gt: COCO, predictions: list[dict], iou_type: str) -> dict:
    detections = []
    for pred in predictions:
        detections.append({
            "image_id": pred["image_id"],
            "category_id": pred["category_id"],
            "segmentation": pred["segmentation"],
            "bbox": pred["bbox"],
            "score": float(pred["score"]),
        })
    if not detections:
        return {}
    coco_dt = coco_gt.loadRes(detections) if iou_type == "segm" else coco_gt.loadRes(
        [{k: v for k, v in d.items() if k != "segmentation"} for d in detections]
    )
    coco_iou_type = "segm" if iou_type == "segm" else "bbox"
    with redirect_stdout(io.StringIO()):
        coco_eval = COCOeval(coco_gt, coco_dt, coco_iou_type)
        coco_eval.evaluate()
        coco_eval.accumulate()
    prec = coco_eval.eval["precision"]  # [T, R, K, A, M]
    recall = coco_eval.eval["recall"]  # [T, K, A, M]
    per_class = {}
    for k, name in enumerate(CLASSES):
        p50 = prec[0, :, k, 0, -1]
        p50 = p50[p50 > -1]
        pall = prec[:, :, k, 0, -1]
        pall = pall[pall > -1]
        ar = recall[:, k, 0, -1]
        ar = ar[ar > -1]
        per_class[name] = {
            "ap50": round(float(p50.mean()), 4) if p50.size else None,
            "ap50_95": round(float(pall.mean()), 4) if pall.size else None,
            "ar100": round(float(ar.mean()), 4) if ar.size else None,
        }
    return per_class


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--external", type=Path, default=EXTERNAL)
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    combined_rows = []
    summary: dict[str, dict] = {}

    for split in SPLITS:
        if only and split not in only:
            continue
        instance_dir = args.results / f"{split}_instance"
        pred_path = instance_dir / "predictions.json"
        labels_dir = args.external / split / "test" / "labels"
        if not pred_path.is_file() or not labels_dir.is_dir():
            print(f"skip {split}: missing {pred_path} or {labels_dir}")
            continue

        predictions = load_predictions(pred_path)
        coco, bundle = build_coco(labels_dir, predictions)
        coco_gt = coco_from_dict(coco, bundle["dataset"])

        gt_counts = Counter(a["category_id"] for a in bundle["dataset"]["annotations"])
        pred_counts = Counter(p["category_id"] for p in predictions)
        gt_images: dict[int, set] = defaultdict(set)
        for ann in bundle["dataset"]["annotations"]:
            gt_images[ann["category_id"]].add(ann["image_id"])

        segm = evaluate(coco_gt, predictions, "segm")
        box = evaluate(coco_gt, predictions, "box")

        rows = []
        for idx, name in enumerate(CLASSES, start=1):
            rows.append({
                "dataset": split,
                "class": name,
                "gt_instances": gt_counts.get(idx, 0),
                "gt_images": len(gt_images.get(idx, ())),
                "predictions": pred_counts.get(idx, 0),
                "box_ap50": box.get(name, {}).get("ap50"),
                "box_ap50_95": box.get(name, {}).get("ap50_95"),
                "mask_ap50": segm.get(name, {}).get("ap50"),
                "mask_ap50_95": segm.get(name, {}).get("ap50_95"),
                "mask_ar100": segm.get(name, {}).get("ar100"),
            })

        with (instance_dir / "per_class.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        lines = [
            f"# Per-class metrics - {split}",
            "",
            f"Images with predictions: **{len(bundle['dataset']['images'])}**; "
            f"skipped (no size info): {len(bundle['skipped'])}",
            "",
            "| Class | GT inst. | GT images | Preds | Box AP50 | Box AP50-95 | "
            "Mask AP50 | Mask AP50-95 | Mask AR100 |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            def fmt(value):
                return "-" if value is None else f"{100 * value:.2f}"
            lines.append(
                f"| {row['class']} | {row['gt_instances']} | {row['gt_images']} "
                f"| {row['predictions']} | {fmt(row['box_ap50'])} | {fmt(row['box_ap50_95'])} "
                f"| {fmt(row['mask_ap50'])} | {fmt(row['mask_ap50_95'])} "
                f"| {fmt(row['mask_ar100'])} |"
            )
        present = [r for r in rows if r["gt_instances"] > 0]
        macro50 = np.mean([r["mask_ap50"] for r in present if r["mask_ap50"] is not None])
        macro5095 = np.mean([r["mask_ap50_95"] for r in present if r["mask_ap50_95"] is not None])
        lines += [
            "",
            f"Macro mask AP50 over present classes: **{100 * macro50:.2f}**; "
            f"macro mask AP50-95: **{100 * macro5095:.2f}**.",
            "",
        ]
        (instance_dir / "per_class.md").write_text("\n".join(lines), encoding="utf-8")

        summary[split] = {
            "images": len(bundle["dataset"]["images"]),
            "annotations": len(bundle["dataset"]["annotations"]),
            "macro_mask_ap50": round(float(macro50), 4),
            "macro_mask_ap50_95": round(float(macro5095), 4),
            "per_class": segm,
        }
        combined_rows.extend(rows)
        print(f"{split}: images={len(bundle['dataset']['images'])} "
              f"macro mask AP50={100 * macro50:.2f} macro AP50-95={100 * macro5095:.2f}")

    if combined_rows:
        out_csv = args.results / "per_class_all.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(combined_rows[0].keys()))
            writer.writeheader()
            writer.writerows(combined_rows)
        (args.results / "per_class_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"wrote {out_csv}")


if __name__ == "__main__":
    sys.exit(main())
