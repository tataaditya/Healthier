"""Evaluate a YOLO segmentation checkpoint and export paper-ready metrics.

Example:
    python experiments/scripts/eval_seg.py \
        --model experiments/runs/seg_main/weights/best.pt \
        --data experiments/dataset/data.yaml \
        --split test \
        --out experiments/results
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, (list, tuple)):
            return float(value[0]) if value else default
        return float(value)
    except (TypeError, ValueError):
        return default


def metric_at(metric: Any, name: str, index: int) -> float:
    values = getattr(metric, name, None)
    if values is None:
        return 0.0
    try:
        return float(values[index])
    except (TypeError, IndexError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO segmentation on a held-out split.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=Path("experiments/results"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"Model tidak ditemukan: {args.model}")
    if not args.data.is_file():
        raise SystemExit(f"data.yaml tidak ditemukan: {args.data}")

    from ultralytics import YOLO

    name = args.name or f"seg_{args.split}"
    out_dir = args.out.resolve() / name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    metrics = model.val(
        data=str(args.data.resolve()),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        plots=not args.no_plots,
        save_json=True,
        project=str(args.out.resolve()),
        name=name,
        exist_ok=True,
        verbose=True,
    )

    names = {int(k): str(v) for k, v in (metrics.names or {}).items()}
    summary = {
        "model": str(args.model),
        "data": str(args.data),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": args.conf,
        "iou": args.iou,
        "box": {
            "precision": as_float(getattr(metrics.box, "mp", None)),
            "recall": as_float(getattr(metrics.box, "mr", None)),
            "map50": as_float(getattr(metrics.box, "map50", None)),
            "map50_95": as_float(getattr(metrics.box, "map", None)),
        },
        "mask": {
            "precision": as_float(getattr(metrics.seg, "mp", None)),
            "recall": as_float(getattr(metrics.seg, "mr", None)),
            "map50": as_float(getattr(metrics.seg, "map50", None)),
            "map50_95": as_float(getattr(metrics.seg, "map", None)),
        },
        "speeds_ms": {
            key: as_float(value)
            for key, value in (getattr(metrics, "speed", {}) or {}).items()
        },
        "save_dir": str(getattr(metrics, "save_dir", out_dir)),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    rows = []
    for class_id, class_name in names.items():
        rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "box_precision": metric_at(metrics.box, "p", class_id),
            "box_recall": metric_at(metrics.box, "r", class_id),
            "box_map50": metric_at(metrics.box, "ap50", class_id),
            "box_map50_95": metric_at(metrics.box, "ap", class_id),
            "mask_precision": metric_at(metrics.seg, "p", class_id),
            "mask_recall": metric_at(metrics.seg, "r", class_id),
            "mask_map50": metric_at(metrics.seg, "ap50", class_id),
            "mask_map50_95": metric_at(metrics.seg, "ap", class_id),
        })
    with (out_dir / "per_class.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["class_id"])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"# Segmentation evaluation ({args.split})",
        "",
        f"- Model: `{args.model}`",
        f"- Data: `{args.data}`",
        f"- Image size: {args.imgsz}",
        "",
        "| Metric | Box | Mask |",
        "| :--- | ---: | ---: |",
        f"| Precision | {summary['box']['precision']:.4f} | {summary['mask']['precision']:.4f} |",
        f"| Recall | {summary['box']['recall']:.4f} | {summary['mask']['recall']:.4f} |",
        f"| mAP50 | {summary['box']['map50']:.4f} | {summary['mask']['map50']:.4f} |",
        f"| mAP50-95 | {summary['box']['map50_95']:.4f} | {summary['mask']['map50_95']:.4f} |",
        "",
        "## Per class (mask)",
        "",
        "| Class | Precision | Recall | mAP50 | mAP50-95 |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['class_name']} | {row['mask_precision']:.4f} | {row['mask_recall']:.4f} "
            f"| {row['mask_map50']:.4f} | {row['mask_map50_95']:.4f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved: {out_dir}")
    print(f"  metrics.json, per_class.csv, summary.md")
    print(f"  confusion matrix: {out_dir / 'confusion_matrix_normalized.png'}")


if __name__ == "__main__":
    main()
