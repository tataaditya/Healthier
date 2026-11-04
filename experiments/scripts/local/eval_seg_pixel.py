"""Pixel-level segmentation evaluation (TP/FP/FN/TN per class).

This reproduces the confusion-matrix style used in the original notebook
(Tabel 4.3/4.5) and complements the instance-level metrics from
``eval_seg.py``.

Example:
    python experiments/scripts/eval_seg_pixel.py \
        --model experiments/runs/seg_main/weights/best.pt \
        --data experiments/dataset/data.yaml \
        --split test \
        --out experiments/results
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def yolo_txt_to_mask(label_path: Path, height: int, width: int, num_classes: int):
    import cv2
    import numpy as np

    mask = np.zeros((height, width), dtype=np.uint8)
    if not label_path.is_file():
        return mask
    content = label_path.read_text(encoding="utf-8").strip()
    if not content:
        return mask
    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cls_id = int(parts[0])
        if not (0 <= cls_id < num_classes):
            continue
        coords = list(map(float, parts[1:]))
        pts = []
        for i in range(0, len(coords), 2):
            pts.append([int(coords[i] * width), int(coords[i + 1] * height)])
        pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts_np], cls_id + 1)
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Pixel-level segmentation confusion matrix.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="data.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT_DIR / "experiments" / "results")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"Model tidak ditemukan: {args.model}")
    if not args.data.is_file():
        raise SystemExit(f"data.yaml tidak ditemukan: {args.data}")

    import cv2
    import numpy as np
    import yaml
    from ultralytics import YOLO

    data = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    root = Path(data.get("path", args.data.parent))
    if not root.is_absolute():
        root = (args.data.parent / root).resolve()
    split_rel = str(data.get(args.split) or "")
    if not split_rel:
        raise SystemExit(f"Split '{args.split}' tidak ada di {args.data}")
    images_dir = Path(split_rel)
    if not images_dir.is_absolute():
        images_dir = root / images_dir
    label_rel = split_rel.replace("images", "labels", 1) if "images" in split_rel else split_rel
    labels_dir = Path(label_rel)
    if not labels_dir.is_absolute():
        labels_dir = root / labels_dir

    names = data.get("names")
    if isinstance(names, dict):
        class_names = [str(names[i]) for i in sorted(names)]
    else:
        class_names = [str(n) for n in names]
    num_classes = len(class_names)

    images = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise SystemExit(f"Tidak ada gambar di {images_dir}")

    model = YOLO(str(args.model))
    # rows = true class (0 background, 1..N), cols = predicted class
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    for index, image_path in enumerate(images, 1):
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        height, width = img.shape[:2]
        label_path = labels_dir / f"{image_path.stem}.txt"
        gt_mask = yolo_txt_to_mask(label_path, height, width, num_classes)

        result = model.predict(
            str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )[0]
        pred_mask = np.zeros((height, width), dtype=np.uint8)
        if result.masks is not None and result.boxes is not None:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            masks = result.masks.data.cpu().numpy()
            for cls_id, mask in zip(classes, masks):
                if not (0 <= cls_id < num_classes):
                    continue
                resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                pred_mask[resized > 0.5] = cls_id + 1

        flat = gt_mask.astype(np.int64) * (num_classes + 1) + pred_mask.astype(np.int64)
        counts = np.bincount(flat.ravel(), minlength=(num_classes + 1) ** 2)
        matrix += counts.reshape((num_classes + 1, num_classes + 1))

        if index % 25 == 0:
            print(f"  ... {index}/{len(images)}")

    out_dir = args.out.resolve() / (args.name or f"seg_pixel_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for class_id, class_name in enumerate(class_names):
        true_id = class_id + 1
        tp = int(matrix[true_id, true_id])
        fn = int(matrix[true_id, :].sum() - tp)
        fp = int(matrix[:, true_id].sum() - tp)
        tn = int(matrix.sum() - tp - fn - fp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
        rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": round(accuracy, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        })

    with (out_dir / "pixel_metrics.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    labels = ["background"] + class_names
    with (out_dir / "pixel_confusion_matrix.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true\\pred"] + labels)
        for i, label in enumerate(labels):
            writer.writerow([label] + matrix[i].tolist())

    macro_precision = sum(r["precision"] for r in rows) / len(rows)
    macro_recall = sum(r["recall"] for r in rows) / len(rows)
    macro_f1 = sum(r["f1"] for r in rows) / len(rows)
    micro_tp = sum(r["tp"] for r in rows)
    micro_fp = sum(r["fp"] for r in rows)
    micro_fn = sum(r["fn"] for r in rows)
    micro_precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall else 0.0

    summary = {
        "model": str(args.model),
        "data": str(args.data),
        "split": args.split,
        "n_images": len(images),
        "conf": args.conf,
        "iou": args.iou,
        "macro": {
            "precision": round(macro_precision, 6),
            "recall": round(macro_recall, 6),
            "f1": round(macro_f1, 6),
        },
        "micro": {
            "precision": round(micro_precision, 6),
            "recall": round(micro_recall, 6),
            "f1": round(micro_f1, 6),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        f"# Pixel-level segmentation ({args.split})",
        "",
        f"- Images: {len(images)}",
        f"- Conf: {args.conf} | IoU: {args.iou}",
        "",
        "| Class | Accuracy | Precision | Recall | F1 | TP | FP | FN | TN |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['class_name']} | {row['accuracy']:.4f} | {row['precision']:.4f} "
            f"| {row['recall']:.4f} | {row['f1']:.4f} | {row['tp']} | {row['fp']} | {row['fn']} | {row['tn']} |"
        )
    lines += [
        "",
        f"Macro: P={macro_precision:.4f}, R={macro_recall:.4f}, F1={macro_f1:.4f}",
        f"Micro: P={micro_precision:.4f}, R={micro_recall:.4f}, F1={micro_f1:.4f}",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved: {out_dir}")
    print(f"  pixel_metrics.csv, pixel_confusion_matrix.csv, summary.md")


if __name__ == "__main__":
    main()
