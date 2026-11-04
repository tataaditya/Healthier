"""Train YOLO segmentation with a fully recorded configuration.

Example (RTX 3050 6 GB):
    python experiments/scripts/train_seg.py \
        --data experiments/dataset/data.yaml \
        --model yolo11s-seg.pt \
        --epochs 80 --imgsz 640 --batch 8 --device 0 \
        --project experiments/runs --name seg_main
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    modules = {
        "torch": "torch",
        "ultralytics": "ultralytics",
        "numpy": "numpy",
        "opencv": "cv2",
        "Pillow": "PIL",
    }
    for label, module_name in modules.items():
        try:
            module = __import__(module_name)
            versions[label] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[label] = "not-installed"
    return versions


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO segmentation for Isi Piringku.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolo11s-seg.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--hsv-h", type=float, default=0.01)
    parser.add_argument("--hsv-s", type=float, default=0.3)
    parser.add_argument("--hsv-v", type=float, default=0.2)
    parser.add_argument("--degrees", type=float, default=10.0)
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--mosaic", type=float, default=0.5)
    parser.add_argument("--erasing", type=float, default=0.4)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--cache", choices=["false", "ram", "disk"], default="disk")
    parser.add_argument("--project", type=Path, default=ROOT_DIR / "experiments" / "runs")
    parser.add_argument("--name", default="seg_main")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--no-deterministic", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_path = args.data.resolve()
    if not data_path.is_file():
        raise SystemExit(f"data.yaml tidak ditemukan: {data_path}")

    run_dir = args.project.resolve() / args.name
    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "cwd": str(ROOT_DIR),
        "git_commit": git_commit(),
        "data": str(data_path),
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "seed": args.seed,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "augmentation": {
            "hsv_h": args.hsv_h, "hsv_s": args.hsv_s, "hsv_v": args.hsv_v,
            "degrees": args.degrees, "translate": args.translate,
            "scale": args.scale, "fliplr": args.fliplr,
            "mosaic": args.mosaic, "erasing": args.erasing,
            "close_mosaic": args.close_mosaic,
        },
        "cache": args.cache,
        "deterministic": not args.no_deterministic,
        "packages": package_versions(),
        "hostname": platform.node(),
        "platform": platform.platform(),
    }
    print(json.dumps(config, indent=2, ensure_ascii=False))
    if args.dry_run:
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        seed=args.seed,
        deterministic=not args.no_deterministic,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        erasing=args.erasing,
        close_mosaic=args.close_mosaic,
        cache=False if args.cache == "false" else args.cache,
        project=str(args.project.resolve()),
        name=args.name,
        resume=args.resume,
        exist_ok=args.exist_ok,
        plots=True,
        save=True,
        val=True,
    )

    # Ultralytics may append a numeric suffix; locate the actual run directory.
    candidates = sorted(
        [p for p in args.project.resolve().glob(f"{args.name}*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    actual = candidates[-1] if candidates else run_dir
    (actual / "experiment_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nRun directory: {actual}")
    print(f"Best weights : {actual / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
