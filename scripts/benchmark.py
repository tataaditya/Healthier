"""Benchmark & evaluasi model Isi Piringku / API healthier.

Subcommand:
    val             Validasi YOLO pada dataset (butuh data.yaml)
    latency         Waktu inferensi lokal pada sekumpulan gambar
    api-latency     Latensi endpoint /api/ocr atau /api/isi-piringku
    dataset-stats   Ringkasan jumlah gambar & ground truth per folder

Contoh:
    python scripts/benchmark.py val --data dataset/data.yaml
    python scripts/benchmark.py latency --images data/samples --limit 5
    python scripts/benchmark.py api-latency --image data/samples/label-sample.jpg
    python scripts/benchmark.py dataset-stats --folder dataset
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app import resolve_yolo_model_path  # noqa: E402

IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")


def _load_model(model_arg: str | None):
    from ultralytics import YOLO

    model_path = Path(model_arg) if model_arg else resolve_yolo_model_path()
    if model_path is None or not model_path.is_file():
        raise SystemExit(
            "Bobot YOLO tidak ditemukan. Letakkan best.pt/last.pt/best.onnx di "
            "models/isipiringku_v2/weights/ atau gunakan --model."
        )
    return YOLO(str(model_path)), model_path


def _collect_images(folder: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in IMAGE_PATTERNS:
        files.extend(folder.rglob(pattern))
    return sorted(set(files), key=lambda p: str(p).lower())


def cmd_val(args: argparse.Namespace) -> None:
    model, model_path = _load_model(args.model)
    results = model.val(data=args.data, imgsz=args.imgsz, verbose=True)
    metrics: dict[str, dict] = {}
    for i, name in results.names.items():
        metrics[name] = {
            "precision": round(float(results.box.p[i]), 4),
            "recall": round(float(results.box.r[i]), 4),
            "mAP50_box": round(float(results.box.ap50[i]), 4),
            "mAP50_95_box": round(float(results.box.ap[i]), 4),
            "mAP50_mask": round(float(results.seg.ap50[i]), 4),
            "mAP50_95_mask": round(float(results.seg.ap[i]), 4),
        }
    print(f"\nModel: {model_path}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def cmd_latency(args: argparse.Namespace) -> None:
    import cv2

    model, model_path = _load_model(args.model)
    images = _collect_images(Path(args.images))[: args.limit]
    if not images:
        raise SystemExit(f"Tidak ada gambar di: {args.images}")

    times: list[float] = []
    print(f"Model: {model_path}")
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        start = time.perf_counter()
        result = model(img, verbose=False)
        ms = (time.perf_counter() - start) * 1000
        times.append(ms)
        print(f"{img_path.name:40s} {ms:8.1f} ms | deteksi: {len(result[0].boxes)}")

    if times:
        print(f"\nRata-rata : {sum(times) / len(times):.1f} ms")
        print(f"Min/Max   : {min(times):.1f} / {max(times):.1f} ms")


def cmd_api_latency(args: argparse.Namespace) -> None:
    import requests

    image_path = Path(args.image)
    if not image_path.is_file():
        raise SystemExit(f"Gambar tidak ditemukan: {image_path}")

    times: list[float] = []
    for i in range(args.runs):
        with image_path.open("rb") as fh:
            start = time.perf_counter()
            resp = requests.post(args.url, files={"file": fh}, timeout=args.timeout)
            ms = (time.perf_counter() - start) * 1000
        times.append(ms)
        print(f"Run {i + 1}: {ms:.0f} ms | status: {resp.status_code}")

    if times:
        print(f"\nRata-rata: {sum(times) / len(times):.0f} ms")


def cmd_dataset_stats(args: argparse.Namespace) -> None:
    root = Path(args.folder)
    if not root.is_dir():
        raise SystemExit(f"Folder tidak ditemukan: {root}")

    print("=== Per Subfolder ===")
    for folder in sorted(p for p in root.rglob("*") if p.is_dir()):
        images = [p for pattern in IMAGE_PATTERNS for p in folder.glob(pattern)]
        gts = list(folder.glob("*_gt.json"))
        if images:
            print(f"{folder.name:30s} | {len(images):4d} gambar | {len(gts)} GT")

    exts = Counter(f.suffix.lower() for f in root.rglob("*") if f.is_file())
    print("\n=== Distribusi Ekstensi ===")
    for ext, count in exts.most_common():
        print(f"  {ext or '(tanpa ekstensi)'}: {count}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("val", help="Validasi YOLO pada dataset")
    p_val.add_argument("--model", default=None)
    p_val.add_argument("--data", required=True, help="Path data.yaml")
    p_val.add_argument("--imgsz", type=int, default=640)
    p_val.set_defaults(func=cmd_val)

    p_lat = sub.add_parser("latency", help="Latensi inferensi lokal")
    p_lat.add_argument("--model", default=None)
    p_lat.add_argument("--images", default=str(ROOT_DIR / "data" / "samples"))
    p_lat.add_argument("--limit", type=int, default=5)
    p_lat.set_defaults(func=cmd_latency)

    p_api = sub.add_parser("api-latency", help="Latensi endpoint API")
    p_api.add_argument("--image", default=str(ROOT_DIR / "data" / "samples" / "label-sample.jpg"))
    p_api.add_argument("--url", default="http://127.0.0.1:8000/api/ocr")
    p_api.add_argument("--runs", type=int, default=3)
    p_api.add_argument("--timeout", type=float, default=120.0)
    p_api.set_defaults(func=cmd_api_latency)

    p_ds = sub.add_parser("dataset-stats", help="Statistik dataset")
    p_ds.add_argument("--folder", default=str(ROOT_DIR / "dataset"))
    p_ds.set_defaults(func=cmd_dataset_stats)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
