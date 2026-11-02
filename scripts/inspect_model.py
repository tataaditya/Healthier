"""Inspeksi cepat bobot YOLO segmentasi Isi Piringku.

Contoh pemakaian (dari root proyek):
    python scripts/inspect_model.py
    python scripts/inspect_model.py --image data/samples/label-sample.jpg
    python scripts/inspect_model.py --model models/isipiringku_v2/weights/best.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app import resolve_yolo_model_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspeksi bobot YOLO segmentasi Isi Piringku.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default=None, help="Path bobot YOLO (default: auto-detect)")
    parser.add_argument(
        "--image",
        default=str(ROOT_DIR / "data" / "samples" / "label-sample.jpg"),
        help="Gambar uji",
    )
    parser.add_argument("--device", default=None, help="cpu / cuda / 0 (default: otomatis)")
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    model_path = Path(args.model) if args.model else resolve_yolo_model_path()
    if model_path is None or not model_path.is_file():
        raise SystemExit(
            "Bobot YOLO tidak ditemukan. Letakkan best.pt/last.pt/best.onnx di "
            "models/isipiringku_v2/weights/ atau gunakan --model."
        )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(model_path))
    print(f"Model      : {model_path}")
    print(f"Task       : {getattr(model, 'task', 'unknown')}")
    print(f"Device     : {device}")
    print(f"Kelas      : {model.names}")
    print(f"Gambar uji : {args.image}")

    for conf in (0.1, 0.25, 0.35, 0.5):
        results = model(args.image, conf=conf, verbose=False, device=device)
        result = results[0]
        boxes = result.boxes
        n_boxes = 0 if boxes is None else len(boxes)
        has_masks = result.masks is not None and len(result.masks.data) > 0
        print(f"\n--- conf={conf} | boxes={n_boxes} | masks={has_masks} ---")
        if n_boxes:
            classes = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()
            for i in range(n_boxes):
                print(
                    f"  [{i}] {result.names[int(classes[i])]} "
                    f"(class {classes[i]}) — conf: {confidences[i]:.4f}"
                )


if __name__ == "__main__":
    main()
