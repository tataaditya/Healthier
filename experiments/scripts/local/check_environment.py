"""Print the environment and dataset status used for the revision experiments."""

from __future__ import annotations

import importlib
import platform
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as exc:
        return f"unavailable ({exc})"


def main() -> None:
    print("=" * 70)
    print("Environment")
    print("=" * 70)
    print(f"Python      : {platform.python_version()}")
    print(f"Platform    : {platform.platform()}")
    print(f"CPU         : {platform.processor() or 'unknown'}")
    print(f"Torch       : {version('torch')}")
    print(f"Ultralytics : {version('ultralytics')}")
    print(f"NumPy       : {version('numpy')}")
    print(f"OpenCV      : {version('cv2')}")
    print(f"Paddle      : {version('paddle')}")
    print(f"PaddleOCR   : {version('paddleocr')}")
    print(f"PyYAML      : {version('yaml')}")

    try:
        import torch

        print(f"CUDA avail. : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"GPU         : {props.name}")
            print(f"VRAM        : {props.total_memory / (1024 ** 3):.1f} GB")
    except Exception as exc:
        print(f"CUDA info   : unavailable ({exc})")

    print("\n" + "=" * 70)
    print("Paths")
    print("=" * 70)
    checks = {
        "dataset/data.yaml": ROOT_DIR / "experiments" / "dataset" / "data.yaml",
        "dataset_summary.json": ROOT_DIR / "experiments" / "dataset" / "dataset_summary.json",
        "OCR test folder": ROOT_DIR / "experiments" / "ocr_data" / "test",
        "runs folder": ROOT_DIR / "experiments" / "runs",
        "results folder": ROOT_DIR / "experiments" / "results",
    }
    for label, path in checks.items():
        print(f"{label:24s}: {'OK' if path.exists() else 'MISSING'}  {path}")

    ocr_test = ROOT_DIR / "experiments" / "ocr_data" / "test"
    if ocr_test.is_dir():
        images = [p for p in ocr_test.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        gts = list(ocr_test.glob("*_gt.json"))
        print(f"OCR images / GT         : {len(images)} / {len(gts)}")

    print("\nDisk free (GB):", round(shutil.disk_usage(ROOT_DIR).free / (1024 ** 3), 1))


if __name__ == "__main__":
    main()
