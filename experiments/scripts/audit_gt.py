"""Re-score existing predictions under a candidate GT patch (offline).

Membandingkan skor field-level **tanpa memanggil model lagi**: cukup baca
``predictions.jsonl`` yang sudah ada dari ``eval_ocr.py`` / ``eval_vlm.py``.

Dipakai oleh `experiments/paper/gt_audit.md` untuk menghitung sensitivitas
"GT asli vs GT terkoreksi". Patch GT adalah JSON:

    {"<nama_gambar>.jpg": {"<field>": <nilai_baru>, "<field>": null}}

``null`` berarti field tersebut dikeluarkan dari GT (mis. tidak tercetak di label).

Contoh:
    python experiments/scripts/audit_gt.py ^
        --data experiments/ocr_data/makanan_ringan ^
        --patch experiments/paper/gt_patch_makanan_ringan.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parents[1]
for candidate in (str(ROOT_DIR), str(SCRIPTS_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import eval_ocr  # noqa: E402


def load_patch(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def base_gt(data_dir: Path, name: str) -> dict:
    gt = eval_ocr.load_gt(data_dir / (Path(name).stem + "_gt.json"))
    return {
        key: value
        for key, value in (gt.get("nutrisi") or {}).items()
        if key in eval_ocr.CANONICAL_FIELDS and value is not None
    }


def patched_gt(data_dir: Path, patch: dict, name: str) -> dict:
    nutrition = dict(base_gt(data_dir, name))
    for field, value in patch.get(name, {}).items():
        if value is None:
            nutrition.pop(field, None)
        else:
            nutrition[field] = value
    return nutrition


def score_file(path: Path, data_dir: Path, patch: dict, patched: bool):
    from backend.app import hitung_field_accuracy

    correct = 0
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        name = record["file"]
        gt = patched_gt(data_dir, patch, name) if patched else base_gt(data_dir, name)
        pred = record.get("data") or {}
        evaluation = hitung_field_accuracy(gt, pred, toleransi_pct=0.05)
        correct += sum(
            1 for info in evaluation["per_field"].values()
            if info["status"] in ("exact", "close")
        )
        total += evaluation["total_fields"]
    return correct, total


def discover_predictions() -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    results = ROOT_DIR / "experiments" / "results"
    for path in sorted(results.glob("ocr_full*/predictions.jsonl")):
        found.append((f"OCR ({path.parent.name})", path))
    vlm_root = results / "ocr_vlm"
    for path in sorted(vlm_root.glob("*/predictions.jsonl")):
        found.append((f"VLM ({path.parent.name})", path))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score predictions under a GT patch.")
    parser.add_argument("--data", type=Path, default=ROOT_DIR / "experiments" / "ocr_data" / "makanan_ringan")
    parser.add_argument("--patch", type=Path, default=ROOT_DIR / "experiments" / "paper" / "gt_patch_makanan_ringan.json")
    args = parser.parse_args()

    if not args.data.is_dir():
        raise SystemExit(f"Folder data tidak ditemukan: {args.data}")
    patch = load_patch(args.patch)

    print(f"data        : {args.data}")
    print(f"patch       : {args.patch} ({'ada' if patch else 'kosong'})")
    print()
    print(f"{'pipeline':36s} {'orig':>12s} {'patched':>12s}")
    print("-" * 64)
    for label, path in discover_predictions():
        oc, ot = score_file(path, args.data, patch, patched=False)
        pc, pt = score_file(path, args.data, patch, patched=True)
        print(f"{label:36s} {oc:4d}/{ot:<3d} {oc/ot:6.1%}  {pc:4d}/{pt:<3d} {pc/pt:6.1%}")


if __name__ == "__main__":
    main()
