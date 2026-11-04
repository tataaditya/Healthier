"""Extract the exact class mappings from the training notebook.

The notebook ``training_model.ipynb`` contains the manual mappings used for the
reported experiments. This script converts those mappings into CSVs consumable
by ``prepare_yolo_seg.py`` so the revision does not silently use a different
heuristic.

Example:
    python experiments/scripts/make_class_map.py \
        --uec-category raw/UECFOODPIXCOMPLETE/.../category.txt \
        --out-dir experiments/configs
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path


def extract_assignments(notebook: Path, names: set[str]) -> dict[str, object]:
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    found: dict[str, object] = {}
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                try:
                    found[target.id] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    continue
    return found


def read_category(path: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        result[int(parts[0])] = parts[1].strip()
    return result


def write_csv(path: Path, source: str, mapping: dict[int, int], names: dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = []
    rows = []
    for raw_id, group in sorted(mapping.items()):
        class_name = names.get(raw_id)
        if not class_name:
            missing.append(raw_id)
            continue
        rows.append({
            "source": source,
            "class_name": class_name,
            "group": ["makanan_pokok", "lauk_pauk", "sayur", "buah", "lainnya"][group],
        })
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["source", "class_name", "group"])
        writer.writeheader()
        writer.writerows(rows)
    if missing:
        print(f"[warn] {source}: id tanpa nama -> {missing}")


def write_classes(path: Path, names: dict[int, str], start: int = 1) -> None:
    path.write_text("\n".join(names[i] for i in sorted(names) if i >= start) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "notebooks" / "training_model.ipynb",
    )
    parser.add_argument("--uec-category", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/configs"))
    args = parser.parse_args()

    if not args.notebook.is_file():
        raise SystemExit(f"Notebook tidak ditemukan: {args.notebook}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    values = extract_assignments(
        args.notebook,
        {"UECFOOD_TO_ISIPIRINGKU", "FOODSEG_TO_ISIPIRINGKU", "FOODSEG103_NAMES", "CLASS_NAMES"},
    )
    for key in ("UECFOOD_TO_ISIPIRINGKU", "FOODSEG_TO_ISIPIRINGKU", "FOODSEG103_NAMES"):
        if key not in values:
            raise SystemExit(f"Assignment {key} tidak ditemukan di notebook.")

    if args.uec_category:
        uec_names = read_category(args.uec_category)
        if not uec_names:
            raise SystemExit(f"category.txt kosong/tidak valid: {args.uec_category}")
        write_csv(
            args.out_dir / "uecfood_class_map.csv",
            "uec",
            values["UECFOOD_TO_ISIPIRINGKU"],
            uec_names,
        )
        write_classes(args.out_dir / "uecfood_classes.txt", uec_names, start=1)
        print(f"[ok] UECFood: {len(uec_names)} kelas -> uecfood_class_map.csv")

    food_names = {int(k): str(v) for k, v in values["FOODSEG103_NAMES"].items()}
    write_csv(
        args.out_dir / "foodseg_class_map.csv",
        "foodseg",
        values["FOODSEG_TO_ISIPIRINGKU"],
        food_names,
    )
    write_classes(args.out_dir / "foodseg_classes.txt", food_names, start=1)
    print(f"[ok] FoodSeg103: {len(food_names) - 1} kelas -> foodseg_class_map.csv")

    print(f"\nOutput: {args.out_dir.resolve()}")
    print("Gunakan dengan prepare_yolo_seg.py --class-map ...")


if __name__ == "__main__":
    main()
