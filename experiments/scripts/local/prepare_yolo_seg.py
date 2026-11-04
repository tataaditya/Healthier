"""Build a reproducible YOLO segmentation dataset from semantic masks.

The script converts one or more semantic-mask datasets into the Ultralytics
YOLO-seg layout, remaps their native classes into the five Isi Piringku groups,
deduplicates exact visual duplicates, and creates a stratified split.

Expected source format
----------------------
``--source name=images_dir=masks_dir=classes.txt``

* ``images_dir``  : JPG/PNG images.
* ``masks_dir``   : one mask per image, same file stem (PNG/L-mode or palette).
* ``classes.txt`` : one native class per line; line index == mask pixel value
                    (minus ``--mask-offset``, default 1 when 0 is background).

Example
-------
    python experiments/scripts/prepare_yolo_seg.py \
        --source uec=raw/UECFoodPIXCOMPLETE/images=raw/UECFoodPIXCOMPLETE/masks=raw/UECFoodPIXCOMPLETE/classes.txt \
        --source fruit=raw/FruitSeg30/images=raw/FruitSeg30/masks=raw/FruitSeg30/classes.txt \
        --source foodseg=raw/FoodSeg103/images=raw/FoodSeg103/masks=raw/FoodSeg103/classes.txt \
        --out experiments/dataset \
        --test-source fruit \
        --copy-mode hardlink

If you have a manual class map, pass ``--class-map my_class_map.csv`` with
columns ``source,class_name,group``. Otherwise a keyword heuristic is used and
``class_map.generated.csv`` is written for review.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_SUFFIXES = {".png", ".bmp", ".tif", ".tiff"}

GROUPS: list[tuple[str, str]] = [
    ("makanan_pokok", "Staple Food"),
    ("lauk_pauk", "Side Dishes"),
    ("sayur", "Vegetables"),
    ("buah", "Fruits"),
    ("lainnya", "Others"),
]
GROUP_KEYS = [g for g, _ in GROUPS]
GROUP_DISPLAY = dict(GROUPS)
DISPLAY_TO_KEY = {v.lower(): k for k, v in GROUPS}
KEYWORDS: dict[str, tuple[str, ...]] = {
    "makanan_pokok": (
        "rice", "noodle", "pasta", "bread", "bun", "toast", "bagel", "cereal",
        "grain", "potato", "corn", "flour", "dough", "dumpling", "sushi",
        "sandwich", "pancake", "waffle", "porridge", "congee", "spaghetti",
        "macaroni", "vermicelli", "cassava", "taro", "yam", "quinoa", "oat",
    ),
    "lauk_pauk": (
        "meat", "beef", "pork", "chicken", "duck", "fish", "shrimp", "prawn",
        "seafood", "egg", "tofu", "tempeh", "bean", "nut", "cheese", "milk",
        "yogurt", "sausage", "ham", "bacon", "steak", "cutlet", "soup", "stew",
        "curry", "sauce", "gravy", "satay", "jerky", "meatball", "patty",
    ),
    "sayur": (
        "vegetable", "veggie", "broccoli", "spinach", "lettuce", "cabbage",
        "carrot", "tomato", "cucumber", "pepper", "onion", "garlic", "mushroom",
        "asparagus", "sprout", "eggplant", "zucchini", "pumpkin", "squash",
        "salad", "green", "kale", "chard", "okra", "radish", "beet", "celery",
        "cauliflower", "pea", "chili",
    ),
    "buah": (
        "fruit", "apple", "banana", "orange", "grape", "strawberry", "blueberry",
        "mango", "pineapple", "melon", "watermelon", "papaya", "pear", "peach",
        "plum", "cherry", "kiwi", "lemon", "lime", "avocado", "coconut",
        "passion", "dragon", "durian", "rambutan", "lychee", "guava", "berry",
    ),
}


@dataclass
class Source:
    name: str
    images: Path
    masks: Path
    classes_file: Path
    classes: list[str] = field(default_factory=list)
    mask_offset: int = 1


@dataclass
class Record:
    source: str
    key: str
    image: Path
    mask: Path
    width: int
    height: int
    primary_group: str
    image_hash: int
    group_pixels: dict[str, int] = field(default_factory=dict)
    class_ids_by_group: dict[str, set[int]] = field(default_factory=dict)
    split: str = ""


def parse_source(spec: str) -> Source:
    parts = spec.split("=")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--source harus berbentuk name=images_dir=masks_dir=classes.txt"
        )
    name, images, masks, classes = parts
    return Source(name=name, images=Path(images), masks=Path(masks), classes_file=Path(classes))


def read_classes(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"File kelas tidak ditemukan: {path}")
    classes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not classes:
        raise SystemExit(f"File kelas kosong: {path}")
    return classes


def normalize_group(raw: str) -> str:
    value = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if value in GROUP_KEYS:
        return value
    if value in DISPLAY_TO_KEY:
        return DISPLAY_TO_KEY[value]
    raise ValueError(f"Grup tidak dikenal: {raw}. Pilih: {', '.join(GROUP_KEYS)}")


def heuristic_group(class_name: str) -> str:
    name = class_name.lower()
    for group in ("makanan_pokok", "lauk_pauk", "sayur", "buah"):
        if any(keyword in name for keyword in KEYWORDS[group]):
            return group
    return "lainnya"


def load_class_map(
    paths: list[Path] | None,
    sources: Iterable[Source],
) -> tuple[dict[tuple[str, str], str], set[str]]:
    """Return ((source, class_name) -> group, explicitly mapped keys)."""
    mapping: dict[tuple[str, str], str] = {}
    explicit: set[str] = set()
    for path in paths or []:
        if not path.is_file():
            raise SystemExit(f"Class map tidak ditemukan: {path}")
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                source = (row.get("source") or "*").strip()
                class_name = (row.get("class_name") or "").strip()
                group = normalize_group(row.get("group") or "")
                if not class_name:
                    continue
                key = (source, class_name)
                mapping[key] = group
                if source != "*":
                    explicit.add(class_name)
    for src in sources:
        for class_name in src.classes:
            key = (src.name, class_name)
            if key in mapping:
                continue
            if (("*", class_name) in mapping):
                mapping[key] = mapping[("*", class_name)]
                continue
            mapping[key] = heuristic_group(class_name)
    return mapping, explicit


def file_map(
    folder: Path,
    suffixes: set[str],
    strip_suffix: str = "",
    recursive: bool = False,
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    if not folder.is_dir():
        raise SystemExit(f"Folder tidak ditemukan: {folder}")
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    for path in sorted(iterator):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        relative = path.relative_to(folder)
        key = str(relative.with_suffix("")).replace("\\", "/")
        if strip_suffix and key.endswith(strip_suffix):
            key = key[: -len(strip_suffix)]
        result.setdefault(key, path)
    return result


def read_mask(path: Path, size: tuple[int, int], channel: str = "auto") -> np.ndarray:
    with Image.open(path) as im:
        if im.mode == "P":
            arr = np.asarray(im)
        elif im.mode in {"L", "I", "I;16"}:
            arr = np.asarray(im)
        elif im.mode in {"RGB", "RGBA"}:
            # UEC-FoodPix encodes the class id in the red channel only.
            rgb = np.asarray(im)
            if channel == "red":
                arr = rgb[:, :, 0]
            elif channel == "green":
                arr = rgb[:, :, 1]
            elif channel == "blue":
                arr = rgb[:, :, 2]
            elif channel == "gray":
                arr = np.asarray(im.convert("L"))
            else:  # auto
                arr = rgb[:, :, 0]
        else:
            arr = np.asarray(im.convert("L"))
    arr = arr.astype(np.int32)
    width, height = size
    if arr.shape[1] != width or arr.shape[0] != height:
        arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_NEAREST)
    return arr


def dhash(path: Path, hash_size: int = 8) -> int | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = img[:, 1:] > img[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def polygons_for_group(
    mask: np.ndarray,
    class_ids: set[int],
    min_area: float,
    epsilon_ratio: float,
) -> list[list[tuple[float, float]]]:
    if not class_ids:
        return []
    binary = np.isin(mask, list(class_ids)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape[:2]
    result: list[list[tuple[float, float]]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        epsilon = epsilon_ratio * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        points = []
        for x, y in approx:
            points.append((min(max(float(x) / width, 0.0), 1.0), min(max(float(y) / height, 0.0), 1.0)))
        result.append(points)
    return result


def place_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    if mode == "symlink":
        try:
            os.symlink(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def split_records(
    records: list[Record],
    val_frac: float,
    test_frac: float,
    test_source: str | None,
    seed: int,
) -> None:
    rng = random.Random(seed)
    if test_source:
        for record in records:
            if record.source == test_source:
                record.split = "test"
        remaining = [r for r in records if r.split != "test"]
    else:
        remaining = list(records)

    buckets: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for record in remaining:
        buckets[(record.source, record.primary_group)].append(record)

    for bucket in buckets.values():
        rng.shuffle(bucket)
        n = len(bucket)
        n_test = 0 if test_source else int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        if n >= 3 and n_val == 0 and n_test == 0:
            n_val = 1
        if n >= 5 and n_test == 0 and not test_source:
            n_test = 1
        if n_val + n_test >= n:
            n_val = max(0, n - n_test - 1)
        for index, record in enumerate(bucket):
            if index < n_test:
                record.split = "test"
            elif index < n_test + n_val:
                record.split = "val"
            else:
                record.split = "train"


def write_yaml(path: Path, out_dir: Path) -> None:
    data = {
        "path": str(out_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: display for index, (_, display) in enumerate(GROUPS)},
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Konversi mask semantik menjadi dataset YOLO-seg + split reproducible.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", action="append", required=True, type=parse_source,
                        help="name=images_dir=masks_dir=classes.txt (boleh berulang)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--class-map", type=Path, action="append", default=[],
                        help="CSV source,class_name,group (boleh berulang)")
    parser.add_argument("--test-source", default=None,
                        help="Nama source yang seluruhnya dijadikan test independen")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-area", type=float, default=0.0005,
                        help="Area minimum kontur sebagai fraksi luas gambar (dipakai jika --min-area-px=0)")
    parser.add_argument("--min-area-px", type=float, default=100.0,
                        help="Area minimum kontur dalam piksel (default notebook: 100)")
    parser.add_argument("--epsilon", type=float, default=0.005,
                        help="approxPolyDP epsilon relatif terhadap keliling (default notebook: 0.005)")
    parser.add_argument("--mask-offset", type=int, default=1,
                        help="Offset indeks kelas pada mask (1 jika 0=background)")
    parser.add_argument("--mask-stem-suffix", default="",
                        help="Suffix nama mask yang harus dibuang, mis. _mask")
    parser.add_argument("--mask-channel", choices=["auto", "red", "green", "blue", "gray"],
                        default="auto",
                        help="Channel mask RGB; UEC-FoodPix memakai red")
    parser.add_argument("--recursive", action="store_true",
                        help="Cari citra/mask di subfolder (FruitSeg30 per kelas)")
    parser.add_argument("--folder-class-source", action="append", default=[],
                        help="Source yang kelasnya ditentukan nama folder (mis. FruitSeg30)")
    parser.add_argument("--copy-mode", choices=["copy", "hardlink", "symlink"], default="copy")
    parser.add_argument("--max-per-source", type=int, default=0,
                        help="Batasi jumlah citra per source (0 = semua)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sources = args.source
    folder_class_sources = set(args.folder_class_source)
    for source in sources:
        if source.name in folder_class_sources and not source.classes_file.is_file():
            source.classes = []
        else:
            source.classes = read_classes(source.classes_file)
        source.mask_offset = args.mask_offset
        print(f"[source] {source.name}: {len(source.classes)} kelas, images={source.images}, masks={source.masks}")

    class_map, explicit = load_class_map(args.class_map, sources)
    unmapped = sorted({c for s in sources for c in s.classes if (s.name, c) not in class_map})
    if unmapped:
        raise SystemExit(f"Kelas tanpa mapping: {unmapped[:10]}")

    records: list[Record] = []
    seen_hashes: dict[int, str] = {}
    warnings: list[str] = []
    per_source_counts: Counter[str] = Counter()
    group_ids_by_source: dict[str, dict[str, set[int]]] = {}
    folder_class_names: set[tuple[str, str]] = set()

    for source in sources:
        images = file_map(source.images, IMAGE_SUFFIXES, recursive=args.recursive)
        masks = file_map(
            source.masks, MASK_SUFFIXES,
            strip_suffix=args.mask_stem_suffix,
            recursive=args.recursive,
        )
        common = sorted(set(images) & set(masks))
        missing_masks = sorted(set(images) - set(masks))
        missing_images = sorted(set(masks) - set(images))
        if missing_masks:
            warnings.append(f"{source.name}: {len(missing_masks)} citra tanpa mask")
        if missing_images:
            warnings.append(f"{source.name}: {len(missing_images)} mask tanpa citra")
        if args.max_per_source:
            common = common[: args.max_per_source]
        print(f"[pair] {source.name}: {len(common)} pasangan citra-mask")

        group_ids: dict[str, set[int]] = defaultdict(set)
        for index, class_name in enumerate(source.classes):
            group = class_map[(source.name, class_name)]
            group_ids[group].add(index + source.mask_offset)
        group_ids_by_source[source.name] = group_ids

        for index, key in enumerate(common, 1):
            image_path = images[key]
            mask_path = masks[key]
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is None:
                warnings.append(f"{source.name}: gagal membaca {image_path.name}")
                continue
            height, width = img.shape[:2]
            mask = read_mask(mask_path, (width, height), channel=args.mask_channel)

            if source.name in folder_class_sources:
                parts = Path(key).parts
                folder_name = parts[0] if len(parts) > 1 else image_path.parent.name
                group = (
                    class_map.get((source.name, folder_name))
                    or class_map.get(("*", folder_name))
                    or heuristic_group(folder_name)
                )
                nonzero = {int(v) for v in np.unique(mask) if int(v) != 0}
                record_class_ids = {group: nonzero}
                group_pixels = {group: int(np.count_nonzero(mask))}
                primary = group
                folder_class_names.add((source.name, folder_name))
            else:
                record_class_ids = group_ids
                group_pixels = {
                    group: int(np.count_nonzero(np.isin(mask, list(ids))))
                    for group, ids in group_ids.items()
                }
                total_food = sum(group_pixels.values())
                primary = max(group_pixels, key=group_pixels.get) if total_food else "lainnya"

            image_hash = dhash(image_path)
            if image_hash is not None:
                if image_hash in seen_hashes:
                    warnings.append(
                        f"{source.name}: duplikat visual {image_path.name} "
                        f"(sama dengan {seen_hashes[image_hash]})"
                    )
                    continue
                seen_hashes[image_hash] = f"{source.name}/{image_path.name}"
            records.append(Record(
                source=source.name,
                key=key,
                image=image_path,
                mask=mask_path,
                width=width,
                height=height,
                primary_group=primary,
                image_hash=image_hash or 0,
                group_pixels=group_pixels,
                class_ids_by_group=record_class_ids,
            ))
            per_source_counts[source.name] += 1
            if index % 500 == 0:
                print(f"  ... {index}/{len(common)}")

    if not records:
        raise SystemExit("Tidak ada pasangan citra-mask yang valid.")

    split_records(records, args.val_frac, args.test_frac, args.test_source, args.seed)
    counts: Counter[tuple[str, str]] = Counter((r.split, r.primary_group) for r in records)
    present: Counter[tuple[str, str]] = Counter(
        (r.split, group)
        for r in records
        for group, pixels in r.group_pixels.items()
        if pixels > 0
    )
    print("\n[split]")
    for split in ("train", "val", "test"):
        total = sum(1 for r in records if r.split == split)
        detail = {g: present[(split, g)] for g in GROUP_KEYS}
        print(f"  {split:5s}: {total:6d}  (images containing group) {detail}")
        for group in GROUP_KEYS:
            if present[(split, group)] == 0:
                warnings.append(f"Split {split} tidak memiliki kelas {group}")

    if any(present[("train", group)] == 0 for group in GROUP_KEYS):
        print("\n[WARN] Ada kelas yang tidak muncul di train. Jangan hold out")
        print("       satu-satunya source untuk kelas tersebut; tambahkan")
        print("       subset source itu ke train atau pakai external test terpisah.")

    if args.dry_run:
        print("\n[dry-run] tidak menulis file.")
        return

    out_dir: Path = args.out
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    group_to_id = {group: index for index, (group, _) in enumerate(GROUPS)}
    instance_counts: dict[str, Counter[str]] = {
        split: Counter() for split in ("train", "val", "test")
    }
    manifest_rows = []
    for record in records:
        safe_stem = f"{record.source}__{record.key.replace('/', '__')}"
        image_dst = out_dir / "images" / record.split / f"{safe_stem}{record.image.suffix.lower()}"
        label_dst = out_dir / "labels" / record.split / f"{safe_stem}.txt"
        place_file(record.image, image_dst, args.copy_mode)
        mask = read_mask(record.mask, (record.width, record.height), channel=args.mask_channel)
        min_area_px = (
            args.min_area_px
            if args.min_area_px > 0
            else args.min_area * record.width * record.height
        )
        lines: list[str] = []
        for group in GROUP_KEYS:
            class_ids = record.class_ids_by_group.get(group, set())
            for points in polygons_for_group(mask, class_ids, min_area_px, args.epsilon):
                coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
                lines.append(f"{group_to_id[group]} {coords}")
                instance_counts[record.split][group] += 1
        label_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        manifest_rows.append({
            "source": record.source,
            "split": record.split,
            "image": str(image_dst.relative_to(out_dir)).replace("\\", "/"),
            "label": str(label_dst.relative_to(out_dir)).replace("\\", "/"),
            "width": record.width,
            "height": record.height,
            "primary_group": record.primary_group,
            "image_hash": f"{record.image_hash:016x}",
            **{f"px_{group}": record.group_pixels.get(group, 0) for group in GROUP_KEYS},
        })

    write_yaml(out_dir / "data.yaml", out_dir)
    with (out_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    map_rows = []
    seen_map_rows: set[tuple[str, str]] = set()
    for source in sources:
        for class_name in source.classes:
            key = (source.name, class_name)
            if key in seen_map_rows:
                continue
            seen_map_rows.add(key)
            map_rows.append({
                "source": source.name,
                "class_name": class_name,
                "group": class_map[key],
                "method": "manual" if class_name in explicit else "heuristic",
            })
    for source_name, class_name in sorted(folder_class_names):
        key = (source_name, class_name)
        if key in seen_map_rows:
            continue
        seen_map_rows.add(key)
        map_rows.append({
            "source": source_name,
            "class_name": class_name,
            "group": class_map.get(key) or class_map.get(("*", class_name)) or heuristic_group(class_name),
            "method": "manual" if key in class_map else "heuristic",
        })
    with (out_dir / "class_map.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["source", "class_name", "group", "method"])
        writer.writeheader()
        writer.writerows(map_rows)

    summary = {
        "sources": {s.name: per_source_counts[s.name] for s in sources},
        "records": len(records),
        "splits": {split: sum(1 for r in records if r.split == split) for split in ("train", "val", "test")},
        "per_group_primary": {
            split: {group: counts[(split, group)] for group in GROUP_KEYS}
            for split in ("train", "val", "test")
        },
        "per_group_present": {
            split: {group: present[(split, group)] for group in GROUP_KEYS}
            for split in ("train", "val", "test")
        },
        "instances": {
            split: {group: instance_counts[split][group] for group in GROUP_KEYS}
            for split in ("train", "val", "test")
        },
        "per_group": {
            split: {group: counts[(split, group)] for group in GROUP_KEYS}
            for split in ("train", "val", "test")
        },
        "test_source": args.test_source,
        "warnings": warnings,
        "config": {
            "val_frac": args.val_frac,
            "test_frac": args.test_frac,
            "seed": args.seed,
            "min_area": args.min_area,
            "min_area_px": args.min_area_px,
            "epsilon": args.epsilon,
            "mask_offset": args.mask_offset,
            "mask_channel": args.mask_channel,
            "recursive": args.recursive,
        },
    }
    (out_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[output] {out_dir}")
    print(f"  data.yaml, manifest.csv, class_map.csv, dataset_summary.json")
    if warnings:
        print(f"  warnings: {len(warnings)} (lihat dataset_summary.json)")
    print("  Selesai.")


if __name__ == "__main__":
    main()
