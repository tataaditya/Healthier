"""Langkah 1 - locked dev/test split for the OCR evaluation.

Reads the five ``data/Dataset`` folders (the agreed OCR pool), removes exact
(md5) and near (dHash<=5) duplicates, assigns each kept image a ``group_id``
so that the same product/brand cannot appear in dev and test at once, then
splits once with ``seed=42`` and locks the result with a SHA-256 hash.

Outputs (``experiments/results/ocr_eval/``):
    split_manifest.csv     one row per file (split in {test, dev, dropped})
    split_manifest.sha256  SHA-256 of the manifest bytes (the "lock")
    split_lock.json        seed / ratio / counts / hash metadata

The six images of the old pilot (``experiments/ocr_data/makanan_ringan``) are
forced into ``dev`` so they can never contaminate the test set.

Run:
    python experiments/scripts/ocr_eval_split.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "results" / "ocr_eval"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_RATIO = 0.30          # -> test comfortably above the 100-image minimum
NEAR_DUP_HAMMING = 5       # dHash distance treated as a duplicate
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# The five agreed folders. ``tag`` is used as the folder column.
FOLDERS = {
    "makanan_ringan": ROOT / "data" / "Dataset" / "Makanan Ringan",
    "frozen_food": ROOT / "data" / "Dataset" / "Frozen Food",
    "makanan_kaleng": ROOT / "data" / "Dataset" / "Makanan Kaleng",
    "minuman_sachet": ROOT / "data" / "Dataset" / "Minuman Sachet",
    "saos_perbumbuan": ROOT / "data" / "Dataset" / "Saos dan Perbumbuan" / "JPG_Output",
}
PILOT_DIR = ROOT / "experiments" / "ocr_data" / "makanan_ringan"

FIELDS = ["file", "folder", "split", "group_id", "md5", "dhash",
          "width", "height", "brand", "drop_reason", "forced_dev"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def dhash(path: Path, size: int = 8) -> int:
    im = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    px = np.asarray(im, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def brand_guess(stem: str) -> str:
    """Very rough product/brand key from a filename (or '' when unknown)."""
    s = stem.lower()
    m = re.match(r"^\d+\s*[\.\)\-]?\s*(.+)$", s)          # "12-Sushi crab"
    if m:
        s = m.group(1)
    s = re.sub(r"[_\-]+", " ", s).strip()
    s = re.sub(r"\bcopy( \d+)?\b", "", s).strip()
    if re.match(r"^\d{8}[_ ]?\d{6}", s) or re.fullmatch(r"\d+", s):
        return ""                                          # camera timestamp
    s = re.sub(r"\s+", " ", s).strip()
    return s


# --------------------------------------------------------------------------- #
# collect + dedup
# --------------------------------------------------------------------------- #
def collect() -> list[dict]:
    rows: list[dict] = []
    for tag, folder in FOLDERS.items():
        if not folder.is_dir():
            raise SystemExit(f"missing folder: {folder}")
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() not in IMAGE_EXT or not p.is_file():
                continue
            rec = {"folder": tag, "path": p,
                   "file": str(p.relative_to(ROOT)).replace("\\", "/"),
                   "brand": brand_guess(p.stem)}
            try:
                with Image.open(p) as im:
                    rec["width"], rec["height"] = im.size
                rec["md5"] = file_md5(p)
                rec["dhash"] = dhash(p)
                rec["ok"] = True
            except (UnidentifiedImageError, OSError):
                rec.update(width=0, height=0, md5="", dhash=0, ok=False)
            rows.append(rec)
    return rows


def dedup(rows: list[dict]) -> None:
    """Union-find clusters (exact md5 OR dHash<=threshold); keep first member."""
    parent = list(range(len(rows)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_md5: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if r["ok"] and r["md5"]:
            by_md5[r["md5"]].append(i)
    for group in by_md5.values():
        for j in group[1:]:
            union(group[0], j)

    ok_idx = [i for i, r in enumerate(rows) if r["ok"]]
    for a in range(len(ok_idx)):
        i = ok_idx[a]
        for b in range(a + 1, len(ok_idx)):
            j = ok_idx[b]
            if hamming(rows[i]["dhash"], rows[j]["dhash"]) <= NEAR_DUP_HAMMING:
                union(i, j)

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(len(rows)):
        clusters[find(i)].append(i)

    for members in clusters.values():
        members.sort(key=lambda i: ((not rows[i]["ok"]), rows[i]["file"]))
        keep = members[0]
        for m in members:
            if m == keep:
                rows[m]["split"] = None
                rows[m]["drop_reason"] = ""
            else:
                rows[m]["split"] = "dropped"
                rows[m]["drop_reason"] = ("exact_md5"
                                          if rows[m]["md5"] and rows[m]["md5"] == rows[keep]["md5"]
                                          else "near_dhash")
                rows[m]["group_id"] = ""
                rows[m]["forced_dev"] = 0


# --------------------------------------------------------------------------- #
# grouping + split
# --------------------------------------------------------------------------- #
def assign_groups(rows: list[dict]) -> None:
    """group_id = 'brand:<key>' when a brand is known, else unique per file."""
    brand_to_gid: dict[str, str] = {}
    n = 0
    for r in rows:
        if r["split"] == "dropped":
            continue
        brand = r["brand"]
        if brand:
            key = brand
            if key not in brand_to_gid:
                n += 1
                brand_to_gid[key] = f"brand-{n:03d}"
            r["group_id"] = brand_to_gid[key]
        else:
            n += 1
            r["group_id"] = f"file-{n:04d}"


def split(rows: list[dict], pilot: set[str]) -> None:
    rng = random.Random(SEED)

    # groups -> members, and each group's home folder (first member's folder)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["split"] != "dropped":
            groups[r["group_id"]].append(r)
    home = {gid: min(m, key=lambda r: r["file"])["folder"]
            for gid, m in groups.items()}

    # force every group touched by a pilot image into dev
    pilot_groups = {r["group_id"] for r in rows
                    if r["path"].name in pilot and r["split"] != "dropped"}

    by_folder: dict[str, list[str]] = defaultdict(list)
    for gid in groups:
        by_folder[home[gid]].append(gid)

    test_groups: set[str] = set()
    for folder, gids in by_folder.items():
        total = sum(len(groups[g]) for g in gids)
        target = round(total * TEST_RATIO)
        order = sorted(gids)
        rng.shuffle(order)
        picked = 0
        for gid in order:
            if gid in pilot_groups:
                continue
            if picked >= target:
                break
            test_groups.add(gid)
            picked += len(groups[gid])

    for r in rows:
        if r["split"] == "dropped":
            r["forced_dev"] = 0
            continue
        gid = r["group_id"]
        forced = 1 if (gid in pilot_groups or r["path"].name in pilot) else 0
        r["forced_dev"] = forced
        r["split"] = "dev" if (forced or gid not in test_groups) else "test"


def main() -> None:
    rows = collect()
    dedup(rows)
    assign_groups(rows)
    pilot = {p.name for p in PILOT_DIR.iterdir()
             if p.is_file() and p.suffix.lower() in IMAGE_EXT}
    split(rows, pilot)

    rows.sort(key=lambda r: (r["folder"], r["file"]))
    with (OUT / "split_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})

    data = (OUT / "split_manifest.csv").read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    (OUT / "split_manifest.sha256").write_text(f"{digest}  split_manifest.csv\n",
                                               encoding="utf-8")

    def count(split_name: str) -> int:
        return sum(1 for r in rows if r["split"] == split_name)

    per_folder = {tag: {"test": 0, "dev": 0, "dropped": 0} for tag in FOLDERS}
    for r in rows:
        per_folder[r["folder"]][r["split"]] += 1

    lock = {
        "seed": SEED,
        "test_ratio": TEST_RATIO,
        "near_dup_hamming": NEAR_DUP_HAMMING,
        "sha256": digest,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_total_files": len(rows),
        "n_dropped": count("dropped"),
        "n_kept": count("test") + count("dev"),
        "n_test": count("test"),
        "n_dev": count("dev"),
        "n_test_groups": len({r["group_id"] for r in rows if r["split"] == "test"}),
        "n_dev_groups": len({r["group_id"] for r in rows if r["split"] == "dev"}),
        "pilot_forced_dev": sorted(pilot),
        "per_folder": per_folder,
    }
    (OUT / "split_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")

    print(f"total files     : {lock['n_total_files']}")
    print(f"dropped (dup)   : {lock['n_dropped']}")
    print(f"kept            : {lock['n_kept']}")
    print(f"  test          : {lock['n_test']}  (groups: {lock['n_test_groups']})")
    print(f"  dev           : {lock['n_dev']}  (groups: {lock['n_dev_groups']})")
    print(f"sha256          : {digest}")
    print("\nper folder (test/dev/dropped):")
    for tag, c in per_folder.items():
        print(f"  {tag:20s} {c['test']:>4}/{c['dev']:>4}/{c['dropped']:>3}")
    print("\nwrote", OUT / "split_manifest.csv")
    print("wrote", OUT / "split_lock.json")


if __name__ == "__main__":
    main()
