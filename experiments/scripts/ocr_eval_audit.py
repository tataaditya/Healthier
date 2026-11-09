"""Langkah 0 - audit dataset OCR label gizi.

Menghitung jumlah citra, resolusi, duplikat (md5 + dHash 8x8, ambang Hamming 5),
citra tidak terbaca, dan tumpang tindih antar folder. Output ditulis ke
``experiments/results/ocr_eval/`` (dataset_audit.csv + dataset_audit.md).

Jalankan:
    python experiments/scripts/ocr_eval_audit.py
"""
from __future__ import annotations

import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "results" / "ocr_eval"
OUT.mkdir(parents=True, exist_ok=True)

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Folder kandidat (semua berisi foto kemasan/tabel gizi)
CANDIDATE_DIRS = {
    "makanan_ringan": ROOT / "data" / "Dataset" / "Makanan Ringan",
    "frozen_food": ROOT / "data" / "Dataset" / "Frozen Food",
    "makanan_kaleng": ROOT / "data" / "Dataset" / "Makanan Kaleng",
    "minuman_sachet": ROOT / "data" / "Dataset" / "Minuman Sachet",
    "saos_perbumbuan": ROOT / "data" / "Dataset" / "Saos dan Perbumbuan" / "JPG_Output",
    # kandidat warisan (diduga subset) - dibandingkan, bukan dipakai otomatis
    "legacy_root_makanan_ringan": ROOT / "Makanan Ringan",
    "pilot_6": ROOT / "experiments" / "ocr_data" / "makanan_ringan",
}
PRIMARY = ["makanan_ringan", "frozen_food", "makanan_kaleng",
           "minuman_sachet", "saos_perbumbuan"]


def dhash(path: Path, size: int = 8) -> int:
    im = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    px = np.asarray(im, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def brand_guess(stem: str) -> str:
    s = stem.lower()
    m = re.match(r"^\d+[\s\.\)\-]+(.+)$", s)
    if m:
        s = m.group(1)
    s = re.sub(r"[_\-]+", " ", s).strip()
    if re.match(r"^\d{8}[_ ]?\d{6}", s):          # timestamp -> tidak ada merek
        return ""
    return s


def collect() -> list[dict]:
    rows = []
    for tag, folder in CANDIDATE_DIRS.items():
        if not folder.is_dir():
            print("WARN missing folder:", folder)
            continue
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() not in IMAGE_EXT or not p.is_file():
                continue
            rec = {"folder": tag, "file": str(p.relative_to(ROOT)),
                   "name": p.name, "stem": p.stem, "brand": brand_guess(p.stem)}
            try:
                with Image.open(p) as im:
                    rec["width"], rec["height"] = im.size
                rec["md5"] = md5(p)
                rec["dhash"] = dhash(p)
                rec["ok"] = 1
            except (UnidentifiedImageError, OSError) as exc:
                rec["width"] = rec["height"] = 0
                rec["md5"] = ""
                rec["dhash"] = 0
                rec["ok"] = 0
                print("UNREADABLE", p, exc)
            rows.append(rec)
    return rows


def dedup(rows: list[dict], threshold: int = 5) -> None:
    """Tandai duplikat persis (md5) dan mirip (dHash <= threshold)."""
    by_md5: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if r["ok"]:
            by_md5[r["md5"]].append(i)
    exact_dup = set()
    for group in by_md5.values():
        if len(group) > 1:
            exact_dup.update(group[1:])       # simpan yang pertama

    near_dup = set()
    for i, ri in enumerate(rows):
        if not ri["ok"]:
            continue
        for j in range(i + 1, len(rows)):
            rj = rows[j]
            if not rj["ok"] or j in near_dup:
                continue
            if hamming(ri["dhash"], rj["dhash"]) <= threshold:
                near_dup.add(j)

    for i, r in enumerate(rows):
        r["dup_md5"] = int(i in exact_dup)
        r["dup_near"] = int(i in near_dup and i not in exact_dup)
        r["unique"] = int(i not in exact_dup and i not in near_dup)


def main() -> None:
    rows = collect()
    dedup(rows)

    with (OUT / "dataset_audit.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def stat(key: str) -> dict:
        sub = [r for r in rows if r["folder"] == key]
        ok = [r for r in sub if r["ok"]]
        uniq = [r for r in ok if r["unique"]]
        widths = [r["width"] for r in ok]
        heights = [r["height"] for r in ok]
        return {
            "files": len(sub), "readable": len(ok),
            "exact_dup": sum(r["dup_md5"] for r in sub),
            "near_dup": sum(r["dup_near"] for r in sub),
            "unique": len(uniq),
            "min_w": min(widths) if widths else 0,
            "med_w": int(np.median(widths)) if widths else 0,
            "min_h": min(heights) if heights else 0,
            "med_h": int(np.median(heights)) if heights else 0,
        }

    lines = ["# Audit dataset OCR (Langkah 0)", ""]
    lines.append("## Ringkasan per folder")
    lines.append("")
    lines.append("| Folder | Files | Readable | Exact dup (md5) | Near dup (dHash<=5) | Unique | Resolusi min (WxH) | Median (WxH) |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |")
    for tag in CANDIDATE_DIRS:
        s = stat(tag)
        lines.append(f"| {tag} | {s['files']} | {s['readable']} | {s['exact_dup']} | "
                     f"{s['near_dup']} | {s['unique']} | {s['min_w']}x{s['min_h']} | "
                     f"{s['med_w']}x{s['med_h']} |")

    primary = [r for r in rows if r["folder"] in PRIMARY]
    p_ok = [r for r in primary if r["ok"]]
    p_uniq = [r for r in p_ok if r["unique"]]
    lines += ["", "## Gabungan folder PRIMARY (5 kategori)", ""]
    lines.append(f"- Total file: **{len(primary)}**")
    lines.append(f"- Terbaca: **{len(p_ok)}**; tidak terbaca: {len(primary) - len(p_ok)}")
    lines.append(f"- Duplikat persis (md5): **{sum(r['dup_md5'] for r in primary)}**")
    lines.append(f"- Duplikat mirip (dHash<=5): **{sum(r['dup_near'] for r in primary)}**")
    lines.append(f"- Unik: **{len(p_uniq)}**")

    # overlap legacy root vs primary (berdasar nama file)
    legacy = {r["name"] for r in rows if r["folder"] == "legacy_root_makanan_ringan"}
    primary_names = {r["name"] for r in primary}
    overlap = legacy & primary_names
    lines += ["", "## Tumpang tindih legacy `Makanan Ringan` (root) vs PRIMARY", ""]
    lines.append(f"- Nama file di legacy root: {len(legacy)}")
    lines.append(f"- Nama yang juga ada di PRIMARY: **{len(overlap)}** "
                 f"({100 * len(overlap) / max(len(legacy), 1):.1f}%)")

    # distribusi 'brand' (kasar) untuk kebutuhan stratifikasi
    brands = defaultdict(int)
    for r in p_uniq:
        brands[r["brand"] or "(tanpa merek di nama)"] += 1
    lines += ["", "## Tebakan grup/merek (kasar, dari nama file) - top 15", ""]
    for name, n in sorted(brands.items(), key=lambda kv: -kv[1])[:15]:
        lines.append(f"- {name}: {n}")

    (OUT / "dataset_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nwrote", OUT / "dataset_audit.csv")
    print("wrote", OUT / "dataset_audit.md")


if __name__ == "__main__":
    main()
