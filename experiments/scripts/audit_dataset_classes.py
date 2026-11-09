"""Audit class remapping + class imbalance for the segmentation dataset.

Uses only artefacts that already exist (``dataset_summary.json`` and
``foodseg_class_map.csv``); it does not touch images or models. Produces the
evidence needed for reviewer points A3/B1:

  * how every source class was remapped into the five Isi Piringku groups,
  * how many source classes collapse into ``Others`` (the "catch-all" risk),
  * instance- and image-level imbalance ratios per split, and
  * the coverage gaps of the external test sets (e.g. UEC has no Fruits).

Run:
    python experiments/scripts/audit_dataset_classes.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUM = ROOT / "experiments" / "dataset" / "revision" / "dataset_summary.json"
MAP = ROOT / "experiments" / "configs" / "foodseg_class_map.csv"
OUT = ROOT / "experiments" / "results" / "segmentation"

GROUPS = ["Staple Food", "Side Dishes", "Vegetables", "Fruits", "Others"]
GROUP_ID = {g: i for i, g in enumerate(GROUPS)}
# the manual FoodSeg map uses Indonesian group names
GROUP_NAME_ID = {
    "makanan_pokok": "Staple Food",
    "lauk_pauk": "Side Dishes",
    "sayur": "Vegetables",
    "buah": "Fruits",
    "lainnya": "Others",
}


def load_mapping() -> list[dict]:
    with MAP.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def mapping_audit(rows: list[dict], summary: dict) -> dict:
    by_source: dict[str, Counter] = defaultdict(Counter)
    examples: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        g = GROUP_NAME_ID.get(r["group"])
        if g is None:
            continue
        by_source[r["source"]][g] += 1
        if len(examples[g]) < 8:
            examples[g].append(r["class_name"])

    # per-source counts straight from the frozen mapping used for training
    id_counts: dict[str, dict[str, int]] = {}
    for src, mp in summary.get("mapping", {}).items():
        c = Counter(GROUPS[int(v)] for v in mp.values() if 0 <= int(v) < len(GROUPS))
        id_counts[src] = dict(c)

    return {
        "n_source_classes": len(rows),
        "per_source_group_counts": {s: dict(c) for s, c in by_source.items()},
        "frozen_mapping_group_counts": id_counts,
        "examples_per_group": dict(examples),
        "n_classes_foodseg_named": len(rows),
        "n_classes_uec_mapped": len(summary.get("mapping", {}).get("uec", {})),
        # every FoodSeg class not explicitly mapped would silently become Others
        "others_share_foodseg": (
            by_source["foodseg"]["Others"] / sum(by_source["foodseg"].values())
            if by_source["foodseg"] else None
        ),
    }


def imbalance(summary: dict) -> dict:
    out: dict[str, dict] = {}
    for split in ("main", "external"):
        if split == "main":
            blocks = {"train": "train_instances", "val": "val_instances"}
            img_blocks = {"train": "train_images_with_class", "val": "val_images_with_class"}
            for label, key in blocks.items():
                inst = summary["main"][key]
                imgs = summary["main"][img_blocks[label]]
                out[f"main_{label}"] = _ratios(inst, imgs)
        else:
            for src, blob in summary["external"].items():
                out[src] = _ratios(blob["instances"], blob["images_with_class"])
    return out


def _ratios(inst: dict[str, int], imgs: dict[str, int]) -> dict:
    inst_vals = [v for v in inst.values() if v > 0]
    img_vals = [v for v in imgs.values() if v > 0]
    total_i = sum(inst.values())
    total_m = sum(imgs.values())
    return {
        "instances": inst,
        "instance_share": {k: round(v / total_i, 4) for k, v in inst.items()} if total_i else {},
        "images_with_class": imgs,
        "image_share": {k: round(v / total_m, 4) for k, v in imgs.items()} if total_m else {},
        "instance_imbalance_max_over_min": round(max(inst_vals) / min(inst_vals), 2) if inst_vals else None,
        "image_imbalance_max_over_min": round(max(img_vals) / min(img_vals), 2) if img_vals else None,
        "classes_absent": [k for k, v in inst.items() if v == 0],
    }


def write_md(audit: dict, imb: dict) -> None:
    lines = [
        "# Audit remapping kelas & class imbalance",
        "",
        "> Dihasilkan oleh `experiments/scripts/audit_dataset_classes.py` dari "
        "`dataset_summary.json` + `foodseg_class_map.csv`. Tanpa menjalankan model.",
        "",
        "## 1. Remapping kelas sumber → 5 grup Isi Piringku",
        "",
        f"Total kelas sumber yang dipetakan: **{audit['n_source_classes']}**.",
        "",
        "| Grup target | Contoh kelas sumber |",
        "| :--- | :--- |",
    ]
    for g in GROUPS:
        ex = ", ".join(audit["examples_per_group"].get(g, []))
        lines.append(f"| {g} | {ex} |")

    lines += ["", "Jumlah kelas sumber per grup (peta manual FoodSeg):", "",
              "| Sumber | " + " | ".join(GROUPS) + " |",
              "| :--- | " + " | ".join(["---:"] * len(GROUPS)) + " |"]
    for src, c in audit["per_source_group_counts"].items():
        lines.append(f"| {src} | " + " | ".join(str(c.get(g, 0)) for g in GROUPS) + " |")
    lines += ["", "Jumlah kelas sumber per grup (pemetaan beku yang dipakai training):", "",
              "| Sumber | " + " | ".join(GROUPS) + " |",
              "| :--- | " + " | ".join(["---:"] * len(GROUPS)) + " |"]
    for src, c in audit["frozen_mapping_group_counts"].items():
        lines.append(f"| {src} | " + " | ".join(str(c.get(g, 0)) for g in GROUPS) + " |")

    share = audit["others_share_foodseg"]
    share_txt = f"{share:.1%}" if share is not None else "n/a"
    lines += [
        "",
        f"**Risiko catch-all:** {share_txt} kelas FoodSeg dipetakan ke `Others` "
        f"({audit['n_classes_foodseg_named']} kelas bernama; {audit['n_classes_uec_mapped']} "
        "ID UEC dipetakan lewat tabel beku). `Others` karena itu bukan kategori semantik "
        "tunggal, melainkan gabungan kelas yang heterogen — ini menjelaskan AP-nya yang "
        "rendah dan harus dilaporkan sebagai keterbatasan, bukan disembunyikan.",
        "",
        "## 2. Class imbalance per split",
        "",
    ]
    for name, r in imb.items():
        lines += [f"### {name}", "",
                  "| Kelas | Instance | Share | Citra mengandung | Share |",
                  "| :--- | ---: | ---: | ---: | ---: |"]
        for g in GROUPS:
            iv = r["instances"].get(g, 0)
            mv = r["images_with_class"].get(g, 0)
            lines.append(f"| {g} | {iv} | {r['instance_share'].get(g, 0):.1%} | "
                         f"{mv} | {r['image_share'].get(g, 0):.1%} |")
        lines.append("")
        if r["instance_imbalance_max_over_min"]:
            lines.append(f"- Rasio imbalance instance (max/min): **{r['instance_imbalance_max_over_min']}x**")
        if r["image_imbalance_max_over_min"]:
            lines.append(f"- Rasio imbalance citra (max/min): **{r['image_imbalance_max_over_min']}x**")
        if r["classes_absent"]:
            lines.append(f"- Kelas tidak ada di split ini: {', '.join(r['classes_absent'])}")
        lines.append("")

    lines += [
        "## 3. Implikasi untuk naskah",
        "",
        "1. Pelatihan utama memakai **plain Ultralytics training** (tanpa oversampling, "
        "tanpa weighted loss). Karena itu imbalance harus dinyatakan apa adanya dan "
        "dibahas bersama per-class AP, bukan diklaim sudah ditangani.",
        "2. `Fruits` adalah kelas minoritas konsisten (instance maupun citra) di semua "
        "split internal; `Vegetables` mayoritas. Korelasi frekuensi kelas vs AP perlu "
        "dilaporkan (lihat `table_segmentation_per_class.md`).",
        "3. UEC tidak memuat `Fruits` dan FruitSeg hanya memuat `Fruits`, sehingga "
        "macro-average eksternal tidak sebanding antar-sumber dan tidak boleh "
        "dibandingkan langsung tanpa catatan cakupan.",
        "",
    ]
    (OUT / "dataset_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary = json.loads(SUM.read_text(encoding="utf-8"))
    rows = load_mapping()
    audit = mapping_audit(rows, summary)
    imb = imbalance(summary)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dataset_audit.json").write_text(
        json.dumps({"mapping": audit, "imbalance": imb}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    write_md(audit, imb)
    print("wrote", OUT / "dataset_audit.md")
    for k, v in imb.items():
        print(f"  {k:16s} instance imbalance {v['instance_imbalance_max_over_min']}x "
              f"image imbalance {v['image_imbalance_max_over_min']}x")


if __name__ == "__main__":
    main()
