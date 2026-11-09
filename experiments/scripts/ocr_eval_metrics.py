"""Field-level metrics for the Healthier / JRI 2026 OCR evaluation.

Joins the prediction files produced by ``ocr_eval_harness.py`` with the
human-verified ground truth (or the draft annotations while verification is
pending — the source is recorded in every output as ``gt_source``).

Implements exactly the frozen definitions in ``metrics.md``:
  * only fields present in the GT (not ``field_absent_on_label``) enter the
    denominator;
  * primary metric = Exact (tolerance 0); secondary = Exact@5%;
  * cluster bootstrap over images (B=10000, seed=42);
  * paired bootstrap + McNemar for the main configurations.

Usage:
    python experiments/scripts/ocr_eval_metrics.py
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "results" / "ocr_eval"
ANN = OUT / "annotation"
PRED = OUT / "predictions"

FIELDS = [
    "kalori", "lemak_total", "lemak_jenuh", "lemak_trans", "kolesterol",
    "natrium", "karbohidrat", "serat", "gula", "protein",
]
KEY_FIELDS = ["kalori", "lemak_total", "karbohidrat", "protein", "natrium", "gula"]
MAIN_ORDER = ["A_tesseract", "B_easyocr", "C_paddle_raw", "D_baseline_parser", "E_full_pipeline"]
CUM_ORDER = [
    "cum_raw", "cum_plus_deskew", "cum_plus_deglare", "cum_plus_clahe", "cum_plus_roi",
    "cum_plus_semantic", "cum_plus_fuzzy", "cum_plus_fallback", "cum_plus_validation",
]
LOO_ORDER = [
    "loo_full_minus_deskew", "loo_full_minus_deglare", "loo_full_minus_clahe",
    "loo_full_minus_roi", "loo_full_minus_semantic", "loo_full_minus_fuzzy",
    "loo_full_minus_fallback", "loo_full_minus_validation",
]
B = 10000
SEED = 42


def load_gt() -> tuple[dict[str, dict], str]:
    vpath = ANN / "ground_truth_verified.json"
    if vpath.is_file():
        data = json.loads(vpath.read_text(encoding="utf-8"))
        src = "human_verified"
        if isinstance(data, dict) and "records" in data:
            data = data["records"]
    else:
        drafts = json.loads((ANN / "drafts.json").read_text(encoding="utf-8"))
        data, src = [], "draft_unverified"
        for r in drafts:
            d = r.get("draft") or {}
            data.append({
                "file": r["file"].replace("\\", "/"),
                "verified": bool(r.get("verified")),
                "nutrisi": d.get("nutrisi") or {},
                "field_absent_on_label": d.get("field_absent_on_label") or {},
                "takaran_saji_g": d.get("takaran_saji_g"),
                "note": d.get("note", ""),
            })
    by_file = {}
    for d in data:
        by_file[d["file"].replace("\\", "/")] = d
    return by_file, src


def load_predictions(name: str) -> dict[str, dict]:
    p = PRED / f"{name}.jsonl"
    if not p.is_file():
        return {}
    out = {}
    for line in p.open(encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            out[r["file"].replace("\\", "/")] = r
    return out


def _within(pred, gt, tol: float) -> bool:
    if pred is None or gt is None:
        return False
    if float(gt) == 0:
        return float(pred) == 0.0
    return abs(float(pred) - float(gt)) / max(abs(float(gt)), 0.001) <= tol


def evaluate(name: str, gt: dict[str, dict]) -> dict:
    preds = load_predictions(name)
    per_image: list[dict] = []
    errors: list[dict] = []
    field_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    durations: list[float] = []
    for f, g in gt.items():
        if f not in preds:
            continue
        ab = g.get("field_absent_on_label") or {}
        nut = g.get("nutrisi") or {}
        pred = preds[f].get("nutrisi") or {}
        n = exact = close = wrong = missing = 0
        key_total = key_exact = 0
        present_fields = 0
        unreadable = 0
        for field in FIELDS:
            if ab.get(field):
                continue
            gv = nut.get(field)
            if gv is None:
                # Field is printed on the label but the value is not legible in
                # the photo. Excluded from every denominator, but counted so the
                # report can state how much of the label was unreadable.
                unreadable += 1
                field_counts[field]["unreadable"] += 1
                continue
            present_fields += 1
            pv = pred.get(field)
            if pv is None:
                status = "missing"; missing += 1
            elif abs(float(pv) - float(gv)) <= 1e-6:
                status = "exact"; exact += 1
            elif _within(pv, gv, 0.05):
                status = "close"; close += 1
            else:
                status = "wrong"; wrong += 1
            n += 1
            field_counts[field][status] += 1
            if field in KEY_FIELDS:
                key_total += 1
                if status == "exact":
                    key_exact += 1
            if status in ("wrong", "missing"):
                gvf = float(gv)
                if pv is None:
                    errors.append({"file": f, "config": name, "field": field, "gt": gvf,
                                   "pred": "", "rel_error": "", "kind": "missing",
                                   "flags": ";".join(map(str, preds[f].get("flags", [])))})
                else:
                    rel = abs(float(pv) - gvf) / max(abs(gvf), 0.001)
                    errors.append({"file": f, "config": name, "field": field, "gt": gvf,
                                   "pred": pv, "rel_error": round(rel, 4),
                                   "kind": classify_error(gvf, float(pv)),
                                   "flags": ";".join(map(str, preds[f].get("flags", [])))})
        d = preds[f].get("durasi_detik")
        if isinstance(d, (int, float)):
            durations.append(float(d))
        per_image.append({
            "file": f, "n": n, "exact": exact, "close": close, "wrong": wrong,
            "missing": missing, "present_fields": present_fields,
            "unreadable": unreadable,
            "all_key_exact": int(key_total > 0 and key_exact == key_total),
            "table_visible": int(present_fields > 0),
        })
    tot_n = sum(r["n"] for r in per_image)
    tot_exact = sum(r["exact"] for r in per_image)
    tot_close = sum(r["close"] for r in per_image)
    tot_wrong = sum(r["wrong"] for r in per_image)
    tot_missing = sum(r["missing"] for r in per_image)
    visible = [r for r in per_image if r["table_visible"]]
    return {
        "config": name,
        "n_images": len(per_image),
        "n_images_table_visible": len(visible),
        "n_fields": tot_n,
        "exact": tot_exact, "close": tot_close, "wrong": tot_wrong, "missing": tot_missing,
        "n_unreadable": sum(r["unreadable"] for r in per_image),
        "field_accuracy": tot_exact / tot_n if tot_n else 0.0,
        "exact_at_5pct": (tot_exact + tot_close) / tot_n if tot_n else 0.0,
        "image_full_match": (sum(r["all_key_exact"] for r in visible) / len(visible)) if visible else 0.0,
        "median_dur": round(sorted(durations)[len(durations) // 2], 3) if durations else None,
        "p95_dur": round(sorted(durations)[max(0, int(0.95 * (len(durations) - 1)))], 3) if durations else None,
        "per_image": per_image,
        "errors": errors,
        "field_counts": {k: dict(v) for k, v in field_counts.items()},
        "n_dur": len(durations),
    }


def classify_error(gt: float, pred: float) -> str:
    if gt == 0:
        return "nonzero_for_zero"
    ratio = pred / gt if gt else 0
    for f in (10, 100, 1000):
        if abs(ratio - f) < 0.05 or abs(ratio - 1 / f) < 0.05:
            return f"scale_{f}x"
    if abs(gt - pred) <= 1:
        return "small_offset"
    return "digit_confusion"


def bootstrap_ci(per_image: list[dict]) -> tuple[float, float]:
    rows = [(r["exact"], r["n"]) for r in per_image if r["n"] > 0]
    if not rows:
        return 0.0, 0.0
    rng = random.Random(SEED)
    n = len(rows)
    vals = []
    for _ in range(B):
        c = t = 0
        for _ in range(n):
            x, y = rows[rng.randrange(n)]
            c += x; t += y
        vals.append(c / t if t else 0.0)
    vals.sort()
    return vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]


def paired_bootstrap(a: dict, b: dict) -> tuple[float, float, float]:
    ra = {r["file"]: r for r in a["per_image"]}
    rb = {r["file"]: r for r in b["per_image"]}
    files = [f for f in ra if f in rb and ra[f]["n"] > 0 and rb[f]["n"] > 0]
    if not files:
        return 0.0, 0.0, 0.0
    diff = a["field_accuracy"] - b["field_accuracy"]
    rng = random.Random(SEED)
    vals = []
    n = len(files)
    for _ in range(B):
        ca = cb = t = 0
        for _ in range(n):
            f = files[rng.randrange(n)]
            ca += ra[f]["exact"]; cb += rb[f]["exact"]; t += ra[f]["n"]
        vals.append((ca - cb) / t if t else 0.0)
    vals.sort()
    return diff, vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]


def mcnemar(a: dict, b: dict) -> tuple[int, int, float]:
    ra = {r["file"]: r for r in a["per_image"]}
    rb = {r["file"]: r for r in b["per_image"]}
    # need per-field status -> rebuild from field_counts is not enough; use per_image+errors
    # Reconstruct per (file, field) exact flags from predictions.
    pa, pb = load_predictions(a["config"]), load_predictions(b["config"])
    n01 = n10 = 0
    for f, g in GT_CACHE.items():
        if f not in pa or f not in pb:
            continue
        ab = g.get("field_absent_on_label") or {}
        nut = g.get("nutrisi") or {}
        for field in FIELDS:
            if ab.get(field) or nut.get(field) is None:
                continue
            gv = float(nut[field])
            va = (pa[f].get("nutrisi") or {}).get(field)
            vb = (pb[f].get("nutrisi") or {}).get(field)
            ca = va is not None and abs(float(va) - gv) <= 1e-6
            cb = vb is not None and abs(float(vb) - gv) <= 1e-6
            if ca and not cb:
                n10 += 1
            elif cb and not ca:
                n01 += 1
    nd = n01 + n10
    if nd == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = sum(math.comb(nd, i) for i in range(0, k + 1)) / (2 ** nd) * 2
    return n01, n10, min(1.0, p)


GT_CACHE: dict[str, dict] = {}


def fmt_pct(x) -> str:
    return f"{x:.2%}" if isinstance(x, (int, float)) else ""


def write_main(results: dict[str, dict], src: str) -> None:
    rows = []
    for name in MAIN_ORDER:
        if name not in results:
            continue
        r = results[name]
        lo, hi = bootstrap_ci(r["per_image"])
        rows.append({
            "config": name, "n_images": r["n_images"],
            "n_images_table_visible": r["n_images_table_visible"], "n_fields": r["n_fields"],
            "exact": r["exact"], "close": r["close"], "wrong": r["wrong"], "missing": r["missing"],
            "n_unreadable": r.get("n_unreadable", 0),
            "field_accuracy": round(r["field_accuracy"], 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "exact_at_5pct": round(r["exact_at_5pct"], 4),
            "image_full_match": round(r["image_full_match"], 4),
            "median_dur_s": r["median_dur"], "p95_dur_s": r["p95_dur"],
            "gt_source": src,
        })
    _write_csv(OUT / "results_main.csv", rows)
    lines = [f"# Hasil utama (n={rows[0]['n_images'] if rows else 0} citra test; GT: {src})", "",
             "| Konfigurasi | Citra | Field | Exact | Close | Wrong | Missing | Field Acc. | 95% CI | Exact@5% | Image full-match |",
             "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['config']} | {r['n_images']} | {r['n_fields']} | {r['exact']} | {r['close']} | "
                     f"{r['wrong']} | {r['missing']} | {r['field_accuracy']:.2%} | "
                     f"[{r['ci_low']:.2%}, {r['ci_high']:.2%}] | {r['exact_at_5pct']:.2%} | {r['image_full_match']:.2%} |")
    if rows and rows[0].get("n_unreadable") is not None:
        lines += ["", f"> {rows[0]['n_unreadable']} field GT **tidak terbaca pada foto** "
                  "(baris tercetak, nilai tidak terlihat) dikeluarkan dari semua denominator. "
                  "Denominator tiap konfigurasi adalah field yang nilai GT-nya terverifikasi."]
    lines += ["", "## Perbandingan berpasangan (paired bootstrap + McNemar)", "",
              "| A vs B | Δ (A−B) | 95% CI Δ | n01 (B benar, A salah) | n10 (A benar, B salah) | McNemar p | Kesimpulan |",
              "| :--- | ---: | :---: | ---: | ---: | ---: | :--- |"]
    pairs = [("E_full_pipeline", "D_baseline_parser"), ("E_full_pipeline", "C_paddle_raw"),
             ("D_baseline_parser", "C_paddle_raw"), ("C_paddle_raw", "A_tesseract"),
             ("C_paddle_raw", "B_easyocr"), ("E_full_pipeline", "A_tesseract"), ("E_full_pipeline", "B_easyocr"),
             # grid/table-structure effect: identical stage set, only grid differs
             ("E_full_pipeline", "cum_plus_validation"),
             # LOO effects measured against the full pipeline
             ("loo_full_minus_semantic", "E_full_pipeline"),
             ("E_full_pipeline", "loo_full_minus_fuzzy"),
             ("E_full_pipeline", "loo_full_minus_fallback"),
             ("E_full_pipeline", "loo_full_minus_validation")]
    for a, b in pairs:
        if a not in results or b not in results:
            continue
        d, lo, hi = paired_bootstrap(results[a], results[b])
        n01, n10, p = mcnemar(results[a], results[b])
        sig = lo > 0 or hi < 0
        concl = ("berbeda (A lebih baik)" if d > 0 and sig else
                 "berbeda (B lebih baik)" if d < 0 and sig else "tidak berbeda secara statistik")
        if sig and p >= 0.05:
            concl += " (McNemar p≥0,05 → hati-hati)"
        lines.append(f"| {a} vs {b} | {d:+.2%} | [{lo:+.2%}, {hi:+.2%}] | {n01} | {n10} | {p:.3g} | {concl} |")
    (OUT / "results_main.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ablation(results: dict[str, dict], order: list[str], label: str,
                   ref: str | None, src: str) -> None:
    rows = []
    prev = None
    for name in order:
        if name not in results:
            continue
        r = results[name]
        lo, hi = bootstrap_ci(r["per_image"])
        delta_prev = round(r["field_accuracy"] - results[prev]["field_accuracy"], 4) if prev in results else ""
        delta_ref = round(r["field_accuracy"] - results[ref]["field_accuracy"], 4) if ref in results else ""
        rows.append({
            "config": name, "n_images": r["n_images"], "n_fields": r["n_fields"],
            "exact": r["exact"], "close": r["close"], "wrong": r["wrong"], "missing": r["missing"],
            "field_accuracy": round(r["field_accuracy"], 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "delta_vs_prev": delta_prev, "delta_vs_full": delta_ref,
            "image_full_match": round(r["image_full_match"], 4), "gt_source": src,
        })
        prev = name
    _write_csv(OUT / f"results_ablation_{label}.csv", rows)
    title = "Ablation kumulatif" if label == "cumulative" else "Ablation leave-one-out (terhadap full)"
    lines = [f"# {title} (GT: {src})", "",
             "| Konfigurasi | Field | Exact | Close | Wrong | Missing | Field Acc. | 95% CI | Δ vs sebelumnya | "
             + ("Image full-match |" if label == "cumulative" else "Δ vs full | Image full-match |"),
             "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | " + ("---: |" if label == "cumulative" else "---: | ---: |")]
    for r in rows:
        d = f"{r['delta_vs_prev']:+.2%}" if isinstance(r["delta_vs_prev"], float) else "—"
        extra = f"{r['image_full_match']:.2%} |" if label == "cumulative" else f"{r['delta_vs_full']:+.2%} | {r['image_full_match']:.2%} |"
        lines.append(f"| {r['config']} | {r['n_fields']} | {r['exact']} | {r['close']} | {r['wrong']} | "
                     f"{r['missing']} | {r['field_accuracy']:.2%} | [{r['ci_low']:.2%}, {r['ci_high']:.2%}] | {d} | {extra}")
    (OUT / f"results_ablation_{label}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_per_field(results: dict[str, dict], src: str) -> None:
    rows = []
    for name in MAIN_ORDER:
        if name not in results:
            continue
        fc = results[name]["field_counts"]
        for field in FIELDS:
            c = fc.get(field, {})
            n = c.get("exact", 0) + c.get("close", 0) + c.get("wrong", 0) + c.get("missing", 0)
            if n == 0:
                continue
            ex, cl = c.get("exact", 0), c.get("close", 0)
            rows.append({"config": name, "field": field, "n": n,
                         "exact": ex, "close": cl, "wrong": c.get("wrong", 0), "missing": c.get("missing", 0),
                         "unreadable": c.get("unreadable", 0),
                         "accuracy": round(ex / n, 4), "exact_at_5pct": round((ex + cl) / n, 4),
                         "gt_source": src})
    _write_csv(OUT / "results_per_field.csv", rows)
    lines = [f"# Hasil per field (GT: {src})", "",
             "| Konfigurasi | Field | n | Exact | Close | Wrong | Missing | Unreadable | Acc. | Exact@5% |",
             "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['config']} | {r['field']} | {r['n']} | {r['exact']} | {r['close']} | "
                     f"{r['wrong']} | {r['missing']} | {r['unreadable']} | {r['accuracy']:.2%} | {r['exact_at_5pct']:.2%} |")
    (OUT / "results_per_field.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_errors(results: dict[str, dict]) -> None:
    rows = []
    for name in MAIN_ORDER + CUM_ORDER:
        if name not in results:
            continue
        rows.extend(results[name]["errors"])
    _write_csv(OUT / "error_taxonomy.csv", rows,
               ["file", "config", "field", "gt", "pred", "rel_error", "kind", "flags"])
    kinds: dict[str, int] = defaultdict(int)
    for r in rows:
        kinds[r["kind"]] += 1
    lines = ["# Taksonomi error (semua konfigurasi)", "", "| Jenis | Jumlah |", "| :--- | ---: |"]
    for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} |")
    (OUT / "error_taxonomy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    global GT_CACHE
    import sys
    allow_partial = "--allow-partial" in sys.argv
    gt, src = load_gt()
    GT_CACHE = gt
    available = [p.stem for p in PRED.glob("*.jsonl")]
    print(f"GT source: {src}  ({len(gt)} records)")
    print(f"Prediction files: {sorted(available)}")
    results = {}
    for name in MAIN_ORDER + CUM_ORDER + LOO_ORDER:
        if name in available:
            r = evaluate(name, gt)
            if r["n_images"] != len(gt) and not allow_partial:
                # A partial prediction file would silently shrink the
                # denominator and distort the comparison. Never publish it.
                print(f"  SKIP {name}: incomplete "
                      f"({r['n_images']}/{len(gt)} images); rerun the harness.")
                continue
            results[name] = r
            print(f"  {name:28s} acc={r['field_accuracy']:.4f} "
                  f"(n={r['n_fields']} fields, {r['n_images']} imgs)")
    # timing table
    timing = [{"config": n, "n_images": r["n_images"],
               "median_s": r["median_dur"], "p95_s": r["p95_dur"], "n_timed": r["n_dur"]}
              for n, r in results.items()]
    _write_csv(OUT / "timing.csv", timing)
    n_unreadable = next(iter(results.values()))["n_unreadable"] if results else 0
    write_main(results, src)
    write_ablation(results, CUM_ORDER, "cumulative", None, src)
    write_ablation(results, LOO_ORDER, "loo", "E_full_pipeline", src)
    write_per_field(results, src)
    write_errors(results)
    (OUT / "metrics_status.json").write_text(json.dumps(
        {"gt_source": src, "n_gt": len(gt), "configs_evaluated": sorted(results),
         "n_unreadable_fields": n_unreadable,
         "gt_unreadable_note": (
             f"{n_unreadable} field yang tercetak di label tidak terbaca pada foto "
             "(terutama lemak_trans, kolesterol, serat) dan dikeluarkan dari semua "
             "denominator; ini keterbatasan kualitas foto, bukan kelalaian anotasi."),
         "note": "GT draft belum diverifikasi manusia" if src != "human_verified" else "GT terverifikasi manusia"},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nWrote results_main, results_per_field, results_ablation_*, error_taxonomy, timing.")


if __name__ == "__main__":
    main()
