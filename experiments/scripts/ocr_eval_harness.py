"""OCR evaluation harness for Healthier / JRI 2026.

Runs every comparison configuration on the locked *test* split and writes one
prediction file per configuration. Two disk caches make the ablation affordable:

* ``cache/preprocessed/`` — output of ``Preprocessor.proses`` keyed by
  (file, preprocessing flags).
* ``cache/detections/``   — raw OCR detections keyed by the hash of the exact
  numpy array handed to ``PPOCREngine.deteksi`` (+ whether the table engine is
  active). Post-processing stages (semantic filter, fuzzy, fallback, validation,
  table-grid) therefore reuse the *same* OCR output.

The harness never reads ground truth: it only produces predictions, so it can
run before the human verification of the draft annotations. Metrics are joined
later by ``ocr_eval_metrics.py``.

Usage:
    python experiments/scripts/ocr_eval_harness.py --configs all
    python experiments/scripts/ocr_eval_harness.py --configs E_full_pipeline --force
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "experiments" / "results" / "ocr_eval"
CACHE = OUT / "cache"
PREDIR = CACHE / "preprocessed"
DETDIR = CACHE / "detections"
PRED = OUT / "predictions"
LOGDIR = OUT / "logs"

STAGE_FLAG = {
    "deskew": "HEALTHIER_OCR_DESKEW",
    "deglare": "HEALTHIER_OCR_DEGLARE",
    "clahe": "HEALTHIER_OCR_CLAHE",
    "roi": "HEALTHIER_OCR_ROI",
    "semantic": "HEALTHIER_OCR_SEMANTIC_FILTER",
    "fuzzy": "HEALTHIER_OCR_FUZZY",
    "fallback": "HEALTHIER_OCR_FALLBACK",
    "validation": "HEALTHIER_OCR_VALIDATION",
}
ALL_STAGES = list(STAGE_FLAG)
PIPELINE_FLAGS = list(STAGE_FLAG.values()) + ["HEALTHIER_OCR_EXIF_ORIENT"]

FIELDS = [
    "kalori", "lemak_total", "lemak_jenuh", "lemak_trans", "kolesterol",
    "natrium", "karbohidrat", "serat", "gula", "protein",
]

# --- configuration catalogue -------------------------------------------------

MAIN = {
    "A_tesseract": {"engine": "tesseract"},
    "B_easyocr": {"engine": "easyocr"},
    "C_paddle_raw": {"stages": set(), "grid": 0, "parser": "simple"},
    "D_baseline_parser": {"stages": {"deskew", "deglare", "clahe", "roi"}, "grid": 0, "parser": "pipeline"},
    "E_full_pipeline": {"stages": set(ALL_STAGES), "grid": 1, "parser": "pipeline"},
}

CUMULATIVE = [
    ("cum_raw", set()),
    ("cum_plus_deskew", {"deskew"}),
    ("cum_plus_deglare", {"deskew", "deglare"}),
    ("cum_plus_clahe", {"deskew", "deglare", "clahe"}),
    ("cum_plus_roi", {"deskew", "deglare", "clahe", "roi"}),
    ("cum_plus_semantic", {"deskew", "deglare", "clahe", "roi", "semantic"}),
    ("cum_plus_fuzzy", {"deskew", "deglare", "clahe", "roi", "semantic", "fuzzy"}),
    ("cum_plus_fallback", {"deskew", "deglare", "clahe", "roi", "semantic", "fuzzy", "fallback"}),
    ("cum_plus_validation", set(ALL_STAGES)),
]

LOO = [
    (f"loo_full_minus_{stage}", set(ALL_STAGES) - {stage})
    for stage in ALL_STAGES
]

CONFIGS: dict[str, dict] = dict(MAIN)
for name, stages in CUMULATIVE:
    CONFIGS[name] = {"stages": stages, "grid": 0, "parser": "pipeline"}
for name, stages in LOO:
    CONFIGS[name] = {"stages": stages, "grid": 1, "parser": "pipeline"}

# priority order: main comparison first (C/D/E reuse OCR cache), then Tesseract,
# then cumulative ablation, then LOO, then the slow CPU EasyOCR baseline last.
ORDER = (["C_paddle_raw", "D_baseline_parser", "E_full_pipeline", "A_tesseract"]
         + [n for n, _ in CUMULATIVE]
         + [n for n, _ in LOO]
         + ["B_easyocr"])


def apply_env(stages: set[str], grid: int) -> None:
    for name, flag in STAGE_FLAG.items():
        os.environ[flag] = "1" if name in stages else "0"
    os.environ["HEALTHIER_OCR_EXIF_ORIENT"] = "1"
    os.environ["HEALTHIER_OCR_WATERMARK"] = "0"
    os.environ["HEALTHIER_OCR_TABLE_GRID"] = str(int(grid))


# --- simple shared parser (identical for A, B, C) ----------------------------

SIMPLE_LABELS = [
    ("energi total/total energy", "kalori"), ("energi total total calories", "kalori"),
    ("total energy", "kalori"), ("total calories", "kalori"),
    ("lemak total/total fat", "lemak_total"), ("lemak total", "lemak_total"),
    ("total fat", "lemak_total"), ("lemak jenuh/saturated fat", "lemak_jenuh"),
    ("lemak jenuh", "lemak_jenuh"), ("saturated fat", "lemak_jenuh"),
    ("lemak trans/trans fat", "lemak_trans"), ("lemak trans", "lemak_trans"),
    ("trans fat", "lemak_trans"), ("karbohidrat total", "karbohidrat"),
    ("total karbohidrat", "karbohidrat"), ("total carbohydrate", "karbohidrat"),
    ("karbohidrat", "karbohidrat"), ("carbohydrate", "karbohidrat"),
    ("serat pangan", "serat"), ("dietary fiber", "serat"),
    ("gula total", "gula"), ("total sugars", "gula"), ("sugars", "gula"),
    ("kolesterol", "kolesterol"), ("cholesterol", "kolesterol"),
    ("natrium", "natrium"), ("sodium", "natrium"), ("garam", "natrium"),
    ("energi", "kalori"), ("kalori", "kalori"), ("energy", "kalori"),
    ("calories", "kalori"), ("protein", "protein"), ("serat", "serat"),
    ("fiber", "serat"), ("gula", "gula"), ("sugar", "gula"),
    ("lemak", "lemak_total"), ("fat", "lemak_total"),
]
SIMPLE_LABELS.sort(key=lambda kv: len(kv[0]), reverse=True)
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().replace("µ", "u")).strip()


def _nums(line: str) -> list[float]:
    out = []
    for m in _NUM_RE.finditer(line):
        tail = line[m.end():m.end() + 2]
        if "%" in tail:
            continue
        out.append(float(m.group(0).replace(",", ".")))
    return out


def simple_parser(detections: list[dict]) -> dict:
    """Line-based regex parser, shared verbatim by configs A, B and C."""
    det = [d for d in detections if d.get("text")]
    if not det:
        return {"takaran_saji_g": None, "nutrisi": {}}
    heights = [max(1.0, float(d["bbox"][3])) for d in det if d.get("bbox")]
    med_h = sorted(heights)[len(heights) // 2] if heights else 20.0
    tol = max(12.0, 0.6 * med_h)

    det = sorted(det, key=lambda d: (d.get("center_y", 0), d.get("center_x", 0)))
    lines: list[dict] = []
    for d in det:
        cy = d.get("center_y", 0)
        target = None
        for line in lines:
            if abs(line["cy"] - cy) <= tol:
                target = line
                break
        if target is None:
            lines.append({"cy": cy, "items": [d]})
        else:
            target["items"].append(d)
            target["cy"] = sum(i.get("center_y", 0) for i in target["items"]) / len(target["items"])

    out: dict[str, float | None] = {}
    takaran: float | None = None
    for line in lines:
        items = sorted(line["items"], key=lambda d: d.get("center_x", 0))
        joined = " ".join(_norm(d["text"]) for d in items)
        if takaran is None and re.search(r"(takaran saji|serving size|jumlah per sajian|\bsaji\b)", joined):
            m = re.search(r"(\d+(?:[.,]\d+)?)\s*(g|ml|gram)\b", joined)
            if m:
                takaran = float(m.group(1).replace(",", "."))
            else:
                nums = _nums(joined)
                if nums:
                    takaran = nums[0]
        for label, field in SIMPLE_LABELS:
            if field in out and out[field] is not None:
                continue
            if label in joined:
                nums = _nums(joined)
                if nums:
                    out[field] = nums[0]
                break
    return {"takaran_saji_g": takaran, "nutrisi": {k: v for k, v in out.items() if k in FIELDS}}


# --- OCR / preprocessing caches ----------------------------------------------

_orig_proses = None
_orig_deteksi = None


def _file_sig(path: Path) -> str:
    st = path.stat()
    return f"{path}|{st.st_size}|{int(st.st_mtime)}"


def _pre_key(rel: str) -> str:
    flags = tuple(os.getenv(f, "") for f in PIPELINE_FLAGS)
    return hashlib.sha256((_file_sig(ROOT / rel) + "|" + "|".join(flags)).encode()).hexdigest()[:40]


def _json_default(o):
    """Make numpy scalars/arrays JSON-serialisable (PaddleOCR returns np types)."""
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def install_caches() -> None:
    global _orig_proses, _orig_deteksi
    import numpy as np
    from backend.app import Preprocessor, PPOCREngine

    _orig_proses = Preprocessor.proses
    _orig_deteksi = PPOCREngine.deteksi

    # NOTE: preprocessed arrays are deliberately *not* cached to disk (hundreds
    # of MB/images); only the expensive OCR detections are cached. Preprocessing
    # is deterministic, so recomputing it yields bit-identical arrays and the
    # detection cache still hits.
    _ = _orig_proses

    def cached_deteksi(self, img_input):
        if isinstance(img_input, np.ndarray):
            h = hashlib.sha256(np.ascontiguousarray(img_input).tobytes()).hexdigest()
            shape = "x".join(map(str, img_input.shape))
        else:
            h = hashlib.sha256(str(img_input).encode()).hexdigest()
            shape = "path"
        key = f"{h[:44]}_{shape}_{int(self.table_engine is not None)}"
        p = DETDIR / f"{key}.json"
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
        res = _orig_deteksi(self, img_input)
        p.write_text(json.dumps(res, ensure_ascii=False, default=_json_default), encoding="utf-8")
        return res

    PPOCREngine.deteksi = cached_deteksi


# --- engines -----------------------------------------------------------------

def _raw_bgr(path: Path):
    import cv2
    import numpy as np
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)


def _to_dets_from_quad(boxes, texts, confs):
    dets = []
    for box, text, conf in zip(boxes, texts, confs):
        if text is None or str(text).strip() == "":
            continue
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        dets.append({"bbox": [x0, y0, x1 - x0, y1 - y0], "text": str(text),
                     "conf": float(conf), "center_x": (x0 + x1) / 2, "center_y": (y0 + y1) / 2})
    dets.sort(key=lambda d: (d["center_y"], d["center_x"]))
    return dets


_TESS_READY = False


def _ensure_tesseract() -> None:
    global _TESS_READY
    if _TESS_READY:
        return
    import shutil
    import pytesseract
    if shutil.which("tesseract") is None:
        candidates = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Tesseract-OCR" / "tesseract.exe",
            Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
            Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
        ]
        for c in candidates:
            if c.is_file():
                pytesseract.pytesseract.tesseract_cmd = str(c)
                break
    _TESS_READY = True


def run_tesseract(path: Path):
    import cv2
    import pytesseract
    from pytesseract import Output
    _ensure_tesseract()
    img = _raw_bgr(path)
    t0 = time.time()
    data = pytesseract.image_to_data(img, output_type=Output.DICT, config="--psm 6")
    dur = time.time() - t0
    dets = []
    for i in range(len(data["text"])):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf = float(data["conf"][i]) / 100.0
        except (TypeError, ValueError):
            conf = 0.0
        if conf <= 0:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        dets.append({"bbox": [float(x), float(y), float(w), float(h)], "text": txt,
                     "conf": conf, "center_x": x + w / 2, "center_y": y + h / 2})
    dets.sort(key=lambda d: (d["center_y"], d["center_x"]))
    return dets, dur


_EASY = None


def run_easyocr(path: Path):
    global _EASY
    import cv2
    import easyocr
    if _EASY is None:
        _EASY = easyocr.Reader(["en"], gpu=False, verbose=False)
    img = _raw_bgr(path)
    t0 = time.time()
    res = _EASY.readtext(img)
    dur = time.time() - t0
    boxes = [r[0] for r in res]
    texts = [r[1] for r in res]
    confs = [float(r[2]) for r in res]
    return _to_dets_from_quad(boxes, texts, confs), dur


# --- pipeline runner ---------------------------------------------------------

def _num(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def run_pipeline(path: Path, cfg: dict) -> dict:
    from backend.app import jalankan_ocr
    result = jalankan_ocr(str(path))
    dets = result.get("ocr_detections") or []
    if cfg["parser"] == "simple":
        parsed = simple_parser(dets)
        nut = parsed["nutrisi"]
        takaran = parsed["takaran_saji_g"]
    else:
        data = result.get("data") or {}
        nut = {}
        for k, v in data.items():
            nut[k] = _num(v.get("per_sajian") if isinstance(v, dict) else v)
        takaran = _num(result.get("takaran_saji_g"))
    return {
        "status": result.get("status"),
        "flags": result.get("flags", []),
        "durasi_detik": result.get("durasi_detik"),
        "takaran_saji_g": takaran,
        "nutrisi": {k: nut.get(k) for k in FIELDS},
        "n_detections": len(dets),
        "pass1_anchor_found": result.get("pass1_anchor_found"),
    }


def run_config(name: str, files: list[str], force: bool) -> None:
    cfg = CONFIGS[name]
    out_path = PRED / f"{name}.jsonl"
    if out_path.is_file() and not force:
        done = sum(1 for _ in out_path.open(encoding="utf-8"))
        if done >= len(files):
            print(f"[skip] {name}: {done} records already present")
            return
    if cfg.get("engine") == "tesseract":
        engine_fn = run_tesseract
    elif cfg.get("engine") == "easyocr":
        engine_fn = run_easyocr
    else:
        apply_env(cfg["stages"], cfg["grid"])
        engine_fn = lambda p: (None, None)  # replaced below

    LOGDIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGDIR / f"{name}.log"
    t_start = time.time()
    print(f"\n=== {name} ===  stages={sorted(cfg.get('stages', []))} grid={cfg.get('grid')} "
          f"parser={cfg.get('parser', cfg.get('engine'))}")
    with out_path.open("w", encoding="utf-8") as fh, log_path.open("a", encoding="utf-8") as log:
        for i, rel in enumerate(files, 1):
            path = ROOT / rel
            try:
                if cfg.get("engine"):
                    dets, dur = engine_fn(path)
                    parsed = simple_parser(dets)
                    rec = {"status": "ok", "flags": [], "durasi_detik": round(dur, 3),
                           "takaran_saji_g": parsed["takaran_saji_g"],
                           "nutrisi": {k: parsed["nutrisi"].get(k) for k in FIELDS},
                           "n_detections": len(dets), "pass1_anchor_found": None}
                else:
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        rec = run_pipeline(path, cfg)
                    log.write(f"--- {rel} ---\n{buf.getvalue()}\n")
            except Exception as exc:  # keep going; record the failure
                rec = {"status": f"error:{type(exc).__name__}", "flags": [str(exc)[:200]],
                       "durasi_detik": None, "takaran_saji_g": None,
                       "nutrisi": {k: None for k in FIELDS}, "n_detections": 0,
                       "pass1_anchor_found": None}
            rec["file"] = rel
            rec["config"] = name
            fh.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")
            fh.flush()
            if i % 10 == 0 or i == len(files):
                print(f"  ... {i}/{len(files)} ({time.time()-t_start:.0f}s)")
    print(f"[done] {name}: {len(files)} images in {time.time()-t_start:.0f}s -> {out_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Healthier OCR evaluation harness")
    ap.add_argument("--configs", default="main",
                    help="'all', 'main', 'cumulative', 'loo', or comma-separated names")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="gpu")
    args = ap.parse_args()

    os.environ["NUTRILABEL_OCR_DEVICE"] = args.device
    os.environ.setdefault("NUTRILABEL_USE_TABLE_STRUCTURE", "1")

    for d in (CACHE, PREDIR, DETDIR, PRED, LOGDIR):
        d.mkdir(parents=True, exist_ok=True)

    files = json.loads((OUT / "annotation" / "test_files.json").read_text(encoding="utf-8"))
    if args.limit:
        files = files[: args.limit]

    sel = args.configs.strip().lower()
    if sel == "all":
        names = ORDER
    elif sel == "main":
        names = list(MAIN)
    elif sel == "cumulative":
        names = [n for n, _ in CUMULATIVE]
    elif sel == "loo":
        names = [n for n, _ in LOO]
    else:
        names = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown configs: {unknown}")

    import torch  # noqa: F401  (must precede paddle)
    install_caches()

    needs_paddle = any(CONFIGS[n].get("engine") is None for n in names)
    if needs_paddle:
        from backend.app import PPOCREngine
        t0 = time.time()
        PPOCREngine().muat()
        print(f"[engine] PaddleOCR ready in {time.time()-t0:.1f}s (device={args.device})")

    for name in names:
        run_config(name, files, args.force)

    print("\nAll selected configs finished.")


if __name__ == "__main__":
    main()
