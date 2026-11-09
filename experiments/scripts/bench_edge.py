"""Edge-deployment benchmark for reviewer point B10.

Measures, on this machine and on an explicit device:

  * artefact size (YOLO weights, PaddleOCR model directories),
  * segmentation latency (median / P95, warm-up excluded),
  * nutrition-label OCR latency through the real ``jalankan_ocr`` pipeline,
  * clinical-rule latency through the real ``frontend/js/clinical-rules.js``,
  * peak process RAM (sampled) and peak CUDA memory for the YOLO phase.

End-to-end latency is reported as the **sum of the measured components** for one
plate image + one label image + one rule evaluation. It is not a concurrent
camera capture; say so in the manuscript.

Run (sequentially, never while another GPU job is active):
    python experiments/scripts/bench_edge.py --device gpu --out experiments/results/edge/gpu
    python experiments/scripts/bench_edge.py --device cpu --out experiments/results/edge/cpu
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    # ``backend`` is a sibling package of experiments/, not installed.
    sys.path.insert(0, str(ROOT))
PRED = ROOT / "experiments" / "results" / "ocr_eval" / "predictions"
TEST = ROOT / "experiments" / "results" / "ocr_eval" / "annotation" / "test_files.json"
YOLO_PT = ROOT / "models" / "isipiringku_v2" / "weights" / "last.pt"
YOLO_ONNX = ROOT / "models" / "isipiringku_v2" / "weights" / "best.onnx"
RULES_JS = ROOT / "frontend" / "js" / "clinical-rules.js"


class Sampler:
    """Background peak-RAM sampler (and optional nvidia-smi process VRAM)."""

    def __init__(self, pid: int):
        import psutil
        self.proc = psutil.Process(pid)
        self.peak_rss = 0
        self.peak_vram = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        import psutil
        while not self._stop.is_set():
            try:
                self.peak_rss = max(self.peak_rss, self.proc.memory_info().rss)
            except psutil.Error:
                pass
            self.peak_vram = max(self.peak_vram, _nvidia_pid_mb(os.getpid()))
            time.sleep(0.05)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=1)


def _nvidia_pid_mb(pid: int) -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        ).stdout
        best = 0
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0] == str(pid):
                best = max(best, int(float(parts[1])))
        return best
    except Exception:
        return 0


def _load_images(rel_files: list[str], n: int) -> list:
    import cv2
    import numpy as np
    from PIL import Image, ImageOps
    imgs = []
    for rel in rel_files[:n]:
        p = ROOT / rel
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im)
            imgs.append(cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR))
    return imgs


def _stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    return {
        "n": len(s),
        "median_s": round(statistics.median(s), 4),
        "p95_s": round(s[max(0, int(0.95 * (len(s) - 1)))], 4),
        "min_s": round(s[0], 4),
        "max_s": round(s[-1], 4),
    }


def _dir_mb(p: Path) -> float | None:
    if not p.exists():
        return None
    return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6, 1)


def seg_bench(device: str, imgs: list, warmup: int) -> dict:
    from ultralytics import YOLO
    import torch
    if device == "gpu" and not torch.cuda.is_available():
        raise SystemExit(
            "GPU requested but this PyTorch build has no CUDA support "
            f"(torch {torch.__version__}). Install a CUDA wheel or use --device cpu."
        )
    # Ultralytics expects "0" for the first CUDA device, not "gpu".
    ultralytics_device = "0" if device == "gpu" else "cpu"
    model = YOLO(str(YOLO_PT))
    for im in imgs[:warmup]:
        model.predict(im, imgsz=640, device=ultralytics_device, verbose=False)
    times = []
    with Sampler(os.getpid()) as smp:
        for im in imgs:
            t0 = time.perf_counter()
            model.predict(im, imgsz=640, device=ultralytics_device, verbose=False)
            times.append(time.perf_counter() - t0)
    peak_cuda = None
    try:
        if torch.cuda.is_available():
            peak_cuda = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    except Exception:
        pass
    return {**_stats(times), "peak_rss_mb": round(smp.peak_rss / 1e6, 1),
            "peak_vram_mb": smp.peak_vram or None, "peak_cuda_alloc_mb": peak_cuda}


def ocr_bench(rel_files: list[str], n: int) -> dict:
    from backend.app import jalankan_ocr
    times = []
    with Sampler(os.getpid()) as smp:
        for rel in rel_files[:n]:
            t0 = time.perf_counter()
            jalankan_ocr(str(ROOT / rel))
            times.append(time.perf_counter() - t0)
    return {**_stats(times), "peak_rss_mb": round(smp.peak_rss / 1e6, 1),
            "peak_vram_mb": smp.peak_vram or None}


def rules_bench(iters: int = 50) -> dict:
    if not RULES_JS.is_file():
        return {"n": 0, "note": "clinical-rules.js not found"}
    script = (
        "global.window={};"
        "require(process.argv[1]);"
        "const R=global.window.ClinicalHealthRules;"
        "const r=R.create({tdee:2000,isDiabetic:true,isHypertensive:false});"
        "const meal={energy_kj:1800,sugar_g:30,sodium_mg:900,"
        "saturated_fat_g:8,protein_g:10,fiber_g:5};"
        "const t0=process.hrtime.bigint();"
        f"for(let i=0;i<{iters};i++){{r.evaluateMeal(meal,100);}}"
        "const ms=Number(process.hrtime.bigint()-t0)/1e6/"
        f"{iters};"
        "console.log(JSON.stringify({median_s:ms/1000}))"
    )
    try:
        out = subprocess.run(["node", "-e", script, str(RULES_JS)],
                             capture_output=True, text=True, timeout=60)
        data = json.loads(out.stdout.strip().splitlines()[-1])
        return {"n": iters, "iterations": iters, "median_s": data["median_s"],
                "per_call_s": data["median_s"]}
    except Exception as exc:
        return {"n": 0, "note": f"node benchmark failed: {type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--n-seg", type=int, default=30)
    ap.add_argument("--n-ocr", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", default="experiments/results/edge")
    args = ap.parse_args()

    os.environ.setdefault("NUTRILABEL_OCR_DEVICE", "gpu" if args.device == "gpu" else "cpu")
    os.environ.setdefault("NUTRILABEL_USE_TABLE_STRUCTURE", "1")

    import torch  # noqa: F401  (must precede paddle)

    files = json.loads(TEST.read_text(encoding="utf-8"))
    imgs = _load_images(files, args.n_seg)

    result: dict = {
        "device": args.device,
        "model": "YOLO11s-seg",
        "artefacts_mb": {
            "yolo_last_pt": round(YOLO_PT.stat().st_size / 1e6, 1) if YOLO_PT.is_file() else None,
            "yolo_best_onnx": round(YOLO_ONNX.stat().st_size / 1e6, 1) if YOLO_ONNX.is_file() else None,
            "ppocr_det": _dir_mb(Path(os.environ.get("USERPROFILE", "")) / ".paddlex/official_models/PP-OCRv4_mobile_det"),
            "ppocr_rec": _dir_mb(Path(os.environ.get("USERPROFILE", "")) / ".paddlex/official_models/en_PP-OCRv4_mobile_rec"),
            "slanet": _dir_mb(Path(os.environ.get("USERPROFILE", "")) / ".paddlex/official_models/SLANet_plus"),
        },
        "segmentation": seg_bench(args.device, imgs, args.warmup),
        "ocr": ocr_bench(files, args.n_ocr),
        "clinical_rules": rules_bench(),
    }
    seg = result["segmentation"].get("median_s") or 0
    ocr = result["ocr"].get("median_s") or 0
    rules = result["clinical_rules"].get("median_s") or 0
    result["end_to_end_sum_s"] = {
        "median_s": round(seg + ocr + rules, 4),
        "note": "sum of component medians (plate seg + label OCR + rules); not a concurrent capture",
    }
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        f"# Edge benchmark — device: {args.device}",
        "",
        f"- Segmen (n={result['segmentation'].get('n')}): median "
        f"{result['segmentation'].get('median_s')} s, P95 {result['segmentation'].get('p95_s')} s",
        f"- OCR (n={result['ocr'].get('n')}): median {result['ocr'].get('median_s')} s, "
        f"P95 {result['ocr'].get('p95_s')} s",
        f"- Aturan klinis: {result['clinical_rules'].get('median_s')} s/panggilan",
        f"- End-to-end (jumlah komponen): {result['end_to_end_sum_s']['median_s']} s",
        f"- Peak RAM: {result.get('segmentation', {}).get('peak_rss_mb')} MB (seg) / "
        f"{result.get('ocr', {}).get('peak_rss_mb')} MB (OCR)",
        "",
        "Ukuran artefak (MB):",
        "",
        *[f"- {k}: {v}" for k, v in result["artefacts_mb"].items()],
        "",
    ]
    (out / "bench.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
