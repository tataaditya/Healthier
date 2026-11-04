"""Evaluate a cloud/local Vision-Language Model (VLM) on the same OCR pilot set.

This is an **analysis harness** (future-work comparison), not part of the core
edge pipeline. It sends each label photo to a vision model and scores the
extracted per-serving fields against the same ground truth used by
``eval_ocr.py`` (exact value or within 5%).

Two provider backends are supported:
    --provider ollama   Ollama API (local or cloud, e.g. ``gemma4:cloud``)
    --provider openai   OpenAI-compatible endpoint (e.g. Helipod
                        ``https://ai.helipod.io/v1``); API key read from
                        ``--api-key`` or ``.env`` (``API_KEY``).

Example (Ollama):
    python experiments/scripts/eval_vlm.py ^
        --data experiments/ocr_data/makanan_ringan ^
        --model gemma4:cloud ^
        --out experiments/results/ocr_vlm/gemma4_cloud ^
        --label "Makanan Ringan pilot (VLM gemma4:cloud)" ^
        --bootstrap 2000

Example (OpenAI-compatible / Helipod):
    python experiments/scripts/eval_vlm.py ^
        --provider openai --base-url https://ai.helipod.io/v1 ^
        --model deepseek-v4.1-flash ^
        --data experiments/ocr_data/makanan_ringan ^
        --out experiments/results/ocr_vlm/helipod_deepseek_v4_1_flash ^
        --label "VLM cloud deepseek-v4.1-flash" ^
        --bootstrap 2000
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parents[1]
for candidate in (str(ROOT_DIR), str(SCRIPTS_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import eval_ocr  # noqa: E402  (reuse scoring helpers so output stays consistent)

CANONICAL_FIELDS = eval_ocr.CANONICAL_FIELDS
IMAGE_SUFFIXES = eval_ocr.IMAGE_SUFFIXES

PROMPT = (
    "Gambar ini adalah foto tabel 'Informasi Nilai Gizi' pada kemasan pangan Indonesia. "
    "Baca tabel dengan teliti dan kembalikan HANYA objek JSON (tanpa penjelasan) berisi "
    "nilai per SATU SAJIAN, yaitu dari kolom 'Per Sajian' atau 'Takaran per Sajian' "
    "(JANGAN pakai kolom 'Per 100 g').\n"
    "Field yang diminta (satuan dalam tanda kurung):\n"
    "- takaran_saji_g (gram)\n"
    "- kalori (kkal)\n"
    "- lemak_total (g), lemak_jenuh (g), lemak_trans (g)\n"
    "- kolesterol (mg)\n"
    "- natrium (mg)\n"
    "- karbohidrat (g), serat (g), gula (g), protein (g)\n"
    "Jika suatu nilai tidak ada di label, tulis null. Jangan mengarang angka.\n"
    "Format keluaran: "
    '{"takaran_saji_g":null,"kalori":null,"lemak_total":null,"lemak_jenuh":null,'
    '"lemak_trans":null,"kolesterol":null,"natrium":null,"karbohidrat":null,'
    '"serat":null,"gula":null,"protein":null}'
)


def to_number(value):
    """Normalisasi keluaran VLM (angka, '2,5', '1.5 g') menjadi float atau None."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower().replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def extract_json(text: str):
    """Ambil objek JSON pertama dari jawaban model, tahan terhadap markdown fence."""
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        candidate = text[start:end + 1] if start != -1 and end > start else None
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        cleaned = re.sub(r",(\s*[}\]])", r"\1", candidate)  # trailing commas
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


def encode_image(path: Path, max_side: int) -> str:
    """Baca gambar, hormati EXIF, kecilkan bila perlu, kembalikan base64 JPEG."""
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        scale = max_side / max(w, h)
        if scale < 1:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
        return base64.b64encode(buf.getvalue()).decode()


def ollama_chat(host: str, model: str, prompt: str, image_b64: str,
                temperature: float, timeout: int, retries: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
        "stream": False,
        "options": {"temperature": temperature},
    }
    data = json.dumps(payload).encode()
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            f"{host.rstrip('/')}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(3 * attempt)
    raise RuntimeError(f"Ollama gagal setelah {retries} percobaan: {last_exc}")


_THINK_RE = re.compile(r"<think\b[^>]*>.*?(?:</think>|$)", re.S)


def strip_think(text: str) -> str:
    """Buang blok reasoning <think>...</think> agar JSON mudah ditemukan."""
    return _THINK_RE.sub("", text or "").strip()


def load_api_key(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    for name in ("HELIPOD_API_KEY", "OPENAI_API_KEY", "API_KEY"):
        value = os.getenv(name)
        if value:
            return value
    env_file = ROOT_DIR / ".env"
    if env_file.is_file():
        match = re.search(
            r"(?:API_KEY|HELIPOD_API_KEY)\s*=\s*(\S+)",
            env_file.read_text(encoding="utf-8"),
        )
        if match:
            return match.group(1)
    return None


def openai_chat(base_url: str, api_key: str, model: str, prompt: str, image_b64: str,
                max_tokens: int, temperature: float, timeout: int, retries: int) -> str:
    """Panggil endpoint OpenAI-compatible (mis. Helipod) dengan gambar base64."""
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        # Cloudflare di depan gateway menolak User-Agent default urllib (error 1010).
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    }
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=data, headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.load(resp)
            message = (body.get("choices") or [{}])[0].get("message", {}) or {}
            return str(message.get("content") or "")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(3 * attempt)
    raise RuntimeError(f"OpenAI-compatible gagal setelah {retries} percobaan: {last_exc}")


def chat_content(args, api_key: str | None, image_b64: str) -> str:
    if args.provider == "openai":
        content = openai_chat(args.base_url, api_key or "", args.model, PROMPT, image_b64,
                              args.max_tokens, args.temperature, args.timeout, retries=1)
    else:
        response = ollama_chat(args.host, args.model, PROMPT, image_b64,
                               args.temperature, args.timeout, retries=1)
        content = (response.get("message") or {}).get("content", "")
    return strip_think(content)


def cached_elapsed(raw_dir: Path, stem: str):
    """Ambil durasi model dari cache mentah, bila ada (untuk mode --rescore)."""
    cache = raw_dir / f"{stem}.json"
    if not cache.is_file():
        return ""
    try:
        return json.loads(cache.read_text(encoding="utf-8")).get("elapsed_s", "")
    except (json.JSONDecodeError, OSError):
        return ""


def pred_from_vlm(parsed: dict) -> dict:
    """Ubah JSON VLM menjadi bentuk prediksi yang dipahami hitung_field_accuracy."""
    pred: dict = {}
    for field in CANONICAL_FIELDS:
        value = to_number(parsed.get(field))
        if value is not None:
            pred[field] = {"per_sajian": value, "satuan": None, "akg_persen": None}
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Field-level VLM evaluation (analysis).")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", choices=["ollama", "openai"], default="ollama",
                        help="ollama = API lokal; openai = endpoint OpenAI-compatible")
    parser.add_argument("--host", default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--base-url", default="https://ai.helipod.io/v1")
    parser.add_argument("--api-key", default=None, help="override; default dibaca dari .env")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--gt-suffix", default="_gt.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--label", default="")
    parser.add_argument("--force", action="store_true", help="abaikan cache respons mentah")
    parser.add_argument("--rescore", action="store_true",
                        help="skor ulang predictions.jsonl yang ada tanpa memanggil model")
    args = parser.parse_args()

    if not args.data.is_dir():
        raise SystemExit(f"Folder data tidak ditemukan: {args.data}")
    api_key = load_api_key(args.api_key) if args.provider == "openai" else None
    if args.provider == "openai" and not api_key and not args.rescore:
        raise SystemExit("Provider openai butuh API key (--api-key atau .env API_KEY).")
    from backend.app import hitung_field_accuracy  # noqa: E402

    images = sorted(
        p for p in args.data.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"Tidak ada gambar di {args.data}")

    out_dir = args.out.resolve()
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    saved_predictions: dict[str, dict] = {}
    if args.rescore:
        pred_path = out_dir / "predictions.jsonl"
        if not pred_path.is_file():
            raise SystemExit(f"Mode --rescore butuh predictions.jsonl di: {pred_path}")
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                saved_predictions[record["file"]] = record

    per_image_rows: list[dict] = []
    error_rows: list[dict] = []
    field_counts: dict[str, Counter] = defaultdict(Counter)
    field_abs_error: dict[str, list[float]] = defaultdict(list)
    field_rel_error: dict[str, list[float]] = defaultdict(list)
    bootstrap_rows: list[tuple[float, int]] = []
    missing_gt: list[str] = []
    predictions_fh = None if args.rescore else (out_dir / "predictions.jsonl").open("w", encoding="utf-8")

    try:
        for index, image_path in enumerate(images, 1):
            gt_path = image_path.with_name(image_path.stem + args.gt_suffix)
            gt = eval_ocr.load_gt(gt_path)
            if gt is None:
                missing_gt.append(image_path.name)
                continue
            gt_nutrition = {
                key: value for key, value in (gt.get("nutrisi") or {}).items()
                if key in CANONICAL_FIELDS and value is not None
            }
            if not gt_nutrition:
                missing_gt.append(image_path.name)
                continue

            if args.rescore:
                record = saved_predictions.get(image_path.name)
                if record is None:
                    missing_gt.append(image_path.name)
                    continue
                parsed = record.get("parsed") or {}
                pred = record.get("data") or pred_from_vlm(parsed)
                raw = {
                    "content": record.get("raw_vlm", ""),
                    "elapsed_s": cached_elapsed(raw_dir, image_path.stem),
                    "error": record.get("error"),
                }
            else:
                cache = raw_dir / f"{image_path.stem}.json"
                raw = None
                if cache.is_file() and not args.force:
                    cached = json.loads(cache.read_text(encoding="utf-8"))
                    # Ulangi respons kosong (model lokal kadang mengembalikan "").
                    if cached.get("content") or cached.get("error"):
                        raw = cached
                if raw is None:
                    b64 = encode_image(image_path, args.max_side)
                    raw = {"content": "", "error": None, "model": args.model}
                    for attempt in range(1, args.retries + 1):
                        try:
                            t0 = time.time()
                            content = chat_content(args, api_key, b64)
                            raw = {
                                "content": content,
                                "elapsed_s": round(time.time() - t0, 2),
                                "model": args.model,
                                "attempts": attempt,
                            }
                        except Exception as exc:  # pragma: no cover - depends on service
                            raw = {"content": "", "error": str(exc), "model": args.model}
                        if raw.get("content"):
                            break
                    cache.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

                parsed = extract_json(raw.get("content", "")) or {}
                pred = pred_from_vlm(parsed)
            evaluation = hitung_field_accuracy(gt_nutrition, pred, toleransi_pct=args.tolerance)
            per_field = evaluation["per_field"]
            correct = 0.0
            for field, info in per_field.items():
                status = info["status"]
                field_counts[field][status] += 1
                gt_value = info.get("gt")
                pred_value = info.get("pred")
                if status in ("exact", "close"):
                    correct += 1.0
                if pred_value is not None and gt_value is not None:
                    abs_error = abs(float(pred_value) - float(gt_value))
                    rel_error = abs_error / max(abs(float(gt_value)), 0.001)
                    field_abs_error[field].append(abs_error)
                    field_rel_error[field].append(rel_error)
                    if status == "wrong":
                        error_rows.append({
                            "file": image_path.name, "field": field, "gt": gt_value,
                            "pred": pred_value, "abs_error": round(abs_error, 4),
                            "rel_error": round(rel_error, 4), "status": status, "flags": "",
                        })
                elif status == "missing":
                    error_rows.append({
                        "file": image_path.name, "field": field, "gt": gt_value,
                        "pred": None, "abs_error": "", "rel_error": "",
                        "status": status, "flags": "",
                    })

            total = evaluation["total_fields"]
            bootstrap_rows.append((correct, total))
            per_image_rows.append({
                "file": image_path.name,
                "status": "ok" if not raw.get("error") else "error",
                "n_fields": total,
                "correct": correct,
                "field_accuracy": round(correct / total, 4) if total else 0.0,
                "all_correct": int(correct == total),
                "flags": int(bool(parsed)),
                "duration_s": raw.get("elapsed_s", ""),
                "gt": json.dumps(gt_nutrition, ensure_ascii=False),
            })
            if predictions_fh:
                predictions_fh.write(json.dumps({
                    "file": image_path.name,
                    "model": args.model,
                    "parsed": parsed,
                    "data": pred,
                    "raw_vlm": raw.get("content", ""),
                    "error": raw.get("error"),
                }, ensure_ascii=False) + "\n")

            print(f"  [{index}/{len(images)}] {image_path.name}: {correct:.0f}/{total} "
                  f"({raw.get('elapsed_s', '?')}s)")
    finally:
        if predictions_fh:
            predictions_fh.close()

    if not per_image_rows:
        raise SystemExit("Tidak ada gambar dengan ground truth yang valid.")

    total_correct = sum(row["correct"] for row in per_image_rows)
    total_fields = sum(row["n_fields"] for row in per_image_rows)
    field_accuracy = total_correct / total_fields if total_fields else 0.0
    image_accuracy = sum(row["all_correct"] for row in per_image_rows) / len(per_image_rows)
    mean_image_accuracy = sum(row["field_accuracy"] for row in per_image_rows) / len(per_image_rows)
    ci_low, ci_high = eval_ocr.bootstrap_ci(bootstrap_rows, args.bootstrap, args.seed)

    status_totals = Counter()
    for counts in field_counts.values():
        status_totals.update(counts)
    total_observed = sum(status_totals.values())

    summary = {
        "kind": "vlm_analysis",
        "label": args.label,
        "provider": args.provider,
        "model": args.model,
        "rescore": bool(args.rescore),
        "endpoint": args.base_url if args.provider == "openai" else args.host,
        "data": str(args.data),
        "max_side": args.max_side,
        "n_images": len(per_image_rows),
        "n_gt_images_skipped": len(missing_gt),
        "n_fields": total_fields,
        "field_accuracy": round(field_accuracy, 4),
        "field_accuracy_ci95": [round(ci_low, 4), round(ci_high, 4)],
        "image_level_accuracy": round(image_accuracy, 4),
        "mean_image_field_accuracy": round(mean_image_accuracy, 4),
        "status_counts": dict(status_totals),
        "status_rates": {
            key: round(value / total_observed, 4) if total_observed else 0.0
            for key, value in status_totals.items()
        },
        "tolerance": args.tolerance,
        "bootstrap_iterations": args.bootstrap,
        "packages": eval_ocr.package_versions(),
        "git_commit": eval_ocr.git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (out_dir / "per_image.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)

    per_field_rows = []
    for field in CANONICAL_FIELDS:
        counts = field_counts.get(field, Counter())
        n = sum(counts.values())
        if n == 0:
            continue
        correct = counts["exact"] + counts["close"]
        per_field_rows.append({
            "field": field, "n": n, "exact": counts["exact"], "close": counts["close"],
            "wrong": counts["wrong"], "missing": counts["missing"],
            "accuracy": round(correct / n, 4), "exact_rate": round(counts["exact"] / n, 4),
            "mean_abs_error": round(sum(field_abs_error[field]) / len(field_abs_error[field]), 4)
            if field_abs_error[field] else "",
            "mean_rel_error": round(sum(field_rel_error[field]) / len(field_rel_error[field]), 4)
            if field_rel_error[field] else "",
        })
    with (out_dir / "per_field.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(per_field_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_field_rows)

    with (out_dir / "errors.csv").open("w", encoding="utf-8", newline="") as fh:
        fieldnames = ["file", "field", "gt", "pred", "abs_error", "rel_error", "status", "flags"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(error_rows)

    lines = [
        f"# VLM analysis{f' — {args.label}' if args.label else ''}",
        "",
        f"- Model: **{args.model}**",
        f"- Images evaluated: **{len(per_image_rows)}**",
        f"- GT images skipped: {len(missing_gt)}",
        f"- Fields evaluated: **{total_fields}**",
        f"- Field accuracy (exact + within {args.tolerance:.0%}): **{field_accuracy:.2%}**",
        f"- 95% CI (cluster bootstrap by image): **[{ci_low:.2%}, {ci_high:.2%}]**",
        f"- Image-level all-fields-correct: **{image_accuracy:.2%}**",
        f"- Mean per-image field accuracy: **{mean_image_accuracy:.2%}**",
        "",
        "| Status | Count | Rate |",
        "| :--- | ---: | ---: |",
    ]
    for key in ("exact", "close", "wrong", "missing"):
        count = status_totals.get(key, 0)
        lines.append(f"| {key} | {count} | {count / total_observed:.2%} |")
    lines += ["", "## Per field", "",
              "| Field | n | Accuracy | Exact | Close | Wrong | Missing |",
              "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in per_field_rows:
        lines.append(
            f"| {row['field']} | {row['n']} | {row['accuracy']:.2%} | {row['exact']} "
            f"| {row['close']} | {row['wrong']} | {row['missing']} |"
        )
    lines += [
        "",
        "> **Future-work analysis only.** This harness is a comparison aid, not part "
        "of the published edge pipeline (which remains PP-OCRv4 + deterministic rules).",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved: {out_dir}")
    print(f"  model={args.model} field_accuracy={field_accuracy:.4f} "
          f"CI=[{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  images={len(per_image_rows)} fields={total_fields}")


if __name__ == "__main__":
    main()
