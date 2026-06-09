"""
NutriLabel v3 - Nutrition-Centric Batch Evaluator
=================================================

Evaluator ini sengaja menjadikan Field Accuracy sebagai metrik utama.
CER/WER tetap dihitung hanya jika tersedia ground truth teks, tetapi status
riset tidak ditentukan oleh full-text transcript.

Struktur folder:
  dataset_gizi/
    foto_001.jpg
    foto_001_gt.json   <- direkomendasikan
    foto_001_gt.txt    <- opsional untuk CER/WER sekunder

Format foto_001_gt.json:
{
  "takaran_saji_g": 50,
  "nutrisi": {
    "kalori": 90,
    "lemak_total": 4.5,
    "protein": 5,
    "karbohidrat": 6,
    "gula": 0,
    "natrium": 330
  }
}
"""

import argparse
import json
from pathlib import Path

from nutrilabel_v3_ppocr import jalankan_ocr, hitung_field_accuracy


CANONICAL_FIELDS = {
    "kalori", "lemak_total", "lemak_jenuh", "lemak_trans", "kolesterol",
    "natrium", "karbohidrat", "serat", "gula", "protein",
}


def load_ground_truth(foto_path: Path) -> tuple[dict | None, str | None]:
    gt_json_path = foto_path.with_name(foto_path.stem + "_gt.json")
    gt_txt_path = foto_path.with_name(foto_path.stem + "_gt.txt")

    gt_json = None
    gt_text = None
    if gt_json_path.exists():
        with open(gt_json_path, "r", encoding="utf-8") as f:
            gt_json = json.load(f)
    if gt_txt_path.exists():
        with open(gt_txt_path, "r", encoding="utf-8") as f:
            gt_text = f.read()
    return gt_json, gt_text


def hitung_nutrition_metrics(gt_json: dict, hasil: dict) -> dict:
    gt_nutrisi = gt_json.get("nutrisi", {})
    pred_nutrisi = hasil.get("data", {})
    field_eval = hitung_field_accuracy(gt_nutrisi, pred_nutrisi)

    gt_keys = set(gt_nutrisi.keys())
    pred_keys = {k for k in pred_nutrisi.keys() if k in CANONICAL_FIELDS}
    correct_keys = {
        k for k, v in field_eval["per_field"].items()
        if v["status"] in ("exact", "close")
    }

    precision = len(correct_keys) / len(pred_keys) if pred_keys else 0.0
    recall = len(correct_keys) / len(gt_keys) if gt_keys else 0.0
    detection_rate = len(pred_keys & gt_keys) / len(gt_keys) if gt_keys else 0.0

    gt_serving = gt_json.get("takaran_saji_g")
    pred_serving = hasil.get("takaran_saji_g")
    serving_ok = False
    if gt_serving is not None and pred_serving is not None:
        serving_ok = abs(float(pred_serving) - float(gt_serving)) <= max(1.0, float(gt_serving) * 0.05)

    return {
        "field_accuracy": round(field_eval["field_accuracy"], 4),
        "exact_or_close_fields": field_eval["benar"],
        "total_fields": field_eval["total_fields"],
        "nutrition_precision": round(precision, 4),
        "nutrition_recall": round(recall, 4),
        "nutrition_detection_rate": round(detection_rate, 4),
        "serving_size_correct": serving_ok,
        "missing_fields": field_eval["missing_fields"],
        "wrong_fields": field_eval["wrong_fields"],
        "per_field": field_eval["per_field"],
    }


def _collect_images(folder_path: Path, recursive: bool) -> list[Path]:
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    foto_list: list[Path] = []
    for pattern in patterns:
        foto_list.extend(folder_path.glob(pattern) if not recursive else folder_path.rglob(pattern))
    # Abaikan artefak debug / hasil evaluasi
    foto_list = [
        p for p in foto_list
        if "_gt." not in p.name
        and "evaluasi_hasil" not in p.name
        and "debug" not in p.parts
    ]
    return sorted(set(foto_list), key=lambda p: str(p).lower())


def evaluasi_batch(folder: str, recursive: bool = False, gt_only: bool = False):
    folder_path = Path(folder)
    foto_list = _collect_images(folder_path, recursive)
    if gt_only:
        foto_list = [p for p in foto_list if p.with_name(p.stem + "_gt.json").exists()]
    foto_list = foto_list[:10]

    hasil_semua = []
    skipped_no_gt = 0
    cer_list, wer_list = [], []
    field_acc_list, precision_list, recall_list, detect_list = [], [], [], []
    serving_ok_list = []

    print(f"\nFolder: {folder}")
    if gt_only:
        print(f"Mode: hanya foto dengan *_gt.json")
    print(f"Ditemukan: {len(foto_list)} foto untuk dievaluasi\n")
    print(f"{'No':>3}  {'File':30s}  {'Status':12s}  {'Field':>7}  {'Prec':>6}  {'Recall':>6}  {'Serv':>5}")
    print("-" * 82)

    for i, foto_path in enumerate(foto_list, 1):
        gt_json, gt_text = load_ground_truth(foto_path)
        if gt_json is None and gt_text is None:
            skipped_no_gt += 1
            if not gt_only:
                print(f"{i:>3}  {foto_path.name:30s}  {'skip (no GT)':12s}")
            continue

        try:
            hasil = jalankan_ocr(str(foto_path), gt_text)
            status = hasil["status"]
            detail = {
                "file": foto_path.name,
                "status": status,
                "quality": hasil.get("quality", {}),
                "roi": hasil.get("roi", {}),
                "semantic_filter": hasil.get("semantic_filter", {}),
            }

            if gt_text and hasil.get("evaluasi"):
                detail["cer_secondary"] = hasil["evaluasi"].get("cer")
                detail["wer_secondary"] = hasil["evaluasi"].get("wer")
                cer_list.append(detail["cer_secondary"])
                wer_list.append(detail["wer_secondary"])

            if gt_json:
                metrics = hitung_nutrition_metrics(gt_json, hasil)
                detail["nutrition_metrics"] = metrics
                field_acc_list.append(metrics["field_accuracy"])
                precision_list.append(metrics["nutrition_precision"])
                recall_list.append(metrics["nutrition_recall"])
                detect_list.append(metrics["nutrition_detection_rate"])
                serving_ok_list.append(metrics["serving_size_correct"])
                print(f"{i:>3}  {foto_path.name:30s}  {status:12s}  "
                      f"{metrics['field_accuracy']*100:6.1f}%  "
                      f"{metrics['nutrition_precision']*100:5.1f}%  "
                      f"{metrics['nutrition_recall']*100:5.1f}%  "
                      f"{'OK' if metrics['serving_size_correct'] else 'MISS':>5s}")
            else:
                print(f"{i:>3}  {foto_path.name:30s}  {status:12s}  {'text-only':>7s}")

            hasil_semua.append(detail)
        except Exception as e:
            print(f"{i:>3}  {foto_path.name:30s}  {'ERROR':12s}  ({e})")

    print("-" * 82)
    if not hasil_semua:
        print("\nTidak ada file dengan ground truth yang valid.")
        return

    ringkasan = {"total": len(hasil_semua), "skipped_no_gt": skipped_no_gt}
    print("\nRINGKASAN EVALUASI NUTRITION EXTRACTION")
    if field_acc_list:
        ringkasan.update({
            "avg_field_accuracy": round(sum(field_acc_list) / len(field_acc_list), 4),
            "avg_nutrition_precision": round(sum(precision_list) / len(precision_list), 4),
            "avg_nutrition_recall": round(sum(recall_list) / len(recall_list), 4),
            "avg_detection_rate": round(sum(detect_list) / len(detect_list), 4),
            "serving_size_accuracy": round(sum(1 for x in serving_ok_list if x) / len(serving_ok_list), 4),
        })
        ringkasan["target_field_accuracy_85"] = ringkasan["avg_field_accuracy"] >= 0.85
        print(f"   Field Accuracy      : {ringkasan['avg_field_accuracy']*100:.1f}%")
        print(f"   Nutrition Precision : {ringkasan['avg_nutrition_precision']*100:.1f}%")
        print(f"   Nutrition Recall    : {ringkasan['avg_nutrition_recall']*100:.1f}%")
        print(f"   Detection Rate      : {ringkasan['avg_detection_rate']*100:.1f}%")
        print(f"   Serving Size Acc.   : {ringkasan['serving_size_accuracy']*100:.1f}%")

    if cer_list:
        ringkasan["avg_cer_secondary"] = round(sum(cer_list) / len(cer_list), 4)
        ringkasan["avg_wer_secondary"] = round(sum(wer_list) / len(wer_list), 4)
        print(f"   CER sekunder        : {ringkasan['avg_cer_secondary']*100:.1f}%")
        print(f"   WER sekunder        : {ringkasan['avg_wer_secondary']*100:.1f}%")

    output = {"ringkasan": ringkasan, "detail": hasil_semua}
    out_path = folder_path / "evaluasi_hasil.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nDetail tersimpan: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluasi nutrition extraction NutriLabel v3")
    parser.add_argument("--folder", "-f", required=True, help="Folder berisi foto + ground truth")
    parser.add_argument(
        "--recursive", "-r", action="store_true",
        help="Cari foto di semua subfolder (mis. Dataset/)",
    )
    parser.add_argument(
        "--gt-only", action="store_true",
        help="Hanya evaluasi foto yang punya file *_gt.json",
    )
    args = parser.parse_args()
    evaluasi_batch(args.folder, recursive=args.recursive, gt_only=args.gt_only)
