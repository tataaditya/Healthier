import json
import argparse
from pathlib import Path
from difflib import SequenceMatcher
import time

try:
    from healthier_v3_ppocr import run_ocr
except ImportError:
    print("⚠️ Gagal mengimpor run_ocr dari healthier_v3_ppocr.py")
    run_ocr = None

def calculate_similarity(a, b):
    """Menghitung persentase kemiripan string menggunakan difflib (Sangat Longgar)"""
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

def evaluate_ocr(image_path, ground_truth=None):
    """
    Evaluasi OCR dengan metrik "Sangat Longgar" (menerima kesalahan ejaan / Levenshtein Distance).
    """
    print(f"\n🚀 Menguji OCR pada gambar: {image_path.name}")
    start_time = time.time()
    
    # 1. Eksekusi OCR
    if run_ocr:
        # Panggil fungsi run_ocr dari sistem utama
        try:
            result = run_ocr(str(image_path))
        except Exception as e:
            result = {"error": str(e)}
    else:
        # Simulasi jika gagal import
        result = {"Kalori": 150, "Protein": "5g", "Lemak": "2g", "Gula": "10g"}
        
    waktu_eksekusi = time.time() - start_time
    print(f"⏱️ Waktu Eksekusi OCR: {waktu_eksekusi:.2f} detik")
    
    # 2. Jika tidak ada ground truth, kita simulasikan untuk testing
    if not ground_truth:
        print("⚠️ Tidak ada ground truth yang diberikan. Menggunakan simulasi untuk pengujian metrik.")
        ground_truth = {
            "energi_total": "150 kkal",
            "lemak_total": "2 g",
            "protein": "5 g",
            "karbohidrat_total": "20 g",
            "gula": "10 g"
        }
        
    print("\n📊 HASIL EKSTRAKSI TEKS (OCR):")
    print(json.dumps(result, indent=2))
    
    print("\n✅ GROUND TRUTH (Nilai Target):")
    print(json.dumps(ground_truth, indent=2))
    
    # 3. Evaluasi String Matching (Sangat Longgar)
    print("\n🏆 EVALUASI MATRIKS AKURASI (SANGAT LONGGAR - STRING MATCHING)")
    print("═" * 70)
    print(f"{'Field':<20} | {'OCR Teks':<15} | {'Ground Truth':<15} | {'Kemiripan (%)':<10}")
    print("─" * 70)
    
    total_similarity = 0
    matched_fields = 0
    
    # Ekstraksi values dari result (asumsikan format dictionary dari OCR)
    ocr_values = {}
    if isinstance(result, dict) and "data" in result:
        ocr_values = result["data"]
    elif isinstance(result, dict):
        ocr_values = result
        
    for key, gt_val in ground_truth.items():
        # Cari nilai OCR yang paling mendekati kunci (fuzzy match key)
        best_match_key = None
        best_match_score = 0
        
        for ocr_key in ocr_values.keys():
            if ocr_key == "error": continue
            score = calculate_similarity(key, ocr_key)
            if score > best_match_score:
                best_match_score = score
                best_match_key = ocr_key
                
        ocr_val = ocr_values.get(best_match_key, "") if best_match_key else ""
        
        # Hitung kemiripan nilai
        similarity = calculate_similarity(str(ocr_val), str(gt_val)) * 100
        
        total_similarity += similarity
        matched_fields += 1
        
        # Longgarkan penilaian: kemiripan > 60% dianggap BENAR (True Positive)
        status = "✅ Lulus" if similarity > 60 else "❌ Gagal"
        
        print(f"{key:<20} | {str(ocr_val)[:15]:<15} | {str(gt_val)[:15]:<15} | {similarity:>6.1f}% {status}")
        
    print("═" * 70)
    avg_accuracy = total_similarity / max(matched_fields, 1)
    
    print(f"\n🎯 Akurasi Total OCR (Fuzzy Matching): {avg_accuracy:.2f}%")
    print(f"💡 Metrik Kelonggaran: Toleransi kesalahan karakter hingga 40%. Cocok untuk kemasan kusut/kabur.")

if __name__ == "__main__":
    dummy_img = Path(__file__).parent.parent / "assets" / "img" / "dummy.jpg"
    if dummy_img.exists():
        evaluate_ocr(dummy_img)
    else:
        print(f"❌ Gambar {dummy_img.name} tidak ditemukan di {dummy_img}. Silakan tambahkan foto kemasan.")
