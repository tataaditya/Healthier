"""
NutriLabel v3 - OCR Pipeline menggunakan PP-OCRv4
===================================================
Jalankan via: python nutrilabel_v3_ppocr.py --image foto_kemasan.jpg

Hardware target: i5-12450H + RTX 3050 6GB + RAM 16GB
Engine: PaddleOCR (PP-OCRv4) â€” ringan, cepat, stabil
"""

import os, sys, json, time, argparse, re, unicodedata, tempfile
import torch  # <-- Fix WinError 127 (Load torch DLLs before paddle)
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# â”€â”€â”€ 1. CEK DEPENDENCIES â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def cek_dependencies():
    """Pastikan semua library tersedia sebelum jalan."""
    deps = {
        "paddle": "pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/",
        "paddleocr": "pip install paddleocr>=2.8.0",
        "cv2": "pip install opencv-python",
        "numpy": "pip install numpy",
        "PIL": "pip install Pillow",
    }
    kurang = []
    for modul, instruksi in deps.items():
        try:
            __import__(modul)
        except ImportError:
            kurang.append((modul, instruksi))
    if kurang:
        print("âŒ Library berikut belum terinstal:")
        for m, i in kurang:
            print(f"   {m:15s} â†’ {i}")
        sys.exit(1)
    print("âœ… Semua dependencies tersedia.")

cek_dependencies()

from paddleocr import PaddleOCR

# â”€â”€â”€ 2. KONFIGURASI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

CONF_MINIMUM = 0.3  # Confidence threshold untuk filter hasil OCR
OCR_VERSION = os.getenv("NUTRILABEL_OCR_VERSION", "PP-OCRv4").strip() or None
OCR_DEVICE = "cpu"
USE_TABLE_STRUCTURE = True
REMOVE_WATERMARK = os.getenv("NUTRILABEL_REMOVE_WATERMARK", "0").strip() != "0"

KAMUS_NUTRISI = {
    # Energi / Kalori â€” termasuk variasi bilingual kemasan Indonesia
    "energi": "kalori", "kalori": "kalori", "energy": "kalori",
    "calories": "kalori", "kkal": "kalori", "kcal": "kalori",
    "energi total": "kalori", "total energy": "kalori",
    "energi total/total energy": "kalori",
    "energi total/total calories": "kalori",
    "energi total total calories": "kalori",
    "total calories": "kalori",
    "total energy/total calories": "kalori",
    "jumlah energi": "kalori",
    "energi dari lemak": None,
    "calories from fat": None,
    "energy from fat": None,
    "energi dari lemak jenuh": None,
    "calories from saturated fat": None,
    # Lemak Total
    "lemak total": "lemak_total", "lemaktotal": "lemak_total",
    "total fat": "lemak_total", "total lemak": "lemak_total",
    "lemak": "lemak_total", "fat": "lemak_total",
    "lemak total/total fat": "lemak_total",
    "emak total/total fat": "lemak_total",  # OCR sering miss huruf pertama
    "total lemak/ total fat": "lemak_total",
    "total lemak/total fat": "lemak_total",
    # Lemak Jenuh
    "lemak jenuh": "lemak_jenuh", "saturated fat": "lemak_jenuh",
    "lemakjenuh": "lemak_jenuh", "sat fat": "lemak_jenuh",
    "lemak jenuh/saturated fat": "lemak_jenuh",
    # Lemak Trans
    "lemak trans": "lemak_trans", "trans fat": "lemak_trans",
    "lemaktrans": "lemak_trans",
    "lemak trans/trans fat": "lemak_trans",
    # Kolesterol
    "kolesterol": "kolesterol", "cholesterol": "kolesterol",
    "kolesterol/cholesterol": "kolesterol",
    "cholesterol/kolesterol": "kolesterol",
    "kwstyol/cholesternl": "kolesterol",
    "cholesternl": "kolesterol",
    # Natrium / Garam
    "natrium": "natrium", "sodium": "natrium",
    "garam": "natrium", "salt": "natrium",
    "garam (natrium)": "natrium", "garam(natrium)": "natrium",
    "garam (natrium)/sodium": "natrium",
    "natrium/sodium": "natrium",
    "sodium/natrium": "natrium",
    "garam(natrium)/sodium": "natrium",
    "salt (sodium)": "natrium",
    "salt(sodium)": "natrium",
    # Karbohidrat
    "karbohidrat": "karbohidrat", "karbohidrat total": "karbohidrat",
    "total karbohidrat": "karbohidrat", "carbohydrate": "karbohidrat",
    "carb": "karbohidrat", "total carbohydrates": "karbohidrat",
    "total carbohydrate": "karbohidrat",
    "karbohidrat total/total carbohydrates": "karbohidrat",
    "karbohidrat total/total carbohydrate": "karbohidrat",
    "total carbohydrate/karbohidrat total": "karbohidrat",
    "karsohidrat total": "karbohidrat", "karsohidrat": "karbohidrat",
    # Serat
    "serat": "serat", "serat pangan": "serat",
    "dietary fiber": "serat", "fiber": "serat", "serat makanan": "serat",
    "serat pangan/dietary fiber": "serat",
    # Gula
    "gula": "gula", "gula total": "gula", "total sugars": "gula",
    "sugar": "gula", "sugars": "gula",
    "gula/sugars": "gula",
    "sugars/gula": "gula",
    "total sugars/gula total": "gula",
    # Protein
    "protein": "protein", "procein": "protein",
    # Vitamin & Mineral
    "vitamin a": "vitamin_a", "vitamin c": "vitamin_c",
    "vitamin d": "vitamin_d", "kalsium": "kalsium",
    "calcium": "kalsium", "zat besi": "zat_besi", "iron": "zat_besi",
}

VALIDASI_KLINIS = {
    "kalori": (0, 900), "lemak_total": (0, 80), "lemak_jenuh": (0, 40),
    "lemak_trans": (0, 10), "kolesterol": (0, 300), "natrium": (0, 2400),
    "karbohidrat": (0, 150), "serat": (0, 25), "gula": (0, 100),
    "protein": (0, 60),
}

NUTRITION_ANCHOR_TERMS = [
    "informasi nilai gizi", "nutrition facts", "nilai gizi", "takaran saji",
    "serving size", "jumlah per sajian", "amount per serving", "%akg", "%dv",
    "energi", "calories", "lemak", "fat", "protein", "karbohidrat",
    "carbohydrate", "gula", "sugar", "natrium", "sodium", "garam",
    "kolesterol", "cholesterol", "serat", "fiber", "per sajian", "per serving",
    "per 100", "per kemasan",
]

NON_NUTRITION_TERMS = [
    "komposisi", "ingredients", "petunjuk", "cara memasak", "goreng",
    "rebus", "kukus", "sajikan", "nikmati", "simpan", "storage",
    "didistribusikan", "distributed", "diproduksi", "best before",
    "baik digunakan", "prod.code", "barcode", "www.", ".com",
]

# â”€â”€â”€ 3. PP-OCR ENGINE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class PPOCREngine:
    """Singleton wrapper untuk PaddleOCR PP-OCRv4."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def muat(self):
        """Inisialisasi PaddleOCR engine. Hanya sekali per session."""
        if self._loaded:
            return
        print("\nðŸ“¦ Memuat PP-OCRv4 engine...")
        versi_label = OCR_VERSION or "default"
        print(f"   [Config] OCR version={versi_label}, device={OCR_DEVICE}")
        t0 = time.time()
        ocr_kwargs = {
            "use_textline_orientation": True,
            "lang": "en",
            "use_gpu": False,
            "text_det_thresh": 0.3,
            "text_det_box_thresh": 0.5,
        }
        # PaddleOCR 3.5 uses these parameter names. OCR_VERSION remains optional
        # so v4/v5 can be switched from the environment without touching code.
        if OCR_VERSION:
            ocr_kwargs["ocr_version"] = OCR_VERSION
        self.ocr = PaddleOCR(**ocr_kwargs)
        print(f"   âœ… Engine OCR siap dalam {time.time()-t0:.1f} detik")
        
        print("ðŸ“¦ Memuat PP-Structure (SLANet)...")
        if USE_TABLE_STRUCTURE:
            try:
                from paddlex import create_pipeline
                self.table_engine = create_pipeline(pipeline="table_recognition")
                print("   [PP-Structure] siap")
            except Exception as e:
                print(f"   [PP-Structure] gagal dimuat: {e}")
                self.table_engine = None
            
        else:
            print("   [PP-Structure] Dinonaktifkan via NUTRILABEL_USE_TABLE_STRUCTURE=0")
            self.table_engine = None
        self._loaded = True

    def deteksi(self, img_input) -> dict:
        """Jalankan OCR pada gambar (numpy array atau path string).
        Return dict: {"deteksi_list": [...], "html_tables": [...] }
        """
        html_tables = []
        if self.table_engine is not None:
            try:
                t0 = time.time()
                results_table = self.table_engine.predict(input=img_input)
                for res in results_table:
                    if hasattr(res, 'html') and isinstance(res.html, dict):
                        html_tables.extend(res.html.values())
                print(f"   âœ… PP-Structure selesai dalam {time.time()-t0:.2f} detik")
            except Exception as e:
                print(f"   âš ï¸ Error PP-Structure: {e}")

        # PaddleOCR 2.8.1 menggunakan ocr() (predict() hanya untuk versi 3.x/PaddleX)
        results = self.ocr.ocr(img_input, cls=True)

        if not results or not results[0]:
            return {"deteksi_list": [], "html_tables": html_tables}

        res = results[0]
        deteksi_list = []

        for line in res:
            if not line or len(line) < 2:
                continue
            bbox_points = line[0]  
            teks = line[1][0]
            conf = line[1][1]

            if conf < CONF_MINIMUM:
                continue

            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            x_min, y_min = min(xs), min(ys)
            x_max, y_max = max(xs), max(ys)

            deteksi_list.append({
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "text": teks,
                "conf": round(float(conf), 4),
                "center_x": (x_min + x_max) / 2,
                "center_y": (y_min + y_max) / 2,
            })

        deteksi_list.sort(key=lambda d: (d["center_y"], d["center_x"]))
        return {"deteksi_list": deteksi_list, "html_tables": html_tables}


# â”€â”€â”€ 4. REKONSTRUKSI TABEL DARI BBOX â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

import html.parser

def ekstrak_takaran_saji(deteksi_list: list[dict]) -> float | None:
    """Cari takaran saji (serving size) dalam gram dari hasil OCR."""
    def is_label_takaran(teks: str) -> bool:
        return (
            "serving size" in teks
            or "takaran saji" in teks
            or "saji/serving" in teks
            or ("saji" in teks and "sajian" not in teks)
        )

    for i, det in enumerate(deteksi_list):
        teks = det["text"].lower()
        if is_label_takaran(teks):
            # Coba cari angka di bbox yang sama
            m = re.search(r"(\d+[\.,]?\d*)\s*(?:g|ml|gram)", teks)
            if m:
                return float(m.group(1).replace(",", "."))
            
            # Jika tidak ada, cari di bbox berikutnya (split bbox)
            for j in range(i+1, min(i+4, len(deteksi_list))):
                teks_next = deteksi_list[j]["text"].lower()
                # Cek apakah Y-nya kurang lebih sama (beda < 30px) atau memang bersebelahan
                if abs(deteksi_list[j]["center_y"] - det["center_y"]) < 30 or j == i+1:
                    m2 = re.search(r"(\d+[\.,]?\d*)\s*(?:g|ml|gram)", teks_next)
                    if m2:
                        return float(m2.group(1).replace(",", "."))

    # Fallback dua arah untuk tabel bilingual: kadang nilai "100g" berada
    # sedikit di atas/sebelum bbox label "Takaran Saji/Serving Size".
    for det in deteksi_list:
        teks = det["text"].lower()
        if not is_label_takaran(teks):
            continue
        tinggi = max(18, det["bbox"][3] * 1.8)
        kandidat = []
        for cand in deteksi_list:
            if cand is det:
                continue
            if abs(cand["center_y"] - det["center_y"]) > tinggi:
                continue
            teks_cand = cand["text"].lower()
            m = re.search(r"(\d+[\.,]?\d*)\s*(?:g|ml|gram)\b", teks_cand)
            if m:
                jarak = abs(cand["center_y"] - det["center_y"]) + abs(cand["center_x"] - det["center_x"]) * 0.05
                kandidat.append((jarak, float(m.group(1).replace(",", "."))))
        if kandidat:
            kandidat.sort()
            return kandidat[0][1]
    return None

class TableHTMLParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_td = False
        self.current_row = []
        self.rows = []
        self.current_data = []

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self.current_row = []
        elif tag == 'td' or tag == 'th':
            self.in_td = True
            self.current_data = []

    def handle_endtag(self, tag):
        if tag == 'tr':
            if self.current_row:
                self.rows.append(self.current_row)
        elif tag == 'td' or tag == 'th':
            self.in_td = False
            text = " ".join(self.current_data).strip()
            if text:
                self.current_row.append(text)

    def handle_data(self, data):
        if self.in_td:
            self.current_data.append(data)

def ekstrak_dari_html(html_str: str) -> dict:
    parser = TableHTMLParser()
    parser.feed(html_str)
    
    hasil = {}
    for row in parser.rows:
        if len(row) < 2:
            continue
            
        label_raw = row[0]
        nilai_raw = None
        for cell in row[1:]:
            if re.search(r'\d', cell):
                nilai_raw = cell
                break
                
        if label_raw and nilai_raw:
            key, info = ParseOCR._parse_baris_nutrisi(label_raw, nilai_raw)
            if key and info:
                # Coba cari AKG dari cell lain jika belum ada
                if info["akg_persen"] is None:
                    for cell in row[1:]:
                        if '%' in cell and cell != nilai_raw:
                            info["akg_persen"] = ParseOCR.ekstrak_persen_akg(cell)
                            break
                hasil[key] = info
                
    return hasil


def _normalisasi_digit_ocr(teks: str) -> str:
    """Ubah huruf yang sering terbaca sebagai digit pada konteks angka."""
    return (
        teks.replace("O", "0").replace("o", "0")
            .replace("C", "0").replace("c", "0")
            .replace("S", "5").replace("s", "5")
            .replace("I", "1").replace("l", "1")
    )


# Pola header kolom tabel gizi bilingual (Indonesia + Inggris)
_SERVING_HDR = (
    r"per\s*sajian", r"per\s*serving", r"amount\s*per\s*serving",
    r"jumlah\s*per\s*sajian", r"umlah\s*per\s*sajian", r"umiah\s*per\s*sajian",
    r"per\s*sajian/serving",
)
_PER100_HDR = (r"per\s*100\s*g", r"per\s*100g", r"per100")
_DV_HDR = (r"%akg", r"%dv", r"akg\s*/\s*dv", r"daily\s*value")
_IGNORE_ROW_LABELS = (
    "monounsaturated", "polyunsaturated", "tunggal", "ganda", "unsaturated",
    "energy from", "energi dari", "calories from", "kkal dari",
)


def normalisasi_teks_nilai_ocr(teks: str) -> str:
    """Normalisasi noise OCR umum pada nilai nutrisi (g dibaca 9, og, dll)."""
    if not teks:
        return ""
    t = unicodedata.normalize("NFKC", teks).strip().lower()
    t = t.replace(",", ".")
    t = (
        t.replace("omg", "0mg").replace("0 mg", "0mg")
        .replace("og", "0g").replace("0 g", "0g")
        .replace("egram", "1gram").replace(" q ", " g ")
    )
    # 8 g / 1 g sering menjadi 89 / 19 tanpa spasi
    t = re.sub(r"(\d)\s*9\b", r"\1g", t)
    t = re.sub(r"(\d)9$", r"\1g", t)
    t = re.sub(r"(\d)9(?=%)", r"\1g", t)
    compact = re.sub(r"\s+", "", t)
    if re.fullmatch(r"\d{1,3}9", compact) and "g" not in compact and "mg" not in compact:
        compact = compact[:-1] + "g"
    if re.fullmatch(r"\d{1,2}9", compact):
        compact = compact[:-1] + "g"
    if "g" in compact or "mg" in compact or "kkal" in compact or "kcal" in compact:
        return compact
    return t


def _header_kind(teks: str) -> str | None:
    """Klasifikasi bbox header kolom tabel (bukan takaran saji / footnote)."""
    norm = ParseOCR.bersihkan_teks(teks)
    # Baris takaran saji: "Serving Size 20.4g" — bukan header kolom nilai
    if re.search(r"takaran\s*saji", norm):
        return None
    if re.search(r"serving\s*size", norm) and not re.search(
        r"per\s*(sajian|serving)|amount\s*per", norm
    ):
        return None
    if re.search(r"sajian\s*per\s*kemasan|servings\s*per\s*pack", norm):
        return None
    if any(re.search(p, norm) for p in _PER100_HDR):
        return "per100"
    if any(re.search(p, norm) for p in _DV_HDR):
        return "dv"
    if any(re.search(p, norm) for p in _SERVING_HDR):
        if not re.search(r"per\s*100", norm):
            return "serving"
    return None


def deteksi_layout_kolom(deteksi_list: list[dict]) -> dict:
    """Deteksi batas kolom Per Sajian vs %AKG vs Per 100g dari header OCR."""
    if not deteksi_list:
        return {"mode": "single", "detected": False}

    image_width = float(max(
        (float(d["bbox"][0]) + float(d["bbox"][2]) for d in deteksi_list if d.get("bbox")),
        default=1.0,
    ))
    headers = []
    for det in deteksi_list:
        kind = _header_kind(det.get("text", ""))
        if kind:
            headers.append({"kind": kind, "x": float(det["center_x"]), "text": det["text"]})

    if not headers:
        # Heuristik: cluster bbox bernilai numerik menjadi 2-3 kolom
        numeric_x = [
            float(d["center_x"]) for d in deteksi_list
            if re.search(r"\d", d.get("text", ""))
            and "%" not in d.get("text", "")
            and ParseOCR.fuzzy_match_nutrisi(d.get("text", "")) is None
        ]
        if len(numeric_x) < 8:
            return {"mode": "single", "detected": False, "image_width": image_width}
        xs = sorted(numeric_x)
        gaps = [(xs[i + 1] - xs[i], i) for i in range(len(xs) - 1)]
        gaps.sort(reverse=True)
        if not gaps or gaps[0][0] < image_width * 0.12:
            return {"mode": "single", "detected": False, "image_width": image_width}
        split_idx = gaps[0][1]
        split_x = (xs[split_idx] + xs[split_idx + 1]) / 2.0
        return {
            "mode": "multi_heuristic",
            "detected": True,
            "image_width": image_width,
            "serving_x0": 0.0,
            "serving_x1": split_x,
            "per100_x0": split_x,
            "per100_x1": image_width,
            "dv_x0": None,
            "dv_x1": None,
        }

    by_kind = {"serving": [], "dv": [], "per100": []}
    for h in headers:
        by_kind[h["kind"]].append(h["x"])

    # Hanya header yang memuat "per sajian/serving" (bukan serving size produk)
    serving_xs = [
        h["x"] for h in headers
        if h["kind"] == "serving"
        and re.search(r"per\s*(sajian|serving)|amount\s*per\s*serving|jumlah\s*per|umlah\s*per", ParseOCR.bersihkan_teks(h["text"]))
    ]
    serving_x = float(np.median(serving_xs)) if serving_xs else None
    if serving_x is None and by_kind["serving"]:
        serving_x = float(np.median(by_kind["serving"]))

    if serving_x is not None and serving_x < image_width * 0.35:
        serving_x = None

    dv_x = float(np.median(by_kind["dv"])) if by_kind["dv"] else None
    per100_x = float(np.median(by_kind["per100"])) if by_kind["per100"] else None

    ordered = [(x, k) for k, xs in by_kind.items() for x in xs]
    ordered.sort(key=lambda t: t[0])

    if serving_x is None:
        # Heuristic fallback: cari nilai numerik paling kiri yang bukan label nutrisi
        numeric_x = sorted(
            float(d["center_x"]) for d in deteksi_list
            if re.search(r"\d", d.get("text", "")) and d["center_x"] > image_width * 0.25 and ParseOCR.fuzzy_match_nutrisi(d.get("text", "")) is None
        )
        if numeric_x:
            serving_x = numeric_x[0]

    if serving_x is None:
        return {"mode": "single", "detected": False, "image_width": image_width}

    cols_right = [x for x, k in ordered if x > serving_x + image_width * 0.02]
    if dv_x is None and cols_right:
        dv_candidates = [x for x in cols_right if x < (per100_x or image_width) - image_width * 0.05]
        if dv_candidates:
            dv_x = float(np.median(dv_candidates))

    half_w = image_width * 0.09
    if per100_x is not None or dv_x is not None:
        cols_after = [x for x in [per100_x, dv_x] if x is not None and x > serving_x]
        if cols_after:
            next_x = min(cols_after)
            serving_x1 = (serving_x + next_x) / 2.0
        else:
            serving_x1 = serving_x + image_width * 0.22
            
        dv_x0 = dv_x1 = per100_x0 = per100_x1 = None
        if per100_x is not None:
            per100_x0 = serving_x1
            per100_x1 = image_width
        if dv_x is not None:
            dv_x0 = serving_x1
            dv_x1 = image_width
    else:
        serving_x1 = serving_x + image_width * 0.22
        dv_x0 = dv_x1 = per100_x0 = per100_x1 = None

    serving_x0 = max(0.0, serving_x - half_w)
    serving_x1 = max(serving_x0 + image_width * 0.06, serving_x1)

    layout = {
        "mode": "multi",
        "detected": True,
        "image_width": image_width,
        "serving_x0": serving_x0,
        "serving_x1": serving_x1,
        "dv_x0": dv_x0,
        "dv_x1": dv_x1,
        "per100_x0": per100_x0 if per100_x else None,
        "per100_x1": per100_x1 if per100_x else None,
        "headers": headers,
    }
    return layout


def _in_x_band(center_x: float, x0: float | None, x1: float | None) -> bool:
    if x0 is None or x1 is None:
        return False
    return x0 <= center_x <= x1


def _row_should_skip(label_text: str) -> bool:
    norm = ParseOCR.bersihkan_teks(label_text)
    if any(term in norm for term in _IGNORE_ROW_LABELS):
        return True
    if ParseOCR.fuzzy_match_nutrisi(norm) is None and not re.search(
        r"(energi|energy|kalori|calories|lemak|fat|protein|karbohidrat|carb|gula|sugar|natrium|sodium|garam|serat|fiber)",
        norm,
    ):
        return True
    return False


def _build_rows(deteksi_list: list[dict], row_tol: float) -> list[dict]:
    rows = []
    for det in sorted(deteksi_list, key=lambda d: (d["center_y"], d["center_x"])):
        if not rows or abs(rows[-1]["center_y"] - det["center_y"]) > row_tol:
            rows.append({"center_y": det["center_y"], "items": [det]})
        else:
            rows[-1]["items"].append(det)
            rows[-1]["center_y"] = float(np.median([x["center_y"] for x in rows[-1]["items"]]))
    return rows


def _pecah_baris_multi_label(rows: list[dict], median_h: float) -> list[dict]:
    """Pisah baris yang berisi lebih dari satu label nutrisi (mis. Protein + Lemak Jenuh)."""
    hasil = []
    y_tol = max(14.0, median_h * 1.15)

    for row in rows:
        items = row["items"]
        label_anchors = []
        for det in items:
            key = ParseOCR.fuzzy_match_nutrisi(det.get("text", ""))
            if key and not _is_value_bbox(det.get("text", "")):
                label_anchors.append((float(det["center_y"]), key, det))

        if len(label_anchors) <= 1:
            hasil.append(row)
            continue

        label_anchors.sort(key=lambda t: t[0])
        for anchor_y, _key, anchor_det in label_anchors:
            sub_items = []
            for det in items:
                if abs(float(det["center_y"]) - anchor_y) <= y_tol:
                    sub_items.append(det)
            if sub_items:
                hasil.append({
                    "center_y": anchor_y,
                    "items": sub_items,
                    "anchor_key": _key,
                })
    return hasil


def _pilih_nilai_untuk_label(label_text: str, value_parts: list[dict], prefer_min_kkal: bool = False) -> tuple[str, dict, str] | None:
    """Coba setiap bbox nilai; ambil yang cocok dengan label dan validasi klinis."""
    label_key = ParseOCR.fuzzy_match_nutrisi(label_text)
    terbaik = None
    skor_terbaik = -1e9

    for det in sorted(value_parts, key=lambda d: d["center_x"]):
        serving_text = normalisasi_teks_nilai_ocr(det["text"])
        key, info = ParseOCR._parse_baris_nutrisi(label_text, serving_text)
        if not key or not info:
            continue
        if label_key and key != label_key:
            continue
        min_v, max_v = VALIDASI_KLINIS.get(key, (0, 9999))
        val = info.get("per_sajian")
        if val is None or val > max_v * 1.25:
            continue
        skor = _score_value_bbox(det, prefer_min_kkal)
        if label_key == key:
            skor += 20.0
        if terbaik is None or skor > skor_terbaik:
            skor_terbaik = skor
            terbaik = (serving_text, info, key)

    if terbaik:
        return terbaik

    if prefer_min_kkal:
        kkal_vals = [_extract_kkal_value(d["text"]) for d in value_parts]
        kkal_vals = [v for v in kkal_vals if v is not None]
        if kkal_vals:
            st = f"{min(kkal_vals)} kkal"
            key, info = ParseOCR._parse_baris_nutrisi(label_text, st)
            if key and info:
                return st, info, key
    return None


def _is_value_bbox(teks: str) -> bool:
    norm = normalisasi_teks_nilai_ocr(teks)
    return bool(re.search(r"\d", norm))


def _score_value_bbox(det: dict, prefer_kkal_min: bool = False) -> float:
    t = normalisasi_teks_nilai_ocr(det.get("text", ""))
    score = 0.0
    if re.search(r"(g|mg|kkal|kcal)\b", t):
        score += 14.0
    if "%" in t:
        score -= 30.0
    if re.fullmatch(r"\d{1,2}", t):
        score -= 6.0
    kkal = _extract_kkal_value(t)
    if kkal is not None:
        score += 8.0
        if prefer_kkal_min:
            score -= kkal * 0.01
    return score


def rekonstruksi_tabel_kolom(deteksi_list: list[dict], layout: dict) -> dict:
    """Rekonstruksi per baris: ambil nilai per sajian = bbox nilai paling kiri sebelum kolom Per 100g."""
    if not layout.get("detected"):
        return {}

    image_width = float(layout.get("image_width", 1))
    serving_cutoff = float(layout.get("serving_x1", image_width * 0.55))

    bbox_heights = [max(1, d["bbox"][3]) for d in deteksi_list if d.get("bbox")]
    median_h = float(np.median(bbox_heights)) if bbox_heights else 20.0
    row_tol = max(10.0, median_h * 0.62)
    rows = _pecah_baris_multi_label(_build_rows(deteksi_list, row_tol), median_h)

    hasil = {}

    for row in rows:
        items = sorted(row["items"], key=lambda d: d["center_x"])
        label_parts = []
        value_parts = []
        dv_parts = []

        for det in items:
            teks = det.get("text", "")
            cx = float(det["center_x"])
            if "%" in teks and re.search(r"\d", teks):
                dv_parts.append(det)
                continue
            if _is_value_bbox(teks) and cx <= serving_cutoff:
                value_parts.append(det)
            elif not _is_value_bbox(teks) and cx < serving_cutoff * 0.95:
                label_parts.append(det)

        if not value_parts:
            continue

        label_text = " ".join(d["text"] for d in label_parts).strip()
        row_all = " ".join(d["text"] for d in items)
        if label_text and _row_should_skip(label_text):
            continue
        if _row_should_skip(row_all) and not label_text:
            continue

        prefer_min_kkal = "energi" in ParseOCR.bersihkan_teks(label_text or row_all)
        label_for_parse = label_text or row_all
        if row.get("anchor_key"):
            for det in label_parts:
                if ParseOCR.fuzzy_match_nutrisi(det.get("text", "")) == row["anchor_key"]:
                    label_for_parse = det["text"]
                    break

        picked = _pilih_nilai_untuk_label(label_for_parse, value_parts, prefer_min_kkal)
        if not picked:
            continue

        _serving_text, info, key = picked
        dv_text = " ".join(d["text"] for d in dv_parts)
        if info.get("akg_persen") is None and dv_text:
            info["akg_persen"] = ParseOCR.ekstrak_persen_akg(dv_text)
        if key not in hasil:
            hasil[key] = info
            print(f"     [Kolom-Saji] {key}: {info['per_sajian']} {info['satuan']}")

    return hasil


def _extract_kkal_value(teks: str) -> float | None:
    """Ambil nilai kkal dari OCR yang kadang menjadi 30Rkal/15Hkal/SRkal."""
    compact = re.sub(r"\s+", "", teks)
    compact = compact.replace(".", "")
    m = re.search(r"([0-9OoCcSsIl]{1,4})[A-Za-z]?(?:kkal|kcal|kal)", compact, re.IGNORECASE)
    if not m:
        return None
    raw = _normalisasi_digit_ocr(m.group(1))
    try:
        val = float(raw)
    except ValueError:
        return None
    if 0 <= val <= 900:
        return val
    return None


def _infer_fat_grams_from_kcal(kcal: float) -> float | None:
    """Estimasi gram lemak dari kalori lemak.

    Label gizi umumnya membulatkan energi lemak ke kelipatan 5 kkal.
    Kita cari gram 0.5-step paling kecil yang masih menjelaskan nilai kkal
    tersebut. Ini membuat 25 kkal -> 2.5 g, 15 kkal -> 1.5 g,
    10 kkal -> 1 g, dan 40 kkal -> 4.5 g.
    """
    if kcal is None or kcal <= 0:
        return None
    candidates = []
    for step in range(0, 161):  # 0.0g sampai 80.0g
        gram = step / 2.0
        energy = gram * 9.0
        if kcal - 2.5 <= energy < kcal + 2.5:
            candidates.append(gram)
    return candidates[0] if candidates else round(kcal / 9.0 * 2) / 2.0


def ekstrak_aux_energi_lemak(deteksi_list: list[dict]) -> dict:
    """Ekstrak sinyal bantu dari urutan kkal sebelum baris makro nutrisi.

    Banyak label Indonesia memiliki urutan:
    Energi Total, Energi dari Lemak, Energi dari Lemak Jenuh.
    Saat labelnya rusak oleh glare/watermark, angka kkal-nya sering masih
    terbaca. Sinyal ini dipakai sebagai fallback, bukan field nutrisi baru.
    """
    if not deteksi_list:
        return {}

    first_macro_y = None
    for det in deteksi_list:
        key = ParseOCR.fuzzy_match_nutrisi(det.get("text", ""))
        if key in {"lemak_total", "lemak_jenuh", "protein", "karbohidrat"}:
            first_macro_y = det["center_y"]
            break

    kkal_values = []
    for det in sorted(deteksi_list, key=lambda d: (d["center_y"], d["center_x"])):
        if first_macro_y is not None and det["center_y"] > first_macro_y + max(12, det["bbox"][3]):
            continue
        val = _extract_kkal_value(det.get("text", ""))
        if val is not None:
            kkal_values.append(val)

    aux = {}
    if kkal_values:
        aux["kalori"] = kkal_values[0]
    if len(kkal_values) >= 2:
        aux["lemak_total_dari_kkal"] = _infer_fat_grams_from_kcal(kkal_values[1])
        aux["energi_dari_lemak_kkal"] = kkal_values[1]
    if len(kkal_values) >= 3:
        aux["lemak_jenuh_dari_kkal"] = _infer_fat_grams_from_kcal(kkal_values[2])
        aux["energi_dari_lemak_jenuh_kkal"] = kkal_values[2]
    return aux


def terapkan_fallback_energi_lemak(data_nutrisi: dict, deteksi_list: list[dict]) -> tuple[dict, list]:
    """Lengkapi/koreksi field dari energi lemak jika OCR nilai gram gagal."""
    flags = []
    aux = ekstrak_aux_energi_lemak(deteksi_list)
    if not aux:
        return data_nutrisi, flags

    def put_or_correct(key: str, val: float, satuan: str, flag: str, overwrite_bad: bool = False):
        if val is None:
            return
        current = data_nutrisi.get(key)
        if current is None:
            data_nutrisi[key] = {"per_sajian": val, "satuan": satuan, "akg_persen": None}
            flags.append(flag)
            print(f"     [Fallback] {key}: {val} {satuan} ({flag})")
            return
        cur_val = current.get("per_sajian")
        if overwrite_bad and cur_val is not None:
            energy_key = "energi_dari_lemak_kkal" if key == "lemak_total" else "energi_dari_lemak_jenuh_kkal"
            kcal_ref = aux.get(energy_key)
            has_unit = bool(current.get("_had_unit"))
            plausible = False
            if kcal_ref is not None:
                plausible = abs(float(cur_val) * 9.0 - float(kcal_ref)) <= 7.0
            # Nilai gram dengan satuan eksplisit dan masih sesuai energi lemak
            # dipertahankan. Nilai tanpa satuan atau tidak konsisten dikoreksi.
            if has_unit and plausible:
                return
            if abs(float(cur_val) - float(val)) < 0.5 and plausible:
                return
            current["per_sajian"] = val
            current["satuan"] = satuan
            flags.append(flag + "_koreksi")
            print(f"     [Fallback] koreksi {key}: {cur_val} -> {val} {satuan}")

    put_or_correct("kalori", aux.get("kalori"), "kkal", "fallback_kalori_dari_kkal")

    fat_val = aux.get("lemak_total_dari_kkal")
    # Jika OCR mengambil angka %AKG sebagai gram, nilai dari energi lemak lebih
    # dapat dipercaya karena hubungan energinya deterministik.
    # Jangan koreksi lemak_total dari energi jika nilai gram sudah punya satuan eksplisit
    # dan masuk akal (hindari 4g -> 54g dari kolom salah).
    fat_overwrite = False
    if fat_val is not None:
        cur = data_nutrisi.get("lemak_total")
        if cur is None:
            fat_overwrite = True
        elif not cur.get("_had_unit") or cur.get("per_sajian", 0) > 25:
            fat_overwrite = True
    put_or_correct("lemak_total", fat_val, "g", "fallback_lemak_total_dari_energi", overwrite_bad=fat_overwrite)

    sat_val = aux.get("lemak_jenuh_dari_kkal")
    current_sat = data_nutrisi.get("lemak_jenuh")
    current_fat = data_nutrisi.get("lemak_total")
    sat_bad = False
    if current_sat and current_fat:
        sat_bad = current_sat.get("per_sajian", 0) > current_fat.get("per_sajian", 0)
    put_or_correct(
        "lemak_jenuh",
        sat_val,
        "g",
        "fallback_lemak_jenuh_dari_energi",
        overwrite_bad=sat_bad,
    )
    return data_nutrisi, flags


def rekonstruksi_tabel_dari_bbox(deteksi_list: list[dict], html_tables: list[str] = None) -> dict:
    """Rekonstruksi struktur tabel gizi: kolom Per Sajian, PP-Structure, lalu proximity."""
    hasil_akhir = {}
    hasil_source = {}

    # 0. Kolom Per Sajian (prioritas untuk tabel bilingual multi-kolom)
    layout = deteksi_layout_kolom(deteksi_list)
    use_serving_band = layout.get("detected") and layout.get("mode") in ("multi", "multi_heuristic")
    if layout.get("detected"):
        print(f"   [Kolom] mode={layout['mode']}, serving X: {layout['serving_x0']:.0f}-{layout['serving_x1']:.0f}")
        hasil_kolom = rekonstruksi_tabel_kolom(deteksi_list, layout)
        for k, v in hasil_kolom.items():
            hasil_akhir[k] = v
            hasil_source[k] = "column_serving"
        if len(hasil_akhir) >= 4:
            return hasil_akhir
    
    # 1. Coba dengan PP-Structure HTML
    if html_tables:
        print("   [Rekonstruksi] Menggunakan hasil PP-Structure (SLANet)")
        for html_str in html_tables:
            hasil_html = ekstrak_dari_html(html_str)
            for k, v in hasil_html.items():
                if k not in hasil_akhir:
                    hasil_akhir[k] = v
                    print(f"     âœ“ [HTML] {k}: {v['per_sajian']} {v['satuan']}")
            
        if len(hasil_akhir) >= 2:
            return hasil_akhir
            
        print("   [Rekonstruksi] PP-Structure gagal mengekstrak nutrisi yang cukup. Fallback...")
    else:
        print("   [Rekonstruksi] Menggunakan metode Spatial Proximity Fallback")

    # 2. Fallback: Enhanced Proximity Matching (jangan hapus hasil kolom parsial)
    kolom_partial = dict(hasil_akhir)
    hasil_akhir.clear()
    label_list = []
    value_list = []
    
    for i, det in enumerate(deteksi_list):
        det["original_index"] = i
        teks = det["text"].strip()
        teks_lower = teks.lower()
        
        teks_norm = normalisasi_teks_nilai_ocr(teks)
        det["teks_norm"] = teks_norm
        
        # Coba cek apakah ini single-line (Label + Angka + Satuan) dalam satu bbox
        m_single = re.match(r"^(.+?)\s*(\d+[\.,]?\d*\s*(?:g|mg|kkal|kcal|mcg|Âµg|%).*)$", teks_norm)
        if m_single:
            label_cand = m_single.group(1).strip()
            kunci = ParseOCR.fuzzy_match_nutrisi(label_cand)
            if kunci:
                nilai_cand = m_single.group(2)
                key, info = ParseOCR._parse_baris_nutrisi(label_cand, nilai_cand)
                if key and info:
                    if info["akg_persen"] is None:
                        m_akg = re.search(r"(\d+)\s*%", nilai_cand)
                        if m_akg:
                            info["akg_persen"] = float(m_akg.group(1))
                    if key not in hasil_akhir:
                        hasil_akhir[key] = info
                        print(f"     âœ“ [Single-Line] {key}: {info['per_sajian']} {info['satuan']}")
                continue # Skip classification karena sudah diekstrak
        
        # Jika ada angka dan satuan (termasuk yang udah dinormalisasi), ini value
        if re.search(r"\d+[\.,]?\d*\s*(g|mg|kkal|kcal|%)", teks_norm) or re.search(r"^\d+[\.,]?\d*$", teks_norm):
            # Simpan indeks agar bisa di-track penggunaannya
            det["original_index"] = i
            det["teks_norm"] = teks_norm
            value_list.append(det)
        else:
            # Coba match ke KAMUS
            kunci = ParseOCR.fuzzy_match_nutrisi(teks)
            if kunci:
                det["key_nutrisi"] = kunci
                label_list.append(det)
            elif re.match(r".*[a-zA-Z]+.*", teks):
                label_list.append(det)
                
    bbox_heights = [max(1, d["bbox"][3]) for d in deteksi_list if d.get("bbox")]
    median_h = float(np.median(bbox_heights)) if bbox_heights else 20.0
    image_width = max((d["bbox"][0] + d["bbox"][2] for d in deteksi_list if d.get("bbox")), default=1)
    image_height = max((d["bbox"][1] + d["bbox"][3] for d in deteksi_list if d.get("bbox")), default=1)
    is_wide_table = image_width > image_height * 1.25

    # Reconstruct text rows first. Ini lebih stabil untuk tabel bilingual lebar
    # karena label dan nilai sering terdeteksi sebagai bbox terpisah.
    rows = []
    row_tol = max(12.0, median_h * 0.85)
    for det in sorted(deteksi_list, key=lambda d: (d["center_y"], d["center_x"])):
        if not rows or abs(rows[-1]["center_y"] - det["center_y"]) > row_tol:
            rows.append({"center_y": det["center_y"], "items": [det]})
        else:
            rows[-1]["items"].append(det)
            rows[-1]["center_y"] = float(np.median([x["center_y"] for x in rows[-1]["items"]]))

    # Row-merge hanya dipakai jika kolom serving belum mengisi field tersebut
    for row in rows:
        items = sorted(row["items"], key=lambda d: d["center_x"])
        row_text = " ".join(item["text"] for item in items)
        key, info = ParseOCR.parse_baris_teks(row_text)
        if key and info and key not in hasil_akhir:
            hasil_akhir[key] = info
            hasil_source[key] = "row"
            print(f"     [Row] {key}: {info['per_sajian']} {info['satuan']}")

    # Urutkan label dari atas ke bawah untuk memprioritaskan yang di atas
    label_list.sort(key=lambda x: x["center_y"])
    used_values_idx = set()
                
    for label in label_list:
        kunci = label.get("key_nutrisi") or ParseOCR.fuzzy_match_nutrisi(label["text"])
        if not kunci:
            continue
            
        h_label = label["bbox"][3]  # Tinggi bbox label
        y_tolerance = max(18, h_label * 1.1, median_h * 1.2)
        if is_wide_table:
            y_tolerance *= 1.5
            
        # Cari semua value yang sebaris dan belum dipakai
        kandidat_values = []
        for val in value_list:
            if val["original_index"] in used_values_idx:
                continue
            if use_serving_band and "%" not in val.get("text", ""):
                if not _in_x_band(
                    float(val["center_x"]),
                    layout.get("serving_x0"),
                    layout.get("serving_x1"),
                ):
                    continue
            if abs(label["center_y"] - val["center_y"]) < y_tolerance:
                if val["center_x"] > label["center_x"]: # Harus di sebelah kanan
                    dx = val["center_x"] - label["center_x"]
                    val["_dx_from_label"] = dx
                    val["_dy_from_label"] = abs(label["center_y"] - val["center_y"])
                    kandidat_values.append(val)
                    
        if not kandidat_values:
            continue
            
        # Ambil nilai per-sajian terdekat. Kandidat yang terlalu jauh dipakai
        # hanya jika tidak ada opsi lain, supaya angka dari kolom/teks lain tidak
        # mudah terseret sebagai nilai nutrisi.
        batas_x = image_width * 0.60
        kandidat_dekat = [v for v in kandidat_values if v.get("_dx_from_label", 0) <= batas_x]
        if kandidat_dekat:
            kandidat_values = kandidat_dekat
        kandidat_values.sort(
            key=lambda x: (
                1 if "%" in x["text"] else 0,
                x.get("_dy_from_label", 0),
                x.get("_dx_from_label", x["center_x"]),
            )
        )
        val_terdekat = kandidat_values[0]
        nilai_terdekat = val_terdekat["teks_norm"] if "teks_norm" in val_terdekat else val_terdekat["text"]
        
        # Cari AKG
        akg_raw = ""
        val_akg = None
        for val in kandidat_values[1:]:
            if "%" in val["text"]:
                akg_raw = val["text"]
                val_akg = val
                break
                
        key, info = ParseOCR._parse_baris_nutrisi(label["text"], nilai_terdekat)
        if key and info:
            if info["akg_persen"] is None and akg_raw:
                info["akg_persen"] = ParseOCR.ekstrak_persen_akg(akg_raw)
            # Jangan timpa hasil kolom serving / row yang sudah valid
            if key not in hasil_akhir or hasil_source.get(key) not in ("column_serving",):
                hasil_akhir[key] = info
                hasil_source[key] = "proximity"
                # Tandai value sudah dipakai
                used_values_idx.add(val_terdekat["original_index"])
                if val_akg:
                    used_values_idx.add(val_akg["original_index"])
                print(f"     âœ“ [Proximity] {key}: {info['per_sajian']} {info['satuan']}")

    # Recovery pass yang lebih longgar untuk nilai 0g/Og yang sering sedikit
    # naik/turun dari labelnya pada plastik mengilap.
    for label in label_list:
        kunci = label.get("key_nutrisi") or ParseOCR.fuzzy_match_nutrisi(label["text"])
        if not kunci or kunci in hasil_akhir:
            continue
        if kunci not in {"gula", "lemak_trans", "kolesterol"}:
            continue

        kandidat_values = []
        for val in value_list:
            if val["original_index"] in used_values_idx:
                continue
            dy = abs(label["center_y"] - val["center_y"])
            if dy > max(36, median_h * 2.5):
                continue
            if val["center_x"] < label["center_x"] - image_width * 0.05:
                continue
            nilai_raw = val.get("teks_norm", val["text"])
            key, info = ParseOCR._parse_baris_nutrisi(label["text"], nilai_raw)
            if key == kunci and info:
                dx = max(0, val["center_x"] - label["center_x"])
                kandidat_values.append((dy + dx * 0.01, val, info))

        if kandidat_values:
            kandidat_values.sort(key=lambda x: x[0])
            _, val, info = kandidat_values[0]
            hasil_akhir[kunci] = info
            hasil_source[kunci] = "recovery"
            used_values_idx.add(val["original_index"])
            print(f"     [Recovery] {kunci}: {info['per_sajian']} {info['satuan']}")

    # Gabungkan: hasil kolom serving lebih dipercaya daripada proximity/row
    for k, v in kolom_partial.items():
        if k not in hasil_akhir or hasil_source.get(k) != "column_serving":
            hasil_akhir[k] = v
            hasil_source[k] = "column_serving"

    return hasil_akhir


# â”€â”€â”€ 5. PRE-PROCESSING â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ImageQualityChecker:
    """Quality gate untuk foto kamera HP sebelum OCR."""

    @staticmethod
    def analyze(img_cv2: np.ndarray) -> dict:
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        glare_ratio = float(np.mean(gray > 245))
        dark_ratio = float(np.mean(gray < 35))
        edge_ratio = float(np.mean(cv2.Canny(gray, 80, 180) > 0))

        blur_score = min(1.0, lap_var / 180.0)
        exposure_score = 1.0 - min(1.0, abs(mean_brightness - 135.0) / 135.0)
        contrast_score = min(1.0, contrast / 55.0)
        glare_score = 1.0 - min(1.0, glare_ratio / 0.12)
        edge_score = min(1.0, edge_ratio / 0.08)
        quality_score = float(np.clip(
            0.32 * blur_score +
            0.22 * exposure_score +
            0.20 * contrast_score +
            0.16 * glare_score +
            0.10 * edge_score,
            0.0, 1.0
        ))

        warnings = []
        if lap_var < 65:
            warnings.append("blur_detected")
        if glare_ratio > 0.08:
            warnings.append("glare_detected")
        if mean_brightness < 55:
            warnings.append("low_light")
        elif mean_brightness > 210:
            warnings.append("overexposed")
        if edge_ratio < 0.015:
            warnings.append("move_closer_or_refocus")

        return {
            "quality_score": round(quality_score, 3),
            "usable": quality_score >= 0.45 and "blur_detected" not in warnings,
            "blur_laplacian": round(lap_var, 2),
            "brightness": round(mean_brightness, 2),
            "contrast": round(contrast, 2),
            "glare_ratio": round(glare_ratio, 4),
            "dark_ratio": round(dark_ratio, 4),
            "edge_ratio": round(edge_ratio, 4),
            "warnings": warnings,
            "image_size": {"width": int(w), "height": int(h)},
        }


class Preprocessor:
    """Pipeline pre-processing citra untuk kemasan makanan Indonesia."""

    def __init__(self):
        self.last_quality = {}
        self.last_roi = {}

    @staticmethod
    def deskew(img_cv2: np.ndarray) -> np.ndarray:
        """Koreksi kemiringan foto menggunakan Hough Transform."""
        abu = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        tepi = cv2.Canny(abu, 50, 150, apertureSize=3)
        garis = cv2.HoughLines(tepi, 1, np.pi/180, threshold=100)

        if garis is None:
            return img_cv2

        sudut_list = []
        for rho, theta in garis[:, 0]:
            sudut = np.degrees(theta) - 90
            if -45 < sudut < 45:
                sudut_list.append(sudut)

        if not sudut_list:
            return img_cv2

        sudut_median = np.median(sudut_list)
        if abs(sudut_median) < 0.5:
            return img_cv2

        h, w = img_cv2.shape[:2]
        pusat = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(pusat, sudut_median, 1.0)
        hasil = cv2.warpAffine(img_cv2, M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        print(f"   [Deskew] Koreksi {sudut_median:.1f}Â°")
        return hasil

    @staticmethod
    def deglare(img_cv2: np.ndarray) -> np.ndarray:
        """Kurangi pantulan cahaya/kilap pada kemasan plastik.

        SAFETY GUARD: Jika area 'kilap' > 12% dari total pixel, SKIP.
        Ini menghindari penghancuran panel label cerah (kuning, putih)
        yang memiliki pixel brightness tinggi tapi BUKAN glare.
        Contoh kasus: markisa.jpeg panel kuning "Informasi Nilai Gizi".
        """
        abu = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        # Threshold dinaikkan dari 240 ke 248 agar hanya menangkap
        # specular highlight sesungguhnya, bukan background label cerah
        _, mask_kilap = cv2.threshold(abu, 248, 255, cv2.THRESH_BINARY)

        area_kilap = np.sum(mask_kilap > 0)
        total_pixel = mask_kilap.size
        rasio_kilap = area_kilap / total_pixel

        # Skip jika area kilap terlalu kecil (< 2%)
        if rasio_kilap < 0.02:
            return img_cv2

        # SAFETY: Skip jika area kilap terlalu besar (> 5%)
        # Area besar = background label cerah, BUKAN glare
        # Real specular glare umumnya < 5% dari total pixel
        if rasio_kilap > 0.05:
            persen = rasio_kilap * 100
            print(f"   [Deglare] SKIP — Area {persen:.1f}% terlalu besar (kemungkinan background label cerah)")
            return img_cv2

        kernel = np.ones((5, 5), np.uint8)
        # Dikurangi dari 2 iterasi ke 1 agar radius inpainting tidak terlalu lebar
        mask_dilate = cv2.dilate(mask_kilap, kernel, iterations=1)
        hasil = cv2.inpaint(img_cv2, mask_dilate, inpaintRadius=5,
                            flags=cv2.INPAINT_TELEA)
        persen = area_kilap / total_pixel * 100
        print(f"   [Deglare] Area kilap: {persen:.1f}% â€” ditangani")
        return hasil

    @staticmethod
    def normalisasi_cahaya(img_cv2: np.ndarray) -> np.ndarray:
        """CLAHE untuk normalisasi pencahayaan tidak merata."""
        lab = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # Foto kemasan plastik/frozen sering punya frost atau gradasi lokal.
        # Variansi Laplacian rendah berarti teks cenderung soft/blurred, jadi CLAHE
        # dibuat sedikit lebih agresif agar stroke huruf naik kontrasnya.
        lap_var = cv2.Laplacian(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        clip_limit = 3.5 if lap_var < 120 else 2.5
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_norm = clahe.apply(l)
        lab_norm = cv2.merge([l_norm, a, b])
        return cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)

    @staticmethod
    def remove_overlay_watermark(img_cv2: np.ndarray) -> np.ndarray:
        """Kurangi overlay berwarna/abu transparan yang besar dan tidak tipis.

        Mask dibuat konservatif: komponen kecil seperti huruf asli tabel tidak
        di-inpaint, sedangkan blob watermark besar pada panel terang boleh
        dibersihkan.
        """
        h, w = img_cv2.shape[:2]
        hsv = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        colored = ((sat > 90) & (val > 80)).astype(np.uint8) * 255
        gray_overlay = ((sat < 55) & (gray > 80) & (gray < 205)).astype(np.uint8) * 255

        # Ambil hanya komponen besar agar teks hitam normal tidak ikut hilang.
        large_mask = np.zeros((h, w), dtype=np.uint8)
        for mask in (colored, gray_overlay):
            num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for idx in range(1, num):
                x, y, bw, bh, area = stats[idx]
                area_ratio = area / float(h * w)
                if 0.002 <= area_ratio <= 0.30 and (bw > w * 0.08 or bh > h * 0.06):
                    large_mask[labels == idx] = 255

        area_ratio = np.mean(large_mask > 0)
        if area_ratio < 0.002 or area_ratio > 0.30:
            return img_cv2

        kernel = np.ones((5, 5), np.uint8)
        large_mask = cv2.dilate(large_mask, kernel, iterations=1)
        hasil = cv2.inpaint(img_cv2, large_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        print(f"   [Watermark] Overlay besar dikurangi ({area_ratio*100:.1f}% area)")
        return hasil

    @staticmethod
    def _score_panel(img_cv2: np.ndarray) -> float:
        hsv = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        sat = hsv[:, :, 1]
        white_ratio = np.mean((gray > 145) & (sat < 95))
        dark_ratio = np.mean(gray < 80)
        edge_ratio = np.mean(cv2.Canny(gray, 80, 180) > 0)
        return float(white_ratio * 2.0 + dark_ratio * 0.6 + edge_ratio * 1.2 - np.mean(sat) / 255.0)

    @classmethod
    def _crop_landscape_side(cls, img_cv2: np.ndarray) -> tuple[np.ndarray, bool]:
        h, w = img_cv2.shape[:2]
        if w <= h * 1.25:
            return img_cv2, False

        cut = int(w * 0.62)
        left = img_cv2[:, :cut]
        right = img_cv2[:, w - cut:]
        left_score = cls._score_panel(left)
        right_score = cls._score_panel(right)
        if abs(left_score - right_score) < 0.35:
            print(f"   [Crop-Landscape] Skip, skor sisi mirip ({left_score:.2f} vs {right_score:.2f})")
            return img_cv2, False

        if left_score >= right_score:
            print(f"   [Crop-Landscape] Ambil sisi kiri (score {left_score:.2f} >= {right_score:.2f})")
            return left, True

        print(f"   [Crop-Landscape] Ambil sisi kanan (score {right_score:.2f} > {left_score:.2f})")
        return right, True

    @staticmethod
    def _crop_white_panel(img_cv2: np.ndarray) -> tuple[np.ndarray, bool]:
        h, w = img_cv2.shape[:2]
        hsv = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        sat = hsv[:, :, 1]
        mask = ((gray > 135) & (sat < 110)).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0.0
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            area_ratio = (bw * bh) / float(w * h)
            if area_ratio < 0.08 or area_ratio > 0.88 or bw < w * 0.28 or bh < h * 0.18:
                continue

            roi_gray = gray[y:y + bh, x:x + bw]
            if roi_gray.size == 0:
                continue
            inv = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, bw // 4), 1))
            h_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, line_kernel)
            line_score = np.mean(h_lines > 0)
            edge_score = np.mean(cv2.Canny(roi_gray, 80, 180) > 0)
            score = area_ratio + line_score * 12.0 + edge_score * 2.0
            if score > best_score:
                best_score = score
                best = (x, y, bw, bh)

        if best is None:
            return img_cv2, False

        x, y, bw, bh = best
        margin_x = max(12, int(w * 0.025))
        margin_y = max(12, int(h * 0.025))
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + bw + margin_x)
        y2 = min(h, y + bh + margin_y)
        if (x2 - x1) < w * 0.25 or (y2 - y1) < h * 0.15:
            return img_cv2, False

        print(f"   [Crop-Panel] Panel tabel: ({x1},{y1}) -> ({x2},{y2})")
        return img_cv2[y1:y2, x1:x2], True

    @staticmethod
    def _crop_by_horizontal_lines(img_cv2: np.ndarray) -> tuple[np.ndarray, bool]:
        h, w = img_cv2.shape[:2]
        abu = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        _, biner = cv2.threshold(abu, 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 4), 1))
        garis_h = cv2.morphologyEx(biner, cv2.MORPH_OPEN, kernel_h)

        coords = cv2.findNonZero(garis_h)
        if coords is None:
            return img_cv2, False

        x, y, lebar, tinggi = cv2.boundingRect(coords)
        margin_x = max(20, int(w * 0.04))
        margin_y = max(25, int(h * 0.06))
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + lebar + margin_x)
        y2 = min(h, y + tinggi + margin_y)

        if (x2 - x1) < 500 or (y2 - y1) < 400:
            return img_cv2, False

        if (x2 - x1) > w * 0.3 and (y2 - y1) > h * 0.2:
            print(f"   [Crop-Line] Area garis tabel: ({x1},{y1}) -> ({x2},{y2})")
            return img_cv2[y1:y2, x1:x2], True

        return img_cv2, False

    @classmethod
    def crop_area_gizi(cls, img_cv2: np.ndarray) -> np.ndarray:
        """Crop area tabel gizi dengan tiga layer: landscape, panel putih, garis."""
        cropped, _ = cls.crop_area_gizi_with_metadata(img_cv2)
        return cropped

    @classmethod
    def crop_area_gizi_with_metadata(cls, img_cv2: np.ndarray) -> tuple[np.ndarray, dict]:
        """Crop ROI tabel gizi dan kembalikan metadata metode deteksi.

        Urutan ROI:
        1. landscape-side candidate,
        2. horizontal-line table isolation,
        3. white nutrition panel fallback.
        """
        original_h, original_w = img_cv2.shape[:2]
        kandidat, landscape_cropped = cls._crop_landscape_side(img_cv2)
        methods = []
        if landscape_cropped:
            methods.append("landscape_side")

        # NOTE: _crop_by_horizontal_lines DINONAKTIFKAN sebagai metode primer.
        # Two-Pass OCR di jalankan_ocr() menggantikan perannya dengan
        # anchor-based cropping yang jauh lebih akurat.
        # Fungsi _crop_by_horizontal_lines() tidak dihapus untuk referensi.
        # by_line, ok_line = cls._crop_by_horizontal_lines(kandidat)
        # if ok_line:
        #     methods.append("horizontal_lines")
        #     roi = by_line
        #     return roi, {
        #         "detected": True,
        #         "method": "+".join(methods),
        #         "original_size": {"width": int(original_w), "height": int(original_h)},
        #         "roi_size": {"width": int(roi.shape[1]), "height": int(roi.shape[0])},
        #     }

        # NOTE: _crop_white_panel JUGA DINONAKTIFKAN.
        # Pada kasus markisa.jpeg, white_panel crop memilih area "Komposisi"
        # (270x244) bukan area tabel gizi, menyebabkan OCR 0 deteksi.
        # Two-Pass OCR di jalankan_ocr() menggantikan semua metode crop ini.
        # panel, ok_panel = cls._crop_white_panel(kandidat)
        # if ok_panel:
        #     methods.append("white_panel")
        #     return panel, {
        #         "detected": True,
        #         "method": "+".join(methods),
        #         "original_size": {"width": int(original_w), "height": int(original_h)},
        #         "roi_size": {"width": int(panel.shape[1]), "height": int(panel.shape[0])},
        #     }

        roi = kandidat if landscape_cropped else img_cv2
        return roi, {
            "detected": bool(landscape_cropped),
            "method": "+".join(methods) if methods else "full_image_fallback",
            "original_size": {"width": int(original_w), "height": int(original_h)},
            "roi_size": {"width": int(roi.shape[1]), "height": int(roi.shape[0])},
        }

    def proses(self, path_gambar: str) -> np.ndarray:
        """Jalankan seluruh pipeline pre-processing. Return numpy BGR."""
        print(f"\nðŸ–¼ï¸  Pre-processing: {Path(path_gambar).name}")

        img = cv2.imread(path_gambar)
        if img is None:
            raise FileNotFoundError(f"Gambar tidak ditemukan: {path_gambar}")

        self.last_quality = ImageQualityChecker.analyze(img)
        print(f"   [Quality] score={self.last_quality['quality_score']:.2f}, warnings={self.last_quality['warnings']}")

        img = self.deskew(img)
        img = self.deglare(img)
        if REMOVE_WATERMARK:
            img = self.remove_overlay_watermark(img)
        img = self.normalisasi_cahaya(img)
        img, self.last_roi = self.crop_area_gizi_with_metadata(img)

        print(f"   [ROI] method={self.last_roi['method']}, detected={self.last_roi['detected']}")
        print(f"   [Final] Ukuran: {img.shape[1]}x{img.shape[0]}")
        return img


# â”€â”€â”€ 6. POST-PROCESSING (ParseOCR v2) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ParseOCR:
    """Parser cerdas hasil teks OCR."""

    @staticmethod
    def bersihkan_teks(teks: str) -> str:
        """Normalisasi unicode dan lowercase."""
        teks = unicodedata.normalize("NFKC", teks)
        teks = teks.lower().strip()
        teks = teks.replace("µ", "u").replace("Âµ", "u")
        teks = re.sub(r"\s*/\s*", "/", teks)
        teks = re.sub(r"\s+", " ", teks)
        return teks

    @staticmethod
    def levenshtein(s1: str, s2: str) -> int:
        """Hitung Levenshtein distance antara dua string."""
        if len(s1) < len(s2):
            return ParseOCR.levenshtein(s2, s1)
        if not s2:
            return len(s1)
        baris_lama = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            baris_baru = [i + 1]
            for j, c2 in enumerate(s2):
                insersi = baris_lama[j + 1] + 1
                hapus   = baris_baru[j] + 1
                ganti   = baris_lama[j] + (c1 != c2)
                baris_baru.append(min(insersi, hapus, ganti))
            baris_lama = baris_baru
        return baris_lama[-1]

    @classmethod
    def fuzzy_match_nutrisi(cls, teks_ocr: str, threshold: float = 0.75) -> str | None:
        """Cari key nutrisi yang paling cocok dengan teks OCR."""
        teks = cls.bersihkan_teks(teks_ocr)

        if teks in KAMUS_NUTRISI:
            return KAMUS_NUTRISI[teks]

        skor_terbaik = 0.0
        match_terbaik = None
        for variasi, key_standar in KAMUS_NUTRISI.items():
            maks_len = max(len(teks), len(variasi))
            if maks_len == 0:
                continue
            jarak = cls.levenshtein(teks, variasi)
            skor = 1.0 - jarak / maks_len
            if skor > skor_terbaik:
                skor_terbaik = skor
                match_terbaik = key_standar

        if skor_terbaik >= threshold:
            return match_terbaik
        return None

    @staticmethod
    def ekstrak_angka(teks: str) -> float | None:
        """Ekstrak angka pertama dari string (abaikan satuan)."""
        teks_bersih = normalisasi_teks_nilai_ocr(teks)
        compact = re.sub(r"\s+", "", teks_bersih)

        # Angka murni percent adalah kolom AKG, bukan nilai per sajian.
        if "%" in compact and re.fullmatch(r"%?\d+\.?\d*%?(akg|dv)?\*?", compact):
            return None

        m_unit = re.search(r"<?(\d+\.?\d*)(?:mg|g|kkal|kcal|mcg|ug)", compact)
        if m_unit:
            try:
                val = float(m_unit.group(1))
                if val > 2400:
                    return None
                return val
            except ValueError:
                return None

        pola = r"<?\s*(\d+\.?\d*)"
        m = re.search(pola, compact)
        if m:
            try:
                val = float(m.group(1))
                # Guardrail umum untuk footnote 2150 kkal atau OCR noise 9000g.
                if val > 2400:
                    return None
                return val
            except ValueError:
                return None
        return None

    @staticmethod
    def ekstrak_persen_akg(teks: str) -> float | None:
        """Ekstrak nilai %AKG jika ada dalam teks."""
        m = re.search(r"(\d+\.?\d*)\s*%", teks)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        return None

    @classmethod
    def parse_baris_teks(cls, row_text: str) -> tuple:
        """Parse satu row OCR yang sudah digabung dari beberapa bbox."""
        row_norm = cls.bersihkan_teks(row_text)
        if not row_norm or not re.search(r"\d", row_norm):
            return None, None

        # Footnote sering berisi angka 2150 kkal dan kata energi; jangan dianggap
        # sebagai baris kalori.
        skip_terms = [
            "persen akg", "percent daily value", "daily value",
            "kebutuhan energi", "calorie diet", "sajian per kemasan",
            "servings per package", "serving per pack", "serving size",
            "takaran saji", "komposisi", "ingredients", "ingredient",
            "tepung", "mengandung", "allergen", "alergen", "petunjuk",
            "simpan", "didistribusikan", "distributed",
        ]
        if any(term in row_norm for term in skip_terms):
            return None, None

        ignore_hits = []
        key_hits = []
        for variasi, key_standar in KAMUS_NUTRISI.items():
            variasi_norm = cls.bersihkan_teks(variasi)
            pos = row_norm.find(variasi_norm)
            if pos < 0:
                continue
            if key_standar is None:
                ignore_hits.append((pos, len(variasi_norm)))
            else:
                key_hits.append((pos, -len(variasi_norm), variasi_norm, key_standar))

        if not key_hits:
            m = re.match(r"^(.+?)\s+<?\d", row_norm)
            if not m:
                return None, None
            label_cand = m.group(1).strip()
            key_cand = cls.fuzzy_match_nutrisi(label_cand)
            if not key_cand:
                return None, None
            return cls._parse_baris_nutrisi(label_cand, row_norm[m.end(1):])

        canonical_keys = {hit[3] for hit in key_hits}
        if len(canonical_keys) > 1:
            return None, None

        key_hits.sort()
        pos, neg_len, variasi_norm, _ = key_hits[0]
        if ignore_hits and any(ignore_pos <= pos for ignore_pos, _ in ignore_hits):
            return None, None

        tail = row_norm[pos + len(variasi_norm):].strip()
        if not tail:
            return None, None
        return cls._parse_baris_nutrisi(variasi_norm, tail)

    @classmethod
    def _parse_baris_nutrisi(cls, label_raw: str, nilai_raw: str) -> tuple:
        """Helper: parse satu baris nutrisi dari label dan nilai mentah."""
        key_nutrisi = cls.fuzzy_match_nutrisi(label_raw)
        if key_nutrisi is None:
            return None, None

        angka = cls.ekstrak_angka(nilai_raw)
        persen_akg = cls.ekstrak_persen_akg(nilai_raw)

        satuan = "g"
        nilai_lower = nilai_raw.lower()
        if any(s in nilai_lower for s in ["kkal", "kcal", "kalori"]):
            satuan = "kkal"
        elif "mg" in nilai_lower:
            satuan = "mg"
        elif "Âµg" in nilai_lower or "µg" in nilai_lower or "mcg" in nilai_lower or "ug" in nilai_lower:
            satuan = "mcg"

        if angka is not None:
            gram_fields = {"lemak_total", "lemak_jenuh", "lemak_trans", "karbohidrat", "serat", "gula", "protein"}
            mg_fields = {"natrium", "kolesterol"}
            compact_raw = re.sub(r"\s+", "", normalisasi_teks_nilai_ocr(nilai_raw))
            compact_raw = compact_raw.replace(".gram", "gram")
            has_explicit_unit = satuan in {"kkal", "mg", "mcg"} or bool(
                re.search(r"\d+(?:\.\d+)?(?:g|gram|mg|kkal|kcal|mcg|ug)\b", compact_raw)
            )
            if key_nutrisi in gram_fields:
                m_g_noise = re.fullmatch(r"(\d)[69]", compact_raw)
                m_unit_noise = re.search(r"(\d)[69](?=g|gram)", compact_raw)
                m_trailing9 = re.fullmatch(r"(\d{1,2})9", compact_raw)
                if m_g_noise:
                    angka = float(m_g_noise.group(1))
                    satuan = "g"
                elif m_unit_noise and angka >= 10:
                    angka = float(m_unit_noise.group(1))
                    satuan = "g"
                elif m_trailing9 and not re.search(r"[a-z]", compact_raw):
                    # 89 -> 8g, 19 -> 1g (huruf g terbaca 9)
                    angka = float(m_trailing9.group(1))
                    satuan = "g"

                has_gram_unit = bool(re.search(r"\d+(?:\.\d+)?(?:g|gram)\b", compact_raw))
                bare_number = bool(re.fullmatch(r"\d+(?:\.\d+)?", compact_raw))

                if key_nutrisi == "gula" and bare_number and angka == 9:
                    angka = 0.0
                    satuan = "g"
                    has_gram_unit = True

                if not has_gram_unit and not m_g_noise and not m_unit_noise and not m_trailing9:
                    if key_nutrisi == "karbohidrat" and 1 <= angka <= 4:
                        angka = angka * 3.0
                    satuan = "g"
                has_explicit_unit = has_gram_unit or bool(m_g_noise or m_unit_noise or m_trailing9)
            if key_nutrisi == "natrium" and angka is not None and satuan == "mg":
                # 109.9 mg -> 110 mg (pembulatan label umum)
                if 100 <= angka <= 120 and abs(angka - round(angka)) < 0.15:
                    angka = float(round(angka))
            if key_nutrisi in gram_fields and satuan != "g":
                return None, None
            if key_nutrisi in mg_fields and satuan != "mg":
                return None, None
            _, batas_atas = VALIDASI_KLINIS.get(key_nutrisi, (0, 2400))
            if angka > batas_atas * 1.25:
                return None, None
            return key_nutrisi, {
                "per_sajian": angka,
                "satuan": satuan,
                "akg_persen": persen_akg,
                "_raw_value": nilai_raw,
                "_had_unit": has_explicit_unit,
            }
        return None, None


# â”€â”€â”€ 7. VALIDASI KLINIS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def validasi_klinis(data_nutrisi: dict) -> tuple[dict, list]:
    """Cek apakah nilai nutrisi masuk akal secara klinis.
    
    Dua lapis validasi:
    1. Range check â€” apakah angka masuk rentang wajar per sajian
    2. Relational check â€” apakah sub-komponen â‰¤ komponen induk
       (contoh: lemak jenuh HARUS â‰¤ lemak total)
    """
    flags = []
    data_bersih = {}

    for key, nilai_info in data_nutrisi.items():
        nilai_info = dict(nilai_info)
        for internal_key in list(nilai_info.keys()):
            if str(internal_key).startswith("_"):
                nilai_info.pop(internal_key, None)
        if key not in VALIDASI_KLINIS:
            data_bersih[key] = nilai_info
            continue

        min_val, maks_val = VALIDASI_KLINIS[key]
        angka = nilai_info.get("per_sajian")

        if angka is None:
            flags.append(f"missing_{key}")
            continue

        if not (min_val <= angka <= maks_val):
            flags.append(f"anomali_{key}_{angka}")
            print(f"   âš ï¸  [{key}] Nilai {angka} di luar range ({min_val}â€“{maks_val})")
            nilai_info["anomali"] = True

        data_bersih[key] = nilai_info

    # --- Cross-validation relasional ---
    # Aturan: sub-komponen HARUS â‰¤ komponen induk
    ATURAN_RELASIONAL = [
        # (anak, induk, deskripsi)
        ("lemak_jenuh", "lemak_total", "Lemak jenuh > lemak total"),
        ("lemak_trans", "lemak_total", "Lemak trans > lemak total"),
        ("gula", "karbohidrat", "Gula > karbohidrat total"),
        ("serat", "karbohidrat", "Serat > karbohidrat total"),
    ]

    for anak_key, induk_key, deskripsi in ATURAN_RELASIONAL:
        if anak_key in data_bersih and induk_key in data_bersih:
            val_anak = data_bersih[anak_key].get("per_sajian", 0)
            val_induk = data_bersih[induk_key].get("per_sajian", 0)

            if val_anak is not None and val_induk is not None and val_anak > val_induk:
                print(f"   ðŸ”„ [{anak_key}] Cross-validation gagal: "
                      f"{val_anak} > {induk_key}={val_induk} â†’ DIHAPUS")
                flags.append(f"crossval_{anak_key}_{val_anak}_gt_{induk_key}_{val_induk}")
                data_bersih[anak_key]["anomali"] = True
                data_bersih[anak_key]["crossval_gagal"] = deskripsi
                # Hapus nilai yang jelas salah â€” lebih baik kosong daripada menyesatkan klinis
                del data_bersih[anak_key]

    return data_bersih, flags


# â”€â”€â”€ 8. EVALUASI CER/WER â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _levenshtein(s1: str, s2: str) -> int:
    """Edit distance antara dua string."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + (0 if c1 == c2 else 1)))
        prev = curr
    return prev[-1]


def hitung_cer(ground_truth: str, prediksi: str) -> float:
    """Character Error Rate = edit_distance(gt, pred) / len(gt)"""
    gt = ground_truth.strip().lower()
    pred = prediksi.strip().lower()
    if not gt:
        return 0.0
    return _levenshtein(gt, pred) / max(len(gt), 1)


def hitung_wer(ground_truth: str, prediksi: str) -> float:
    """Word Error Rate = edit_distance_kata(gt, pred) / jumlah_kata_gt"""
    gt_kata   = ground_truth.strip().lower().split()
    pred_kata = prediksi.strip().lower().split()

    if not gt_kata:
        return 0.0

    n, m = len(gt_kata), len(pred_kata)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if gt_kata[i - 1] == pred_kata[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[n][m] / n


def hitung_field_accuracy(ground_truth_dict: dict, prediksi_dict: dict, toleransi_pct: float = 0.05) -> dict:
    """Evaluasi utama untuk skripsi: akurasi nilai field nutrisi.

    Exact match mendapat skor 1.0. Nilai yang masih dalam toleransi relatif
    mendapat skor 0.9. Missing/wrong mendapat 0.0.
    """
    total = len(ground_truth_dict)
    skor_total = 0.0
    benar = 0
    per_field = {}
    missing_fields = []
    wrong_fields = []

    for key, gt_val in ground_truth_dict.items():
        pred_info = prediksi_dict.get(key)
        pred_val = pred_info.get("per_sajian") if isinstance(pred_info, dict) else pred_info

        if pred_val is None:
            missing_fields.append(key)
            per_field[key] = {"gt": gt_val, "pred": None, "score": 0.0, "status": "missing"}
            continue

        exact = float(pred_val) == float(gt_val)
        if gt_val == 0:
            close = exact
        else:
            close = abs(float(pred_val) - float(gt_val)) / max(abs(float(gt_val)), 0.001) <= toleransi_pct

        if exact:
            score = 1.0
            benar += 1
            status = "exact"
        elif close:
            score = 0.9
            benar += 1
            status = "close"
        else:
            score = 0.0
            wrong_fields.append(key)
            status = "wrong"

        skor_total += score
        per_field[key] = {"gt": gt_val, "pred": pred_val, "score": score, "status": status}

    accuracy = skor_total / total if total else 0.0
    return {
        "field_accuracy": accuracy,
        "accuracy": accuracy,
        "total_fields": total,
        "benar": benar,
        "per_field": per_field,
        "detail": per_field,
        "missing_fields": missing_fields,
        "wrong_fields": wrong_fields,
    }


# â”€â”€â”€ 9. PIPELINE UTAMA â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _is_nutrition_anchor_text(text: str) -> bool:
    norm = ParseOCR.bersihkan_teks(text)
    if any(term in norm for term in NUTRITION_ANCHOR_TERMS):
        return True
    return ParseOCR.fuzzy_match_nutrisi(norm, threshold=0.70) is not None


def _is_non_nutrition_text(text: str) -> bool:
    norm = ParseOCR.bersihkan_teks(text)
    return any(term in norm for term in NON_NUTRITION_TERMS)


def semantic_filter_ocr(deteksi_list: list[dict]) -> tuple[list[dict], dict]:
    """Filter OCR agar rekonstruksi fokus ke tabel nutrisi, bukan seluruh kemasan."""
    if not deteksi_list:
        return [], {"enabled": False, "reason": "empty_ocr", "kept": 0, "removed": 0}

    anchors = [d for d in deteksi_list if _is_nutrition_anchor_text(d.get("text", ""))]
    if len(anchors) < 3:
        return deteksi_list, {
            "enabled": False,
            "reason": "insufficient_nutrition_anchors",
            "anchors": len(anchors),
            "kept": len(deteksi_list),
            "removed": 0,
        }

    median_h = float(np.median([max(1, d["bbox"][3]) for d in deteksi_list if d.get("bbox")])) if deteksi_list else 20.0
    min_y = max(0, min(a["center_y"] for a in anchors) - median_h * 3.0)
    max_y = max(a["center_y"] for a in anchors) + median_h * 8.0
    anchor_ys = [a["center_y"] for a in anchors]
    kept = []

    for det in deteksi_list:
        text = det.get("text", "")
        norm = ParseOCR.bersihkan_teks(text)
        has_number = bool(re.search(r"\d", norm))
        has_unit = bool(re.search(r"\d+[\.,]?\d*\s*(g|mg|kkal|kcal|%|gram)", norm))
        is_anchor = _is_nutrition_anchor_text(text)
        near_anchor = any(abs(det["center_y"] - ay) <= max(34, median_h * 2.2) for ay in anchor_ys)
        in_roi_band = min_y <= det["center_y"] <= max_y
        reject_marketing = _is_non_nutrition_text(text) and not is_anchor and not has_unit

        if reject_marketing:
            continue
        if is_anchor or has_unit or (has_number and near_anchor) or (has_number and in_roi_band):
            kept.append(det)

    if len(kept) < max(6, len(anchors)):
        return deteksi_list, {
            "enabled": False,
            "reason": "filtered_too_aggressive",
            "anchors": len(anchors),
            "candidate_kept": len(kept),
            "kept": len(deteksi_list),
            "removed": 0,
        }

    return kept, {
        "enabled": True,
        "reason": "nutrition_semantic_filter",
        "anchors": len(anchors),
        "kept": len(kept),
        "removed": len(deteksi_list) - len(kept),
    }


def simpan_debug_artifacts(debug_dir: str, img_np: np.ndarray, deteksi_list: list[dict], hasil_final: dict) -> dict:
    """Simpan crop, overlay bbox, dan deteksi mentah untuk debugging skripsi."""
    out_dir = Path(debug_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    crop_path = out_dir / "01_preprocessed_crop.jpg"
    overlay_path = out_dir / "02_ocr_overlay.jpg"
    deteksi_path = out_dir / "03_ocr_detections.json"

    cv2.imwrite(str(crop_path), img_np)
    overlay = img_np.copy()
    for idx, det in enumerate(deteksi_list, start=1):
        x, y, w, h = [int(v) for v in det["bbox"]]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 180, 0), 2)
        label = f"{idx}:{det['conf']:.2f}"
        cv2.putText(overlay, label, (x, max(14, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(overlay_path), overlay)

    deteksi_serializable = []
    for det in deteksi_list:
        deteksi_serializable.append({
            "bbox": [float(v) for v in det.get("bbox", [])],  # cast numpy scalars
            "text": det.get("text", ""),
            "conf": float(det.get("conf", 0)),
            "center_x": float(det.get("center_x", 0)),
            "center_y": float(det.get("center_y", 0)),
        })

    payload = {
        "detections": deteksi_serializable,
        "hasil": hasil_final,
    }
    with open(deteksi_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2)

    return {
        "preprocessed_crop": str(crop_path),
        "ocr_overlay": str(overlay_path),
        "detections_json": str(deteksi_path),
    }


def jalankan_ocr(path_gambar: str, ground_truth: str = None, debug_dir: str = None) -> dict:
    """Pipeline lengkap: pre-process → Two-Pass OCR → rekonstruksi tabel → validasi.
    Return dict hasil siap pakai oleh Clinical Rules Engine.

    Two-Pass OCR Strategy:
      Pass 1 (Quick Anchor Scan): Resize ke max 960px, cari anchor "Informasi Nilai Gizi".
      Pass 2 (Precision OCR): OCR full-res pada ROI yang ditemukan Pass 1.
    """
    print("\n" + "="*60)
    print("  NutriLabel v3 — PP-OCR Pipeline (Two-Pass)")
    print("="*60)

    # --- Pre-processing → numpy array ---
    preprocessor = Preprocessor()
    img_np = preprocessor.proses(path_gambar)

    # --- Load PP-OCR engine (singleton) ---
    engine = PPOCREngine()
    engine.muat()

    # ═══════════════════════════════════════════════════════════
    # PASS 1 — Quick Anchor Scan (resized image)
    # ═══════════════════════════════════════════════════════════
    full_h, full_w = img_np.shape[:2]
    max_side = 960
    scale = 1.0
    if max(full_h, full_w) > max_side:
        scale = max_side / max(full_h, full_w)
        small_img = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small_img = img_np

    print(f"\n🔍 Pass 1 — Anchor Scan ({small_img.shape[1]}x{small_img.shape[0]}, scale={scale:.3f})...")
    t0 = time.time()
    pass1_result = engine.deteksi(small_img)
    pass1_list = pass1_result["deteksi_list"]
    print(f"   Pass 1: {len(pass1_list)} teks dalam {time.time()-t0:.2f}s")

    # Cari anchor tabel gizi
    _ANCHOR_PATTERNS = [
        "informasi nilai gizi", "nutrition information",
        "nutrition facts", "nilai gizi",
    ]
    anchor_bbox = None
    anchor_text = None
    pass1_anchor_found = False

    for det in pass1_list:
        norm = ParseOCR.bersihkan_teks(det.get("text", ""))
        for pat in _ANCHOR_PATTERNS:
            if pat in norm:
                anchor_bbox = det
                anchor_text = det["text"]
                pass1_anchor_found = True
                break
        if pass1_anchor_found:
            break

    # Hitung ROI berdasarkan anchor
    roi_y1 = 0
    roi_img = img_np  # default: full image

    if pass1_anchor_found:
        # Scale koordinat kembali ke resolusi asli
        anchor_y_orig = anchor_bbox["center_y"] / scale
        margin_top = full_h * 0.02
        roi_y1 = max(0, int(anchor_y_orig - margin_top))
        roi_img = img_np[roi_y1:, :]  # dari anchor ke bawah
        print(f"   ✂️ Anchor ditemukan: \"{anchor_text}\" di Y≈{int(anchor_y_orig)}")
        print(f"   ✂️ ROI crop: y={roi_y1}..{full_h} ({roi_img.shape[1]}x{roi_img.shape[0]})")
    else:
        print("   ⚠️ Anchor tidak ditemukan, gunakan full image")

    # ═══════════════════════════════════════════════════════════
    # PASS 2 — Precision OCR (full resolution ROI)
    # ═══════════════════════════════════════════════════════════
    print(f"\n🤖 Pass 2 — Precision OCR ({roi_img.shape[1]}x{roi_img.shape[0]})...")
    t0 = time.time()
    hasil_deteksi = engine.deteksi(roi_img)
    deteksi_list = hasil_deteksi["deteksi_list"]
    html_tables = hasil_deteksi["html_tables"]
    durasi = time.time() - t0
    print(f"   ✅ {len(deteksi_list)} teks terdeteksi dalam {durasi:.2f} detik")

    deteksi_nutrisi, semantic_meta = semantic_filter_ocr(deteksi_list)
    print(
        f"   [SemanticFilter] enabled={semantic_meta.get('enabled')} "
        f"kept={semantic_meta.get('kept')} removed={semantic_meta.get('removed')}"
    )

    # Buat raw_ocr string dari semua teks terdeteksi
    teks_mentah = "\n".join(
        f"{d['text']} (conf:{d['conf']:.2f})" for d in deteksi_list
    )
    teks_nutrisi = "\n".join(
        f"{d['text']} (conf:{d['conf']:.2f})" for d in deteksi_nutrisi
    )
    print(f"\n📄 Hasil OCR mentah:\n{'─'*40}")
    for d in deteksi_list:
        print(f"   [{d['conf']:.2f}] {d['text']}")
    print(f"{'─'*40}")

    # --- Ekstrak takaran saji ---
    takaran_saji = ekstrak_takaran_saji(deteksi_nutrisi) or ekstrak_takaran_saji(deteksi_list)
    if takaran_saji:
        print(f"\n🔎 Takaran saji: {takaran_saji} g")
    else:
        print("\n🔎 Takaran saji: tidak terdeteksi")

    # --- Rekonstruksi tabel dari bbox ---
    print("\n🔁 Rekonstruksi tabel dari bbox...")
    data_nutrisi = rekonstruksi_tabel_dari_bbox(deteksi_nutrisi, html_tables)

    # --- Fallback domain label gizi ---
    print("\nðŸ§® Fallback energi lemak...")
    data_nutrisi, fallback_flags = terapkan_fallback_energi_lemak(data_nutrisi, deteksi_nutrisi)

    # --- Validasi klinis ---
    print("\nðŸ¥ Validasi klinis...")
    data_valid, flags = validasi_klinis(data_nutrisi)
    flags = fallback_flags + flags
    for quality_warning in preprocessor.last_quality.get("warnings", []):
        flags.append(f"quality_{quality_warning}")

    # --- Evaluasi CER/WER ---
    evaluasi = {}
    if ground_truth:
        raw_teks = " ".join(d["text"] for d in deteksi_nutrisi)
        cer = hitung_cer(ground_truth, raw_teks)
        wer = hitung_wer(ground_truth, raw_teks)
        evaluasi = {
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "lulus": cer < 0.10 and wer < 0.15,
        }
        print(f"\nðŸ“Š Evaluasi:")
        print(f"   CER : {cer*100:.1f}% ({'âœ…' if cer < 0.10 else 'âŒ'})")
        print(f"   WER : {wer*100:.1f}% ({'âœ…' if wer < 0.15 else 'âŒ'})")

    # --- Status ---
    if not data_valid:
        status = "failed"
        print("\nâŒ Tidak ada nutrisi diekstrak â€” foto ulang!")
    elif flags:
        status = "low_confidence"
        print(f"\nâš ï¸  Parsial â€” {len(data_valid)} OK, {len(flags)} flag")
    else:
        status = "success"
        print(f"\nâœ… Berhasil â€” {len(data_valid)} nutrisi diekstrak")

    hasil_final = {
        "status": status,
        "data": data_valid,
        "takaran_saji_g": takaran_saji,
        "flags": flags,
        "durasi_detik": round(durasi, 2),
        "raw_ocr": teks_mentah,
        "nutrition_ocr": teks_nutrisi,
        "quality": preprocessor.last_quality,
        "roi": preprocessor.last_roi,
        "semantic_filter": semantic_meta,
        "column_layout": deteksi_layout_kolom(deteksi_nutrisi or deteksi_list),
        "evaluasi": evaluasi,
        # --- NEW: Bounding box data untuk frontend ---
        "ocr_detections": [
            {
                "bbox": [float(v) for v in d.get("bbox", [])],
                "text": d.get("text", ""),
                "conf": round(float(d.get("conf", 0)), 4),
                "center_x": float(d.get("center_x", 0)),
                "center_y": float(d.get("center_y", 0)),
            }
            for d in deteksi_list
        ],
        "roi_offset": {"x": 0, "y": roi_y1},
        "pass1_anchor_found": pass1_anchor_found,
    }

    if debug_dir:
        hasil_final["debug_artifacts"] = simpan_debug_artifacts(
            debug_dir, roi_img, deteksi_nutrisi, hasil_final
        )
        print(f"\nDebug artifacts disimpan ke: {debug_dir}")

    print("\nðŸ“‹ Ringkasan nutrisi:")
    for nutrisi, info in data_valid.items():
        akg = f" ({info['akg_persen']}%AKG)" if info.get("akg_persen") else ""
        print(f"   {nutrisi:20s}: {info['per_sajian']} {info['satuan']}{akg}")

    return hasil_final


class YOLOIsiPiringku:
    _instance = None
    _device = "cpu"

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            from ultralytics import YOLO
            import torch
            import numpy as np
            print("🤖 Loading YOLO Isi Piringku Model...")
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"✅ YOLO berjalan di GPU: {gpu_name}")
                cls._device = "cuda"
            else:
                print("⚠️ YOLO berjalan di CPU")
                cls._device = "cpu"
            cls._instance = YOLO("isipiringku_v2/weights/best.pt")
            print("   Model loaded. Warming up...")
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            cls._instance(dummy, verbose=False, device=cls._device)
            print("   YOLO Warmup completed.")
        return cls._instance


def deteksi_isi_piringku(path_gambar: str) -> dict:
    """Jalankan deteksi YOLOv11-seg untuk Isi Piringku dan hitung proporsi area."""
    try:
        from ultralytics import YOLO
        import cv2
        import numpy as np
        import base64
    except ImportError as e:
        return {"status": "error", "message": f"Missing dependency: {str(e)}"}

    try:
        model = YOLOIsiPiringku.get_instance()
        
        t0 = time.time()
        # Menggunakan parameter seperti default Kaggle/Ultralytics untuk akurasi maksimal
        results = model(path_gambar, conf=0.25, iou=0.7, imgsz=640, retina_masks=True, verbose=False, device=YOLOIsiPiringku._device)
        inference_time = time.time() - t0
        
        if not results:
            return {
                "status": "success",
                "detections": [],
                "proportions": {
                    "makanan_pokok_percent": 0.0,
                    "lauk_pauk_percent": 0.0,
                    "sayur_percent": 0.0,
                    "buah_percent": 0.0
                },
                "inference_time": round(inference_time, 3),
                "annotated_image": None
            }
            
        result = results[0]
        
        # 0: Makanan Pokok, 1: Lauk Pauk, 2: Sayuran, 3: Buah-buahan, 4: Lainnya
        class_mapping = {
            0: "makanan_pokok",
            1: "lauk_pauk",
            2: "sayur",
            3: "buah",
            4: "ignored"
        }
        
        detections = []
        pixel_counts = {
            "makanan_pokok": 0,
            "lauk_pauk": 0,
            "sayur": 0,
            "buah": 0,
            "ignored": 0
        }
        
        has_masks = result.masks is not None and len(result.masks.data) > 0
        
        if has_masks:
            masks = result.masks.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy().astype(float)
            boxes = result.boxes.xyxy.cpu().numpy().astype(float)
            
            mask_h, mask_w = masks.shape[1], masks.shape[2]
            seg_map = np.full((mask_h, mask_w), -1, dtype=int)
            indices = np.argsort(confidences)
            
            for idx in indices:
                cls_idx = int(classes[idx])
                mask = masks[idx] > 0.5
                class_name = result.names[cls_idx].lower()
                category = class_mapping.get(cls_idx, "ignored")
                
                non_food_keywords = ["hand", "finger", "fork", "spoon", "knife", "plate", "bowl", "cup", "glass", "chopstick"]
                if any(kw in class_name for kw in non_food_keywords):
                    category = "ignored"
                    cls_idx = 4
                    print(f"⚠️ Non-food object ignored: {class_name}")

                seg_map[mask] = cls_idx
                
                bbox = boxes[idx].tolist()
                detections.append({
                    "class_id": cls_idx,
                    "class_name": result.names[int(classes[idx])],
                    "category": category,
                    "confidence": float(confidences[idx]),
                    "bbox": bbox
                })
            
            for cls_idx, cat in class_mapping.items():
                pixel_counts[cat] = int(np.sum(seg_map == cls_idx))
                
        else:
            if result.boxes is not None and len(result.boxes) > 0:
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy().astype(float)
                boxes = result.boxes.xyxy.cpu().numpy().astype(float)
                
                for idx in range(len(boxes)):
                    cls_idx = int(classes[idx])
                    class_name = result.names[cls_idx].lower()
                    category = class_mapping.get(cls_idx, "ignored")
                    
                    non_food_keywords = ["hand", "finger", "fork", "spoon", "knife", "plate", "bowl", "cup", "glass", "chopstick"]
                    if any(kw in class_name for kw in non_food_keywords):
                        category = "ignored"
                        cls_idx = 4
                        print(f"⚠️ Non-food object ignored: {class_name}")

                    bbox = boxes[idx].tolist()
                    xmin, ymin, xmax, ymax = bbox
                    area = (xmax - xmin) * (ymax - ymin)
                    pixel_counts[category] += int(area)
                    
                    detections.append({
                        "class_id": cls_idx,
                        "class_name": result.names[int(classes[idx])],
                        "category": category,
                        "confidence": float(confidences[idx]),
                        "bbox": bbox
                    })
        
        total_food_pixels = pixel_counts["makanan_pokok"] + pixel_counts["lauk_pauk"] + pixel_counts["sayur"] + pixel_counts["buah"]
        
        if total_food_pixels > 0:
            makanan_pokok_percent = (pixel_counts["makanan_pokok"] / total_food_pixels) * 100
            lauk_pauk_percent = (pixel_counts["lauk_pauk"] / total_food_pixels) * 100
            sayur_percent = (pixel_counts["sayur"] / total_food_pixels) * 100
            buah_percent = (pixel_counts["buah"] / total_food_pixels) * 100
        else:
            makanan_pokok_percent = 0.0
            lauk_pauk_percent = 0.0
            sayur_percent = 0.0
            buah_percent = 0.0
            
        proportions = {
            "makanan_pokok_percent": round(makanan_pokok_percent, 2),
            "lauk_pauk_percent": round(lauk_pauk_percent, 2),
            "sayur_percent": round(sayur_percent, 2),
            "buah_percent": round(buah_percent, 2)
        }
        
        # Plot/Annotate image (draw boxes, masks, and labels)
        annotated_img = result.plot(boxes=True, masks=True, labels=True)
        _, buffer = cv2.imencode('.jpg', annotated_img)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "status": "success",
            "detections": detections,
            "proportions": proportions,
            "pixel_counts": pixel_counts,
            "inference_time": round(inference_time, 3),
            "annotated_image": annotated_base64
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


try:
    from fastapi import FastAPI, UploadFile, File
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:
    FastAPI = None
    UploadFile = None
    File = None
    CORSMiddleware = None


def buat_fastapi_app():
    """Buat FastAPI app hanya jika dependensinya tersedia."""
    if FastAPI is None:
        return None

    api = FastAPI(title="NutriLabel v3 API")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.get("/api/health")
    async def health():
        try:
            import paddle
            gpu_ready = bool(paddle.device.is_compiled_with_cuda())
        except Exception:
            gpu_ready = False
        
        yolo_status = "not_loaded"
        try:
            if os.path.exists("isipiringku_v2/weights/best.pt"):
                yolo_status = "ready"
        except Exception:
            pass

        return {
            "status": "ok",
            "model": OCR_VERSION or "default",
            "gpu": gpu_ready,
            "yolo_model": yolo_status
        }

    @api.post("/api/ocr")
    async def ocr_endpoint(file: UploadFile = File(...)):
        suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name
            return jalankan_ocr(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    @api.post("/api/isi-piringku")
    async def isi_piringku_endpoint(file: UploadFile = File(...)):
        suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name
            return deteksi_isi_piringku(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return api



app = buat_fastapi_app()


# â”€â”€â”€ 10. CLI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _json_safe(obj):
    """Konversi numpy scalars agar bisa di-json.dump."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    parser = argparse.ArgumentParser(
        description="NutriLabel v3 â€” OCR Tabel Gizi via PP-OCRv4"
    )
    parser.add_argument("--image", "-i", required=False,
                        help="Path ke foto kemasan makanan (JPG/PNG)")
    parser.add_argument("--ground-truth", "-g", default=None,
                        help="Teks ground truth untuk evaluasi CER/WER")
    parser.add_argument("--output", "-o", default=None,
                        help="Simpan hasil JSON ke file")
    parser.add_argument("--debug-dir", default=None,
                        help="Folder untuk menyimpan crop, overlay bbox, dan deteksi OCR")
    parser.add_argument("--serve", action="store_true",
                        help="Jalankan FastAPI server")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host FastAPI")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port FastAPI")
    args = parser.parse_args()

    if args.serve:
        if app is None:
            print("FastAPI belum terinstal. Jalankan: pip install fastapi uvicorn python-multipart")
            sys.exit(1)
        try:
            import uvicorn
        except ImportError:
            print("uvicorn belum terinstal. Jalankan: pip install uvicorn")
            sys.exit(1)
        uvicorn.run("nutrilabel_v3_ppocr:app", host=args.host, port=args.port, reload=False)
        return None

    if not args.image:
        parser.error("--image wajib diisi kecuali memakai --serve")

    hasil = jalankan_ocr(args.image, args.ground_truth, args.debug_dir)

    output_path = args.output or args.image.rsplit(".", 1)[0] + "_hasil.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(hasil), f, ensure_ascii=False, indent=2)
    print(f"\nðŸ’¾ Hasil disimpan ke: {output_path}")

    return hasil


if __name__ == "__main__":
    main()
