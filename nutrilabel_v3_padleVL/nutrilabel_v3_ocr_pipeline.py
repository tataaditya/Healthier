# DITOLAK, gagal karena butuh sumber daya yang besar dan kompleksitas yang rumit


"""
NutriLabel v3 - OCR Pipeline menggunakan PaddleOCR-VL
======================================================
Jalankan via: python nutrilabel_v3_ocr_pipeline.py --image foto_kemasan.jpg

Hardware target: i5-12450H + RTX 3050 6GB + RAM 16GB
Cara load: HuggingFace transformers (BUKAN PaddlePaddle native)
Alasan: tidak perlu install CUDA toolkit Paddle yang rumit
"""

import os
import sys
import json
import time
import argparse
import re
import unicodedata
from pathlib import Path

# ─── 1. CEK DEPENDENCIES ─────────────────────────────────────────────────────

def cek_dependencies():
    """Pastikan semua library tersedia sebelum jalan."""
    deps = {
        "torch": "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124",
        "transformers": "pip install transformers>=4.45.0",
        "PIL": "pip install Pillow",
        "cv2": "pip install opencv-python",
        "numpy": "pip install numpy",
    }
    kurang = []
    for modul, instruksi in deps.items():
        try:
            __import__(modul)
        except ImportError:
            kurang.append((modul, instruksi))
    if kurang:
        print("❌ Library berikut belum terinstal:")
        for m, i in kurang:
            print(f"   {m:15s} → {i}")
        sys.exit(1)
    print("✅ Semua dependencies tersedia.")

cek_dependencies()

# ─── 2. IMPORT ────────────────────────────────────────────────────────────────

import torch
import numpy as np
import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# ─── 3. KONFIGURASI ──────────────────────────────────────────────────────────

MODEL_ID     = "PaddlePaddle/PaddleOCR-VL"   # dari HuggingFace
MAX_TOKENS   = 1024
VRAM_LIMIT   = 5.0   # GB — batas aman untuk RTX 3050 6GB (sisakan untuk OS)
CONF_MINIMUM = 0.0   # PaddleOCR-VL tidak return confidence per-token, pakai heuristik lain

# Kamus nutrisi Indonesia → key terstandarisasi
# Format: "variasi ocr salah/beda" → "key_standar"
KAMUS_NUTRISI = {
    # Energi / Kalori
    "energi":           "kalori",
    "kalori":           "kalori",
    "energy":           "kalori",
    "calories":         "kalori",
    "kkal":             "kalori",
    "kcal":             "kalori",
    # Lemak
    "lemak total":      "lemak_total",
    "lemaktotal":       "lemak_total",
    "total fat":        "lemak_total",
    "total lemak":      "lemak_total",
    "lemak":            "lemak_total",
    "fat":              "lemak_total",
    # Lemak jenuh
    "lemak jenuh":      "lemak_jenuh",
    "saturated fat":    "lemak_jenuh",
    "lemakjenuh":       "lemak_jenuh",
    "sat fat":          "lemak_jenuh",
    # Lemak trans
    "lemak trans":      "lemak_trans",
    "trans fat":        "lemak_trans",
    "lemaktrans":       "lemak_trans",
    # Kolesterol
    "kolesterol":       "kolesterol",
    "cholesterol":      "kolesterol",
    # Natrium / Garam
    "natrium":          "natrium",
    "sodium":           "natrium",
    "garam":            "natrium",
    "salt":             "natrium",
    # Karbohidrat
    "karbohidrat":      "karbohidrat",
    "karbohidrat total": "karbohidrat",
    "total karbohidrat": "karbohidrat",
    "carbohydrate":     "karbohidrat",
    "carb":             "karbohidrat",
    # Serat
    "serat":            "serat",
    "serat pangan":     "serat",
    "dietary fiber":    "serat",
    "fiber":            "serat",
    "serat makanan":    "serat",
    # Gula
    "gula":             "gula",
    "gula total":       "gula",
    "total sugars":     "gula",
    "sugar":            "gula",
    "sugars":           "gula",
    # Protein
    "protein":          "protein",
    # Vitamin & Mineral (bonus)
    "vitamin a":        "vitamin_a",
    "vitamin c":        "vitamin_c",
    "vitamin d":        "vitamin_d",
    "kalsium":          "kalsium",
    "calcium":          "kalsium",
    "zat besi":         "zat_besi",
    "iron":             "zat_besi",
}

# Range validasi klinis (per sajian) — berdasarkan PERKENI 2024 & BPOM
# Format: (min, max) dalam satuan aslinya
VALIDASI_KLINIS = {
    "kalori":      (0,   900),    # kkal per sajian
    "lemak_total": (0,   80),     # gram
    "lemak_jenuh": (0,   40),     # gram
    "lemak_trans": (0,   10),     # gram
    "kolesterol":  (0,   300),    # mg
    "natrium":     (0,   2400),   # mg
    "karbohidrat": (0,   150),    # gram
    "serat":       (0,   25),     # gram
    "gula":        (0,   100),    # gram
    "protein":     (0,   60),     # gram
}

# ─── 4. LOADER MODEL ─────────────────────────────────────────────────────────

class ModelOCR:
    """Singleton loader untuk PaddleOCR-VL via HuggingFace transformers.
    
    PENTING: PaddleOCR-VL menggunakan AutoModelForCausalLM (bukan Vision2Seq).
    Input harus diformat via apply_chat_template() dengan task prompt tetap.
    Task prompt yang tersedia: 'OCR:', 'Table Recognition:', 
    'Chart Recognition:', 'Formula Recognition:'
    """

    # Task prompt resmi PaddleOCR-VL (JANGAN diubah/ditambah teks lain)
    TASK_TABLE = "Table Recognition:"
    TASK_OCR   = "OCR:"

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def muat(self):
        """Download & load model. Hanya sekali per session."""
        if self._loaded:
            return

        print(f"\n📦 Memuat model: {MODEL_ID}")
        print("   (Download ~1.8GB pertama kali, lalu cache otomatis)\n")

        # Tentukan device & dtype
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   GPU: {torch.cuda.get_device_name(0)} | VRAM: {vram_gb:.1f} GB")
            if vram_gb >= VRAM_LIMIT:
                self.device = "cuda"
                # Gunakan float16 sesuai request untuk VRAM optimization
                dtype = torch.float16
            else:
                print(f"   ⚠️  VRAM {vram_gb:.1f}GB < {VRAM_LIMIT}GB — fallback ke CPU")
                self.device = "cpu"
                dtype = torch.float32
        else:
            print("   GPU tidak tersedia — menggunakan CPU")
            self.device = "cpu"
            dtype = torch.float32

        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        )

        # Gunakan eager attention dan device_map auto
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
            device_map="auto"
        )
        print("   ⚡ attention: eager | device_map: auto")

        self.model.eval()

        print(f"   ✅ Model siap dalam {time.time()-t0:.1f} detik")

        if self.device == "cuda":
            vram_pakai = torch.cuda.memory_allocated() / 1e9
            print(f"   VRAM terpakai: {vram_pakai:.2f} GB")

        self._loaded = True

    def inferensi(self, gambar_pil: Image.Image, task_prompt: str) -> str:
        """Jalankan inferensi dengan task prompt resmi PaddleOCR-VL.
        
        Args:
            gambar_pil: Gambar PIL RGB
            task_prompt: Salah satu dari TASK_TABLE atau TASK_OCR
        
        Returns:
            Teks hasil OCR/table recognition
        """
        # Resize image untuk mencegah OOM (max dimension 1024)
        MAX_DIM = 1024
        w, h = gambar_pil.size
        if max(w, h) > MAX_DIM:
            ratio = MAX_DIM / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            gambar_pil = gambar_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        print(f"   [Inference] Ukuran gambar final: {gambar_pil.size[0]}x{gambar_pil.size[1]}")
        print(f"   [Inference] Device aktif: {self.device}")

        # Format input sesuai chat template PaddleOCR-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": gambar_pil},
                    {"type": "text", "text": task_prompt},
                ]
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[gambar_pil],
            return_tensors="pt"
        ).to(self.device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Bersihkan cache sebelum generate
            vram_alloc = torch.cuda.memory_allocated() / (1024**2)
            vram_res = torch.cuda.memory_reserved() / (1024**2)
            print(f"   [Inference] VRAM Sebelum Generate -> Allocated: {vram_alloc:.0f} MB | Reserved: {vram_res:.0f} MB")

        # KRITIS: inference_mode() mencegah VRAM meledak (40GB+ tanpa ini)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,   # Dikurangi dari 1024 untuk hemat memory saat OOM
                do_sample=False,      # greedy — deterministik untuk OCR
                num_beams=1,          # beam search off — hemat VRAM
            )

        hasil = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return hasil.strip()


# ─── 5. PRE-PROCESSING ───────────────────────────────────────────────────────

class Preprocessor:
    """Pipeline pre-processing citra untuk kemasan makanan Indonesia."""

    @staticmethod
    def deskew(img_cv2: np.ndarray) -> np.ndarray:
        """Koreksi kemiringan foto menggunakan Hough Transform."""
        abu = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        tepi = cv2.Canny(abu, 50, 150, apertureSize=3)
        garis = cv2.HoughLines(tepi, 1, np.pi/180, threshold=100)

        if garis is None:
            return img_cv2   # tidak ada garis terdeteksi, skip

        sudut_list = []
        for rho, theta in garis[:, 0]:
            sudut = np.degrees(theta) - 90
            if -45 < sudut < 45:    # filter sudut wajar
                sudut_list.append(sudut)

        if not sudut_list:
            return img_cv2

        sudut_median = np.median(sudut_list)
        if abs(sudut_median) < 0.5:  # sudah lurus, skip rotate
            return img_cv2

        h, w = img_cv2.shape[:2]
        pusat = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(pusat, sudut_median, 1.0)
        hasil = cv2.warpAffine(img_cv2, M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        print(f"   [Deskew] Koreksi {sudut_median:.1f}°")
        return hasil

    @staticmethod
    def deglare(img_cv2: np.ndarray) -> np.ndarray:
        """Kurangi pantulan cahaya/kilap pada kemasan plastik."""
        # Deteksi area sangat terang (kilap)
        abu = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        _, mask_kilap = cv2.threshold(abu, 240, 255, cv2.THRESH_BINARY)

        area_kilap = np.sum(mask_kilap > 0)
        total_pixel = mask_kilap.size

        if area_kilap / total_pixel < 0.02:   # < 2% piksel kilap, skip
            return img_cv2

        # Inpaint area kilap
        kernel = np.ones((5, 5), np.uint8)
        mask_dilate = cv2.dilate(mask_kilap, kernel, iterations=2)
        hasil = cv2.inpaint(img_cv2, mask_dilate, inpaintRadius=5,
                            flags=cv2.INPAINT_TELEA)
        persen = area_kilap / total_pixel * 100
        print(f"   [Deglare] Area kilap: {persen:.1f}% — ditangani")
        return hasil

    @staticmethod
    def normalisasi_cahaya(img_cv2: np.ndarray) -> np.ndarray:
        """CLAHE untuk normalisasi pencahayaan tidak merata."""
        lab = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_norm = clahe.apply(l)
        lab_norm = cv2.merge([l_norm, a, b])
        return cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)

    @staticmethod
    def crop_area_gizi(img_cv2: np.ndarray) -> np.ndarray:
        """
        Coba crop area tabel 'Informasi Nilai Gizi' saja.
        Strategi: cari area teks padat di bagian tertentu gambar.
        Jika tidak berhasil, return gambar utuh.
        """
        h, w = img_cv2.shape[:2]

        # Cari garis horizontal (ciri khas tabel gizi)
        abu = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        _, biner = cv2.threshold(abu, 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 3, 1))
        garis_h = cv2.morphologyEx(biner, cv2.MORPH_OPEN, kernel_h)

        # Cari bounding box area bergaris
        coords = cv2.findNonZero(garis_h)
        if coords is None:
            return img_cv2   # fallback: kirim utuh

        x, y, lebar, tinggi = cv2.boundingRect(coords)
        # Expand sedikit untuk tidak memotong teks pinggir
        margin = 30
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w, x + lebar + margin)
        y2 = min(h, y + tinggi + margin)

        # Hanya crop jika area cukup besar (bukan noise)
        if (x2 - x1) > w * 0.3 and (y2 - y1) > h * 0.2:
            print(f"   [Crop] Area tabel gizi: ({x1},{y1}) → ({x2},{y2})")
            return img_cv2[y1:y2, x1:x2]

        return img_cv2   # fallback

    def proses(self, path_gambar: str) -> Image.Image:
        """Jalankan seluruh pipeline pre-processing."""
        print(f"\n🖼️  Pre-processing: {Path(path_gambar).name}")

        img = cv2.imread(path_gambar)
        if img is None:
            raise FileNotFoundError(f"Gambar tidak ditemukan: {path_gambar}")

        img = self.deskew(img)
        img = self.deglare(img)
        img = self.normalisasi_cahaya(img)
        img = self.crop_area_gizi(img)

        # Konversi ke PIL RGB untuk transformers
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)


# ─── 6. POST-PROCESSING (ParseOCR v2) ────────────────────────────────────────

class ParseOCR:
    """Parser cerdas hasil teks OCR dari PaddleOCR-VL."""

    @staticmethod
    def bersihkan_teks(teks: str) -> str:
        """Normalisasi unicode dan lowercase."""
        teks = unicodedata.normalize("NFKC", teks)
        return teks.lower().strip()

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
        """
        Cari key nutrisi yang paling cocok dengan teks OCR.
        Return key standar atau None jika tidak ada yang cukup mirip.
        """
        teks = cls.bersihkan_teks(teks_ocr)

        # Cek exact match dulu
        if teks in KAMUS_NUTRISI:
            return KAMUS_NUTRISI[teks]

        # Fuzzy match dengan semua variasi di kamus
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
        # Tangani format: "250 kkal", "8,5 g", "< 1", "0.5"
        teks_bersih = teks.replace(",", ".").replace(" ", "")
        pola = r"<?\s*(\d+\.?\d*)"
        m = re.search(pola, teks_bersih)
        if m:
            try:
                return float(m.group(1))
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
    def _parse_baris_nutrisi(cls, label_raw: str, nilai_raw: str) -> tuple:
        """Helper: parse satu baris nutrisi dari label dan nilai mentah.
        Return (key_nutrisi, info_dict) atau (None, None) jika tidak cocok.
        """
        key_nutrisi = cls.fuzzy_match_nutrisi(label_raw)
        if key_nutrisi is None:
            return None, None

        angka = cls.ekstrak_angka(nilai_raw)
        persen_akg = cls.ekstrak_persen_akg(nilai_raw)

        # Tentukan satuan
        satuan = "g"
        nilai_lower = nilai_raw.lower()
        if any(s in nilai_lower for s in ["kkal", "kcal", "kalori"]):
            satuan = "kkal"
        elif "mg" in nilai_lower:
            satuan = "mg"
        elif "µg" in nilai_lower or "mcg" in nilai_lower:
            satuan = "µg"

        if angka is not None:
            return key_nutrisi, {
                "per_sajian": angka,
                "satuan": satuan,
                "akg_persen": persen_akg,
            }
        return None, None

    @classmethod
    def parse_markdown_table(cls, teks_mentah: str) -> dict:
        """Parse output Markdown table dari PaddleOCR-VL 'Table Recognition:'.
        
        Contoh input model:
        | Nutrisi | Per Sajian | %AKG |
        |---|---|---|
        | Energi Total | 100 kkal | 5% |
        | Lemak Total | 4 g | 6% |
        """
        hasil = {}
        baris_list = teks_mentah.strip().split("\n")

        for baris in baris_list:
            baris = baris.strip()
            if not baris.startswith("|"):
                continue
            # Skip separator row (|---|---|)
            if re.match(r"^\|[\s\-:|]+\|$", baris):
                continue

            sel_list = [s.strip() for s in baris.split("|")]
            # Hapus elemen kosong dari split (awal dan akhir)
            sel_list = [s for s in sel_list if s]

            if len(sel_list) < 2:
                continue

            label_raw = sel_list[0]
            # Gabungkan sisa kolom sebagai nilai (per sajian + %AKG bisa di kolom berbeda)
            nilai_parts = sel_list[1:]
            nilai_gabung = " ".join(nilai_parts)

            key, info = cls._parse_baris_nutrisi(label_raw, nilai_gabung)
            if key and info:
                # Jika %AKG ada di kolom terpisah, coba ekstrak dari kolom terakhir
                if info["akg_persen"] is None and len(sel_list) >= 3:
                    info["akg_persen"] = cls.ekstrak_persen_akg(sel_list[-1])
                hasil[key] = info

        return hasil

    @classmethod
    def parse_key_value(cls, teks_mentah: str) -> dict:
        """Parse output format key:value (fallback untuk mode OCR biasa)."""
        hasil = {}
        baris_list = teks_mentah.strip().split("\n")

        for baris in baris_list:
            baris = baris.strip()
            if not baris:
                continue

            # Pisahkan label dan nilai berdasarkan ":"
            if ":" in baris:
                bagian = baris.split(":", 1)
            else:
                continue

            if len(bagian) < 2:
                continue

            label_raw, nilai_raw = bagian[0].strip(), bagian[1].strip()
            key, info = cls._parse_baris_nutrisi(label_raw, nilai_raw)
            if key and info:
                hasil[key] = info

        return hasil

    @classmethod
    def parse_output_model(cls, teks_mentah: str) -> dict:
        """Parse output PaddleOCR-VL — coba Markdown table dulu, lalu key:value.
        
        Strategi:
        1. Jika output mengandung '|', parse sebagai Markdown table
        2. Jika gagal, parse sebagai key:value
        3. Pilih yang menghasilkan lebih banyak nutrisi valid
        """
        hasil_md = {}
        hasil_kv = {}

        # Coba Markdown table parser
        if "|" in teks_mentah:
            hasil_md = cls.parse_markdown_table(teks_mentah)

        # Coba key:value parser
        if ":" in teks_mentah:
            hasil_kv = cls.parse_key_value(teks_mentah)

        # Pilih yang menghasilkan lebih banyak nutrisi
        if len(hasil_md) >= len(hasil_kv):
            if hasil_md:
                print(f"   [Parser] Markdown table → {len(hasil_md)} nutrisi")
            return hasil_md if hasil_md else hasil_kv
        else:
            print(f"   [Parser] Key:Value → {len(hasil_kv)} nutrisi")
            return hasil_kv


# ─── 7. VALIDASI KLINIS ──────────────────────────────────────────────────────

def validasi_klinis(data_nutrisi: dict) -> tuple[dict, list]:
    """
    Cek apakah nilai nutrisi masuk akal secara klinis.
    Return: (data_bersih, daftar_flag)
    """
    flags = []
    data_bersih = {}

    for key, nilai_info in data_nutrisi.items():
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
            print(f"   ⚠️  [{key}] Nilai {angka} di luar range normal ({min_val}–{maks_val})")
            # Masih simpan tapi tandai
            nilai_info["anomali"] = True

        data_bersih[key] = nilai_info

    return data_bersih, flags


# ─── 8. EVALUASI CER/WER ─────────────────────────────────────────────────────

def hitung_cer(ground_truth: str, prediksi: str) -> float:
    """
    Character Error Rate = edit_distance(gt, pred) / len(gt)
    Nilai 0.0 = sempurna, 1.0 = semua salah
    """
    gt = ground_truth.strip().lower()
    pred = prediksi.strip().lower()
    if not gt:
        return 0.0
    jarak = ParseOCR.levenshtein(gt, pred)
    return jarak / max(len(gt), 1)


def hitung_wer(ground_truth: str, prediksi: str) -> float:
    """
    Word Error Rate = edit_distance_kata(gt, pred) / jumlah_kata_gt
    """
    gt_kata   = ground_truth.strip().lower().split()
    pred_kata = prediksi.strip().lower().split()

    if not gt_kata:
        return 0.0

    # Levenshtein di level kata
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


# ─── 9. PIPELINE UTAMA ───────────────────────────────────────────────────────

def jalankan_ocr(path_gambar: str, ground_truth: str = None) -> dict:
    """
    Pipeline lengkap: pre-process → OCR → post-process → validasi.
    Return dict hasil siap pakai oleh Clinical Rules Engine.
    """
    print("\n" + "="*60)
    print("  NutriLabel v3 — OCR Pipeline")
    print("="*60)

    # --- Pre-processing ---
    preprocessor = Preprocessor()
    gambar_pil = preprocessor.proses(path_gambar)

    # --- Load model (singleton, hanya sekali) ---
    model = ModelOCR()
    model.muat()

    # --- DUAL-PASS INFERENCE ---
    # Pass 1: Table Recognition (output = Markdown table)
    # Pass 2: OCR biasa (fallback jika tabel gagal diparse)
    print("\n🤖 Pass 1: Table Recognition...")
    t0 = time.time()
    teks_tabel = model.inferensi(gambar_pil, ModelOCR.TASK_TABLE)
    durasi_tabel = time.time() - t0
    print(f"   ✅ Selesai dalam {durasi_tabel:.2f} detik")
    print(f"\n📄 Output Table Recognition:\n{'─'*40}\n{teks_tabel}\n{'─'*40}")

    # Parse hasil Table Recognition
    print("\n🔍 Post-processing (ParseOCR v2)...")
    data_nutrisi = ParseOCR.parse_output_model(teks_tabel)
    teks_mentah = teks_tabel
    durasi = durasi_tabel

    # Jika Table Recognition kurang berhasil, coba OCR biasa
    MIN_NUTRISI_VALID = 3  # minimal 3 nutrisi untuk dianggap berhasil
    if len(data_nutrisi) < MIN_NUTRISI_VALID:
        print(f"\n🔄 Pass 2: OCR (hanya {len(data_nutrisi)} nutrisi dari tabel, coba OCR)...")
        t0 = time.time()
        teks_ocr = model.inferensi(gambar_pil, ModelOCR.TASK_OCR)
        durasi_ocr = time.time() - t0
        print(f"   ✅ Selesai dalam {durasi_ocr:.2f} detik")
        print(f"\n📄 Output OCR:\n{'─'*40}\n{teks_ocr}\n{'─'*40}")

        data_ocr = ParseOCR.parse_output_model(teks_ocr)
        if len(data_ocr) > len(data_nutrisi):
            data_nutrisi = data_ocr
            teks_mentah = teks_ocr
            durasi = durasi_tabel + durasi_ocr
            print(f"   [Fallback] OCR menghasilkan {len(data_ocr)} nutrisi (lebih baik)")
        else:
            durasi = durasi_tabel + durasi_ocr

    # --- Validasi klinis ---
    print("\n🏥 Validasi klinis...")
    data_valid, flags = validasi_klinis(data_nutrisi)

    # --- Evaluasi CER/WER (jika ground truth tersedia) ---
    evaluasi = {}
    if ground_truth:
        cer = hitung_cer(ground_truth, teks_mentah)
        wer = hitung_wer(ground_truth, teks_mentah)
        evaluasi = {
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "lulus": cer < 0.10 and wer < 0.15,
        }
        print(f"\n📊 Evaluasi:")
        print(f"   CER : {cer*100:.1f}% ({'✅ lulus' if cer < 0.10 else '❌ gagal'})")
        print(f"   WER : {wer*100:.1f}% ({'✅ lulus' if wer < 0.15 else '❌ gagal'})")

    # --- Tentukan status ---
    if not data_valid:
        status = "failed"
        print("\n❌ Tidak ada nutrisi berhasil diekstrak — minta foto ulang!")
    elif flags:
        status = "low_confidence"
        print(f"\n⚠️  Parsial — {len(data_valid)} nutrisi OK, {len(flags)} flag")
    else:
        status = "success"
        print(f"\n✅ Berhasil — {len(data_valid)} nutrisi diekstrak")

    # --- Susun output final ---
    hasil_final = {
        "status":      status,
        "data":        data_valid,
        "flags":       flags,
        "durasi_detik": round(durasi, 2),
        "raw_ocr":     teks_mentah,
        "evaluasi":    evaluasi,
    }

    # Tampilkan ringkasan
    print("\n📋 Ringkasan nutrisi:")
    for nutrisi, info in data_valid.items():
        akg = f" ({info['akg_persen']}%AKG)" if info.get("akg_persen") else ""
        print(f"   {nutrisi:20s}: {info['per_sajian']} {info['satuan']}{akg}")

    return hasil_final


# ─── 10. CLI ENTRY POINT ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NutriLabel v3 — OCR Tabel Gizi via PaddleOCR-VL"
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path ke foto kemasan makanan (JPG/PNG)"
    )
    parser.add_argument(
        "--ground-truth", "-g",
        default=None,
        help="Teks ground truth untuk evaluasi CER/WER (opsional)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Simpan hasil JSON ke file (opsional)"
    )
    args = parser.parse_args()

    hasil = jalankan_ocr(args.image, args.ground_truth)

    # Simpan ke file jika diminta
    output_path = args.output or args.image.rsplit(".", 1)[0] + "_hasil.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(hasil, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Hasil disimpan ke: {output_path}")

    return hasil


if __name__ == "__main__":
    main()
