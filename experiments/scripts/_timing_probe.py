"""Quick timing probe: per-stage cost of jalankan_ocr on a few test images."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("NUTRILABEL_OCR_DEVICE", "gpu")
os.environ.setdefault("NUTRILABEL_USE_TABLE_STRUCTURE", "1")

import torch  # noqa: F401  (must precede paddle)
import cv2
from PIL import Image

from backend.app import Preprocessor, PPOCREngine, jalankan_ocr  # noqa: E402

samples = [
    "data/Dataset/Makanan Ringan/20241108_131905.jpg",
    "data/Dataset/Makanan Kaleng/Screenshot 2024-11-04 123615.png",
    "data/Dataset/Saos dan Perbumbuan/JPG_Output/70. sajiku ayam goreng.jpg",
]

eng = PPOCREngine()
t0 = time.time()
eng.muat()
print(f"ENGINE LOAD {time.time()-t0:.1f}s  table={eng.table_engine is not None}")

for rel in samples:
    p = ROOT / rel
    with Image.open(p) as im:
        w, h = im.size
    t0 = time.time()
    Preprocessor().proses(str(p))
    tpre = time.time() - t0
    t0 = time.time()
    res = jalankan_ocr(str(p))
    tot = time.time() - t0
    print(f"\n### {rel}  ({w}x{h})  preprocess={tpre:.1f}s  total={tot:.1f}s  "
          f"pass2={res.get('durasi_detik')}s  anchor={res.get('pass1_anchor_found')}  "
          f"detections={len(res.get('ocr_detections', []))}")
