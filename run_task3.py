from ultralytics import YOLO
from pathlib import Path
import time, cv2

try:
    model = YOLO("isipiringku_v2/weights/best.pt")
    images = list(Path("Dataset").rglob("*.jpg"))[:5]
    times = []
    
    print("=== TASK 3 OUTPUT ===")
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None: continue
        start = time.perf_counter()
        result = model(img, verbose=False)
        ms = (time.perf_counter() - start) * 1000
        times.append(ms)
        print(f"{img_path.name}: {ms:.1f}ms | deteksi: {len(result[0].boxes)}")
    
    if times:
        print(f"\nRata-rata : {sum(times)/len(times):.1f}ms")
        print(f"Min/Max   : {min(times):.1f}ms / {max(times):.1f}ms")
    else:
        print("Tidak ada gambar yang diproses.")
except Exception as e:
    print(f"Error in Task 3: {e}")
