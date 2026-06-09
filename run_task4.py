from pathlib import Path
import requests, time

try:
    img_path = Path("Dataset/test/termudah_ocrtest.jpg")
    times = []
    
    print("=== TASK 4 OUTPUT ===")
    for i in range(3):
        start = time.perf_counter()
        resp = requests.post(
            "http://127.0.0.1:8000/api/ocr",
            files={"file": open(img_path, "rb")}
        )
        ms = (time.perf_counter() - start) * 1000
        times.append(ms)
        print(f"Run {i+1}: {ms:.0f}ms | status: {resp.status_code}")
    
    if times:
        print(f"\nRata-rata: {sum(times)/len(times):.0f}ms")
except Exception as e:
    print(f"Error in Task 4: {e}")
