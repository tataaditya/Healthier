from ultralytics import YOLO
import json

try:
    model = YOLO("isipiringku_v2/weights/best.pt")
    results = model.val(data="isipiringku_v2/data.yaml", imgsz=640, verbose=True)

    metrics = {}
    for i, name in results.names.items():
        metrics[name] = {
            "precision":     round(float(results.box.p[i]), 4),
            "recall":        round(float(results.box.r[i]), 4),
            "mAP50_box":     round(float(results.box.ap50[i]), 4),
            "mAP50_95_box":  round(float(results.box.ap[i]), 4),
            "mAP50_mask":    round(float(results.seg.ap50[i]), 4),
            "mAP50_95_mask": round(float(results.seg.ap[i]), 4),
        }
    print("=== TASK 2 OUTPUT ===")
    print(json.dumps(metrics, indent=2))
except Exception as e:
    print(f"Error in Task 2: {e}")
