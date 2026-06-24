"""Test inference langsung dari best.pt — cek deteksi mentah"""
import torch
from ultralytics import YOLO
import sys

model = YOLO("best.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"Model task: {model.task}")
print(f"Model type: {type(model.model).__name__ if hasattr(model, 'model') else 'unknown'}")

# Test dengan gambar dari argumen, atau gunakan gambar default
img_path = sys.argv[1] if len(sys.argv) > 1 else "termudah_ocrtest.jpg"
print(f"\nTesting pada: {img_path}")

# Coba berbagai confidence thresholds
for conf in [0.1, 0.25, 0.35, 0.5]:
    results = model(img_path, conf=conf, verbose=False, device=device)
    result = results[0]
    
    n_boxes = len(result.boxes) if result.boxes is not None else 0
    has_masks = result.masks is not None and len(result.masks.data) > 0 if result.masks is not None else False
    
    print(f"\n--- conf={conf} ---")
    print(f"  Boxes detected: {n_boxes}")
    print(f"  Has masks: {has_masks}")
    
    if result.boxes is not None and n_boxes > 0:
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        
        for i in range(n_boxes):
            cls_name = result.names[int(classes[i])]
            print(f"  [{i}] {cls_name} (class {classes[i]}) — conf: {confidences[i]:.4f}")
            
        if has_masks:
            masks = result.masks.data.cpu().numpy()
            print(f"  Mask shape: {masks.shape}")
            for i in range(len(masks)):
                mask_pixels = (masks[i] > 0.5).sum()
                print(f"    Mask[{i}] ({result.names[int(classes[i])]}): {mask_pixels} pixels")
