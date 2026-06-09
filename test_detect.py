import sys
import json
sys.path.append('.')
from nutrilabel_v3_ppocr import deteksi_isi_piringku

if __name__ == '__main__':
    # Test on jeruk nipis image
    img_path = 'd:/Labolatorium Anti Gravity/NutrilLabel v3 - Copy/Dataset/test/jeruk nipis.png'
    print("Testing deteksi_isi_piringku on:", img_path)
    res = deteksi_isi_piringku(img_path)
    
    # Print keys and some details
    print("Keys in response:", list(res.keys()))
    print("Status:", res.get("status"))
    if "message" in res:
        print("Message:", res.get("message"))
    print("Detections:")
    for d in res.get("detections", []):
        print(f"  - {d['class_name']} ({d['category']}): {d['confidence']:.2f}")
    print("Proportions:", res.get("proportions"))
    
    # Save the base64 image to a file to verify
    img_b64 = res.get("annotated_image")
    if img_b64:
        import base64
        with open('scratch_annotated.jpg', 'wb') as f:
            f.write(base64.b64decode(img_b64))
        print("Saved annotated image to scratch_annotated.jpg")
