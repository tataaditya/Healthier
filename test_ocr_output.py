from paddleocr import PaddleOCR
import numpy as np

ocr = PaddleOCR(use_textline_orientation=True, lang="en", device="gpu", ocr_version="PP-OCRv4", text_det_thresh=0.3, text_det_box_thresh=0.5)

img = np.zeros((100, 100, 3), dtype=np.uint8)
res = ocr.predict(img)
print(type(res))
if len(res) > 0:
    print(type(res[0]))
    if hasattr(res[0], 'keys'):
        print(res[0].keys())
    print(res)
