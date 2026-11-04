# Local execution path (optional)

Folder ini berisi pipeline alternatif untuk menjalankan eksperimen **di PC sendiri**,
bukan di Kaggle. Untuk revisi jurnal, jalur utama adalah
[`revision_kaggle.ipynb`](../../notebooks/revision_kaggle.ipynb).

| Script | Fungsi |
| :--- | :--- |
| `check_environment.py` | Cek Python, GPU/CUDA, PyTorch, PaddleOCR, dan path dataset |
| `make_class_map.py` | Ekstrak mapping kelas dari notebook asli |
| `prepare_yolo_seg.py` | Mask semantik → dataset YOLO-seg + split stratified |
| `train_seg.py` | Training YOLO dengan config tercatat |
| `eval_seg.py` | Evaluasi instance-level + confusion matrix |
| `eval_seg_pixel.py` | Evaluasi pixel-level TP/FP/FN/TN per kelas |

Kebutuhan lokal: GPU CUDA + PyTorch GPU (training di CPU akan sangat lambat).
Jika tidak tersedia, gunakan notebook Kaggle.

Contoh urutan:

```powershell
python experiments/scripts/local/check_environment.py
python experiments/scripts/local/make_class_map.py --uec-category raw/.../category.txt --out-dir experiments/configs
python experiments/scripts/local/prepare_yolo_seg.py --help
python experiments/scripts/local/train_seg.py --help
python experiments/scripts/local/eval_seg.py --help
python experiments/scripts/local/eval_seg_pixel.py --help
```

Hasil lokal dan hasil Kaggle memakai skema file yang sama, sehingga
`experiments/scripts/make_tables.py` bisa membaca keduanya.
