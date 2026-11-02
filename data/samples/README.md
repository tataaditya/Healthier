# Sample data

Contoh kecil untuk uji manual pipeline (bukan bagian dataset evaluasi):

| File | Keterangan |
| :--- | :--- |
| `label-sample.jpg` | Foto panel Informasi Nilai Gizi. |
| `label-sample.result.json` | Hasil OCR contoh untuk gambar di atas. |
| `markisa-sample.jpeg` | Foto buah untuk uji cepat inferensi. |
| `markisa-sample.result.json` | Hasil OCR contoh untuk gambar markisa. |

Jalankan dari root proyek:

```powershell
python -m backend.app -i data/samples/label-sample.jpg -o data/samples/label-sample.result.json
```
