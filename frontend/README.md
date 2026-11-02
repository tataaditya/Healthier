# Frontend healthier

Aplikasi web statis (HTML/CSS/JS, tanpa build step).

```text
frontend/
├── index.html
├── assets/
│   ├── demo.png
│   ├── logo.png
│   └── models/best_float32.tflite
├── css/
│   ├── style.css          # layout & komponen utama
│   └── segmentation.css   # editor segmentasi
└── js/
    ├── config.js          # API base, timeout, path model
    ├── nutri-score.js     # Extended Nutri-Score
    ├── clinical-rules.js  # PERKENI 2024 / JNC 8
    ├── isi-piringku.js    # validasi proporsi piring
    ├── seefood-segmenter.js # YOLO backend + fallback TFLite
    └── app.js             # state, UI, kamera, analisis
```

Jalankan dari root proyek:

```powershell
python -m http.server 5500 --directory frontend
```

Buka `http://127.0.0.1:5500`.

## Konfigurasi API

Default: `http://127.0.0.1:8000`. Override dengan salah satu cara:

1. `?api=http://192.168.1.10:8000`
2. `localStorage.setItem('healthier:apiBase', 'http://192.168.1.10:8000')`
3. `window.HEALTHIER_API_BASE` sebelum `js/config.js` dimuat

## Urutan script

`config.js` harus dimuat sebelum modul lain karena menyediakan
`window.HEALTHIER_CONFIG`. Modul domain (`nutri-score`, `clinical-rules`,
`isi-piringku`, `seefood-segmenter`) harus dimuat sebelum `app.js`.
