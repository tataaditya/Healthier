/**
 * Konfigurasi runtime frontend healthier.
 *
 * API base dapat dioverride tanpa mengubah kode, dengan urutan prioritas:
 *   1. Query string  : ?api=http://192.168.1.10:8000
 *   2. Global        : window.HEALTHIER_API_BASE = '...' (sebelum config.js)
 *   3. localStorage  : localStorage.setItem('healthier:apiBase', '...')
 *   4. Default lokal : http://127.0.0.1:8000
 */
(function () {
    'use strict';

    const DEFAULT_API_BASE = 'http://127.0.0.1:8000';

    function readStoredBase() {
        try {
            return localStorage.getItem('healthier:apiBase');
        } catch (_) {
            return null;
        }
    }

    const params = new URLSearchParams(window.location.search);
    const apiBase = (
        params.get('api') ||
        window.HEALTHIER_API_BASE ||
        readStoredBase() ||
        DEFAULT_API_BASE
    ).replace(/\/+$/, '');

    window.HEALTHIER_CONFIG = Object.freeze({
        API_BASE_URL: apiBase,
        ENDPOINTS: Object.freeze({
            health: `${apiBase}/api/health`,
            ocr: `${apiBase}/api/ocr`,
            plate: `${apiBase}/api/isi-piringku`,
        }),
        OCR_TIMEOUT_MS: 90000,
        PLATE_TIMEOUT_MS: 7000,
        TFLITE_MODEL_URL: './assets/models/best_float32.tflite',
    });
})();
