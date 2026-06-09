/**
 * ============================================================
 * SEEFOOD IMAGE SEGMENTER — Computer Vision Pipeline
 * ============================================================
 * Model: Google Mobile Food Segmenter V1 (DeepLab-V3 + MobileNet-V2)
 * Runtime: TensorFlow Lite via @tensorflow/tfjs-tflite (WASM)
 * Ref: Ridhani et al. (2021) "Isi Piringku Dietary Meal Proportion
 * Estimator Applications Using SeeFood Image Segmentations"
 * Sanitas Vol.12 No.2, 115-130.
 *
 * @module SeeFoodSegmenter
 * @version 1.0.0
 */
window.SeeFoodSegmenter = (function () {
    'use strict';

    // ── Configuration ──────────────────────────────────────
    const MODEL_PATH = './best_float16.tflite';
    const MODEL_INPUT_SIZE = 640;  // YOLOv8/v11 standard max
    const DEBUG = true;            // Set false for production

    // ── Internal State ─────────────────────────────────────
    let model = null;
    let status = 'idle'; // idle | loading | ready | error
    let lastError = null;

    // ── SeeFood Label Index Map (27 classes) ───────────────
    // Based on Google Mobile Food Segmenter V1 (Kaggle/TF Hub)
    // and Ridhani et al. (2021) Table 1
    const SEEFOOD_LABELS = [
        'background',                           // 0
        'food_container',                       // 1  (piring/mangkok)
        'dining_tools',                         // 2  (sendok/garpu)
        'beverages',                            // 3  (minuman)
        'starch/grains: noodles/pasta',         // 4
        'starch/grains: rice/grains/cereals',   // 5
        'starch/grains: baked_goods',           // 6
        'starch/grains: starchy_vegetables',    // 7
        'starch/grains: other',                 // 8
        'protein: eggs',                        // 9
        'protein: beans/nuts',                  // 10
        'protein: meat',                        // 11
        'protein: poultry',                     // 12
        'protein: seafood',                     // 13
        'dairy',                                // 14
        'vegetables: stem_vegetables',          // 15
        'vegetables: leafy_greens',             // 16
        'vegetables: non-starchy_roots',        // 17
        'vegetables: other',                    // 18
        'fruits',                               // 19
        'snack',                                // 20
        'sweet/desserts',                       // 21
        'herbs/spices',                         // 22
        'fats/oils/sauces',                     // 23
        'soups/stews',                          // 24
        'other_food',                           // 25
        'salad'                                 // 26
    ];

    // ── Map SeeFood index → Isi Piringku category ─────────
    // Uses LABEL_MAP already defined in isi-piringku.js
    let INDEX_TO_PIRINGKU = null;

    function buildIndexMap() {
        if (INDEX_TO_PIRINGKU) return INDEX_TO_PIRINGKU;
        const LM = window.IsiPiringku.LABEL_MAP;
        INDEX_TO_PIRINGKU = SEEFOOD_LABELS.map(label => {
            if (LM.MAKANAN_POKOK.includes(label)) return 'makanan_pokok';
            if (LM.LAUK_PAUK.includes(label))     return 'lauk_pauk';
            if (LM.SAYUR.includes(label))          return 'sayur';
            if (LM.BUAH.includes(label))           return 'buah';
            // Special: 'salad' → sayur (leafy-based)
            if (label === 'salad')                 return 'sayur';
            return null; // ignored
        });
        if (DEBUG) {
            console.log('[SeeFoodSegmenter] Index → Piringku map built:');
            INDEX_TO_PIRINGKU.forEach((cat, i) => {
                console.log(`  [${i}] ${SEEFOOD_LABELS[i]} → ${cat || 'IGNORED'}`);
            });
        }
        return INDEX_TO_PIRINGKU;
    }

    // ── Overlay colors per Isi Piringku category ──────────
    const PIRINGKU_COLORS = {
        makanan_pokok: [255, 193, 7, 120],   // Kuning amber
        lauk_pauk:     [244, 67, 54, 120],   // Merah
        sayur:         [76, 175, 80, 120],   // Hijau
        buah:          [156, 39, 176, 120]   // Ungu
    };

    // ════════════════════════════════════════════════════════
    //  MODEL LOADING
    // ════════════════════════════════════════════════════════

    /**
     * Load the SeeFood TFLite model into browser memory.
     * Caches the model instance — subsequent calls return immediately.
     * @param {Function} [statusCb] - Callback(status, message)
     * @returns {Promise<Object>} Loaded TFLite model
     */
    async function loadModel(statusCb) {
        if (model) {
            status = 'ready';
            if (statusCb) statusCb('ready', 'Model siap digunakan.');
            return model;
        }

        status = 'loading';
        if (statusCb) statusCb('loading', 'Memuat model AI segmentasi makanan (~10 MB)...');

        try {
            // Verify library availability
            if (typeof tflite === 'undefined') {
                throw new Error('Library @tensorflow/tfjs-tflite belum dimuat. Periksa koneksi internet.');
            }

            if (DEBUG) console.log('[SeeFoodSegmenter] Loading model from:', MODEL_PATH);
            const t0 = performance.now();

            model = await tflite.loadTFLiteModel(MODEL_PATH);
            window.seefoodModel = model; // global ref for debugging

            const loadTime = ((performance.now() - t0) / 1000).toFixed(2);
            if (DEBUG) console.log(`[SeeFoodSegmenter] ✅ Model loaded in ${loadTime}s`);

            // Build index map (depends on IsiPiringku being loaded)
            buildIndexMap();

            status = 'ready';
            if (statusCb) statusCb('ready', `Model siap! (dimuat dalam ${loadTime}s)`);
            return model;
        } catch (err) {
            status = 'error';
            lastError = err;
            console.error('[SeeFoodSegmenter] ❌ Failed to load model:', err);
            if (statusCb) statusCb('error', `Gagal memuat model: ${err.message}`);
            throw err;
        }
    }

    // ════════════════════════════════════════════════════════
    //  IMAGE PREPROCESSING
    // ════════════════════════════════════════════════════════

    /**
     * Preprocess an image source for model inference.
     * Center-crops to square, resizes to MODEL_INPUT_SIZE, returns tensor.
     * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} imgSrc
     * @returns {{ tensor: tf.Tensor4D, size: number, previewCanvas: HTMLCanvasElement }}
     */
    function preprocessImage(imgSrc) {
        const size = MODEL_INPUT_SIZE;
        const canvas = document.createElement('canvas');
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext('2d');

        // Source dimensions
        const w = imgSrc.naturalWidth || imgSrc.videoWidth || imgSrc.width;
        const h = imgSrc.naturalHeight || imgSrc.videoHeight || imgSrc.height;

        // Letterbox (pad to square) instead of center-crop to preserve full image
        const scale = Math.min(size / w, size / h);
        const newW = w * scale;
        const newH = h * scale;
        const dx = (size - newW) / 2;
        const dy = (size - newH) / 2;

        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, size, size);
        ctx.drawImage(imgSrc, 0, 0, w, h, dx, dy, newW, newH);

        // Convert to tensor [H, W, 3] uint8
        const imgTensor = tf.browser.fromPixels(canvas);

        // YOLOv8 models require float32 tensors normalized to [0, 1]
        // expandDims to add batch: [1, H, W, 3]
        const batched = tf.expandDims(imgTensor, 0).cast('float32').div(255.0);

        if (DEBUG) console.log('[SeeFoodSegmenter] Input tensor shape:', batched.shape, 'dtype:', batched.dtype);

        imgTensor.dispose();
        return { tensor: batched, size, previewCanvas: canvas };
    }

    // ════════════════════════════════════════════════════════
    //  INFERENCE & SEGMENTATION
    // ════════════════════════════════════════════════════════

    async function processYoloSegmentation(outputs, inputSize) {
        let out1, out2;
        if (Array.isArray(outputs)) {
            out1 = outputs.find(t => t.shape.length === 3 && t.shape[2] === 8400); 
            out2 = outputs.find(t => t.shape.length === 4 && t.shape[3] === 32); 
        } else if (typeof outputs === 'object') {
            const values = Object.values(outputs);
            out1 = values.find(t => t.shape.length === 3 && t.shape[2] === 8400);
            out2 = values.find(t => t.shape.length === 4 && t.shape[3] === 32);
        }
        
        if (!out1 || !out2) {
            throw new Error('YOLO output tensors not found');
        }
        
        const boxesTensor = tf.squeeze(out1, [0]); // [41, 8400]
        const transposed = tf.transpose(boxesTensor); // [8400, 41]
        const data = await transposed.data();
        
        const numAnchors = out1.shape[2];
        const numElements = out1.shape[1];
        const numMasks = 32;
        const numClasses = numElements - 4 - numMasks;
        const CONF_THRESH = 0.35;
        
        const boxes = [];
        const scores = [];
        const classIds = [];
        const maskCoeffs = [];
        
        for (let i = 0; i < numAnchors; i++) {
            const offset = i * numElements;
            const x = data[offset];
            const y = data[offset + 1];
            const w = data[offset + 2];
            const h = data[offset + 3];
            
            // YOLOv8 removed obj_conf. Class scores start immediately after bbox (index 4)
            let maxClassScore = -Infinity;
            let classId = -1;
            for (let c = 0; c < numClasses; c++) {
                const classScore = data[offset + 4 + c];
                if (classScore > maxClassScore) {
                    maxClassScore = classScore;
                    classId = c;
                }
            }
            
            if (maxClassScore > CONF_THRESH) {
                boxes.push([
                    y - h/2, // ymin
                    x - w/2, // xmin
                    y + h/2, // ymax
                    x + w/2  // xmax
                ]);
                scores.push(maxClassScore);
                classIds.push(classId);
                
                const coeffs = [];
                for (let m = 0; m < numMasks; m++) {
                    coeffs.push(data[offset + 4 + numClasses + m]);
                }
                maskCoeffs.push(coeffs);
            }
        }
        
        const finalMask = new Int32Array(inputSize * inputSize).fill(-1);
        
        if (boxes.length > 0) {
            const nmsIndices = await tf.image.nonMaxSuppressionAsync(
                boxes, scores, 100, 0.45, CONF_THRESH
            );
            const selectedIndices = await nmsIndices.data();
            nmsIndices.dispose();
            
            const protoMasks = tf.squeeze(out2, [0]); // [160, 160, 32]
            const protoFlat = tf.reshape(protoMasks, [160*160, 32]); 
            
            const YOLO_TO_SEEFOOD = { 0: 5, 1: 11, 2: 16, 3: 19 };
            
            for (let i = 0; i < selectedIndices.length; i++) {
                const idx = selectedIndices[i];
                const coeffs = maskCoeffs[idx];
                const classId = classIds[idx];
                const box = boxes[idx];
                
                const coeffTensor = tf.tensor2d([coeffs], [1, 32]);
                const maskRes = tf.matMul(coeffTensor, protoFlat, false, true);
                const maskSigmoid = tf.sigmoid(maskRes);
                const mask2d = tf.reshape(maskSigmoid, [160, 160]);
                
                const maskResized = tf.image.resizeBilinear(tf.expandDims(mask2d, -1), [inputSize, inputSize]);
                const maskResizedFlat = await maskResized.data();
                
                const ymin = Math.max(0, Math.floor(box[0]));
                const xmin = Math.max(0, Math.floor(box[1]));
                const ymax = Math.min(inputSize - 1, Math.floor(box[2]));
                const xmax = Math.min(inputSize - 1, Math.floor(box[3]));
                
                for (let yy = ymin; yy <= ymax; yy++) {
                    for (let xx = xmin; xx <= xmax; xx++) {
                        const mIdx = yy * inputSize + xx;
                        if (maskResizedFlat[mIdx] > 0.5) {
                            const mappedId = YOLO_TO_SEEFOOD[classId];
                            if (mappedId !== undefined) {
                                finalMask[mIdx] = mappedId;
                            }
                        }
                    }
                }
                
                coeffTensor.dispose();
                maskRes.dispose();
                maskSigmoid.dispose();
                mask2d.dispose();
                maskResized.dispose();
            }
            protoMasks.dispose();
            protoFlat.dispose();
        }
        
        boxesTensor.dispose();
        transposed.dispose();
        
        return finalMask;
    }

    /**
     * Run segmentation on an image source.
     * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} imgSrc
     * @returns {Promise<{ maskData: number[], width: number, height: number, inferenceTime: number }>}
     */
    async function segment(imgSrc) {
        if (!model) throw new Error('Model belum dimuat. Panggil loadModel() terlebih dahulu.');

        const t0 = performance.now();

        // 1. Preprocess
        const { tensor: inputTensor, size } = preprocessImage(imgSrc);

        // 2. Inference
        // Model expects uint8 input [0-255]. tf.browser.fromPixels returns int32,
        // which tfjs-tflite maps to uint8.
        let output = model.predict(inputTensor);
        
        if (DEBUG) {
            console.log('[SeeFoodSegmenter] Raw output type:', typeof output);
        }

        let maskData;
        let maskH = size;
        let maskW = size;

        let isYolo = false;
        if (Array.isArray(output) && output.length >= 2) {
            isYolo = output.some(t => t.shape && t.shape.length === 3 && t.shape[2] === 8400);
        } else if (output && typeof output === 'object') {
            isYolo = Object.values(output).some(t => t.shape && t.shape.length === 3 && t.shape[2] === 8400);
        }

        let mask2D;

        if (isYolo) {
            const finalMask = await processYoloSegmentation(output, size);
            maskData = Array.from(finalMask);
        } else {
            let outputTensor = output;
            if (!outputTensor.shape) {
                if (Array.isArray(outputTensor)) {
                    outputTensor = outputTensor[0];
                } else if (typeof outputTensor === 'object') {
                    const keys = Object.keys(outputTensor);
                    outputTensor = outputTensor[keys[0]];
                }
            }

            const shape = outputTensor.shape;
            if (shape.length === 4 && shape[3] > 1) {
                mask2D = tf.squeeze(tf.argMax(outputTensor, -1), [0]);
            } else if (shape.length === 4 && shape[3] === 1) {
                mask2D = tf.squeeze(outputTensor, [0, 3]);
            } else if (shape.length === 3) {
                mask2D = tf.squeeze(outputTensor, [0]);
            } else if (shape.length === 2) {
                mask2D = outputTensor;
            } else {
                mask2D = tf.reshape(outputTensor, [size, size]);
            }

            maskData = Array.from(await mask2D.data());
            maskH = mask2D.shape[0];
            maskW = mask2D.shape[1];
        }

        const inferenceTime = parseFloat(((performance.now() - t0) / 1000).toFixed(2));

        // 5. Debug: log detected classes
        if (DEBUG) {
            console.log(`[SeeFoodSegmenter] ⏱ Inference: ${inferenceTime}s | Mask: ${maskH}×${maskW}`);
            const unique = [...new Set(maskData)].sort((a, b) => a - b);
            console.log('[SeeFoodSegmenter] Kelas terdeteksi:', unique.length);
            unique.forEach(c => {
                const count = maskData.filter(v => v === c).length;
                const pct = (count / maskData.length * 100).toFixed(1);
                const label = SEEFOOD_LABELS[c] || `unknown_${c}`;
                const cat = (INDEX_TO_PIRINGKU || [])[c] || 'ignored';
                console.log(`  [${c}] ${label} → ${cat}: ${count}px (${pct}%)`);
            });
        }

        // 6. Cleanup tensors
        inputTensor.dispose();
        if (mask2D && typeof mask2D.dispose === 'function' && mask2D !== outputTensor) {
            mask2D.dispose();
        }
        
        if (typeof output === 'object' && output !== null && !output.shape) {
            Object.values(output).forEach(t => {
                if (t && typeof t.dispose === 'function') t.dispose();
            });
        } else if (output && typeof output.dispose === 'function') {
            output.dispose();
        }
        return { maskData, width: maskW, height: maskH, inferenceTime };
    }

    // ════════════════════════════════════════════════════════
    //  MAP TO ISI PIRINGKU PROPORTIONS
    // ════════════════════════════════════════════════════════

    /**
     * Calculate Isi Piringku proportions from segmentation mask.
     * Excludes non-food pixels (background, container, tools, beverages).
     * @param {number[]} maskData - Flat array of class indices
     * @returns {{ proportions: Object, pixelCounts: Object, totalPixels: number, foodPixels: number }}
     */
    function calcProportions(maskData) {
        if (!INDEX_TO_PIRINGKU) buildIndexMap();

        const counts = { makanan_pokok: 0, lauk_pauk: 0, sayur: 0, buah: 0, ignored: 0 };

        for (let i = 0; i < maskData.length; i++) {
            const cat = INDEX_TO_PIRINGKU[maskData[i]];
            if (cat) counts[cat]++;
            else counts.ignored++;
        }

        const totalPixels = maskData.length;
        const proportions = window.IsiPiringku.calculateProportions(counts, totalPixels);

        if (DEBUG) {
            console.log('[SeeFoodSegmenter] Pixel counts:', counts);
            console.log('[SeeFoodSegmenter] Proportions:', proportions);
        }

        return {
            proportions,
            pixelCounts: counts,
            totalPixels,
            foodPixels: totalPixels - counts.ignored
        };
    }

    // ════════════════════════════════════════════════════════
    //  RENDER OVERLAY ON CANVAS
    // ════════════════════════════════════════════════════════

    /**
     * Draw colored segmentation overlay on a canvas, on top of the original image.
     * @param {number[]} maskData - Flat array of class indices
     * @param {number} maskW - Mask width
     * @param {number} maskH - Mask height
     * @param {HTMLCanvasElement} targetCanvas - Canvas element to draw on
     * @param {HTMLImageElement} originalImg - Original source image for background
     */
    function renderOverlay(maskData, maskW, maskH, targetCanvas, originalImg) {
        const ctx = targetCanvas.getContext('2d');
        const dW = targetCanvas.width;
        const dH = targetCanvas.height;

        if (!INDEX_TO_PIRINGKU) buildIndexMap();

        // Draw original (letterboxed to square, same as preprocessing)
        const w = originalImg.naturalWidth || originalImg.width;
        const h = originalImg.naturalHeight || originalImg.height;
        const scale = Math.min(dW / w, dH / h);
        const newW = w * scale;
        const newH = h * scale;
        const dx = (dW - newW) / 2;
        const dy = (dH - newH) / 2;
        
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, dW, dH);
        ctx.drawImage(originalImg, 0, 0, w, h, dx, dy, newW, newH);

        // Build overlay in offscreen canvas at mask resolution
        const offCanvas = document.createElement('canvas');
        offCanvas.width = maskW;
        offCanvas.height = maskH;
        const offCtx = offCanvas.getContext('2d');
        const imgData = offCtx.createImageData(maskW, maskH);

        for (let i = 0; i < maskData.length; i++) {
            const cat = INDEX_TO_PIRINGKU[maskData[i]];
            const px = i * 4;
            if (cat && PIRINGKU_COLORS[cat]) {
                const c = PIRINGKU_COLORS[cat];
                imgData.data[px]     = c[0];
                imgData.data[px + 1] = c[1];
                imgData.data[px + 2] = c[2];
                imgData.data[px + 3] = c[3];
            } else {
                imgData.data[px + 3] = 0; // transparent
            }
        }

        offCtx.putImageData(imgData, 0, 0);

        // Scale overlay to display canvas
        ctx.drawImage(offCanvas, 0, 0, dW, dH);
    }

    // ════════════════════════════════════════════════════════
    //  YOLO BACKEND INTEGRATION
    // ════════════════════════════════════════════════════════

    function imageToBlob(imgElement) {
        return new Promise((resolve) => {
            if (imgElement instanceof HTMLCanvasElement) {
                imgElement.toBlob(resolve, 'image/jpeg', 0.98);
            } else {
                const canvas = document.createElement('canvas');
                canvas.width = imgElement.naturalWidth || imgElement.width || 300;
                canvas.height = imgElement.naturalHeight || imgElement.height || 300;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(imgElement, 0, 0);
                canvas.toBlob(resolve, 'image/jpeg', 0.98);
            }
        });
    }

    async function analyzeViaYOLO(imgElement) {
        const blob = await imageToBlob(imgElement);
        const formData = new FormData();
        formData.append('file', blob, 'piringku.jpg');

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 7000);

        try {
            const response = await fetch('http://127.0.0.1:8000/api/isi-piringku', {
                method: 'POST',
                body: formData,
                signal: controller.signal
            });

            if (!response.ok) {
                throw new Error(`Server returned status ${response.status}`);
            }

            const data = await response.json();
            if (data.status !== 'success') {
                throw new Error(data.message || 'Error running YOLO segmentation');
            }

            return data;
        } finally {
            clearTimeout(timeoutId);
        }
    }

    function renderYOLOOverlay(annotatedBase64, targetCanvas) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                const ctx = targetCanvas.getContext('2d');
                targetCanvas.width = img.width;
                targetCanvas.height = img.height;
                ctx.drawImage(img, 0, 0);
                resolve();
            };
            img.onerror = (e) => {
                console.error('Failed to load annotated image:', e);
                reject(e);
            };
            img.src = 'data:image/jpeg;base64,' + annotatedBase64;
        });
    }

    // ════════════════════════════════════════════════════════
    //  FULL PIPELINE (dual-mode)
    // ════════════════════════════════════════════════════════

    function showModeNotification(mode) {
        let toast = document.getElementById('modeToast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'modeToast';
            toast.style.cssText = 'display:none; position:fixed; top:20px; left:50%; transform:translateX(-50%); z-index:9999; padding:10px 20px; border-radius:30px; font-weight:700; box-shadow:0 4px 12px rgba(0,0,0,0.15); transition: opacity 0.3s; font-family: sans-serif;';
            document.body.appendChild(toast);
        }
        toast.style.display = 'block';
        toast.style.opacity = '1';
        if (mode === 'yolo') {
            toast.style.background = '#e6f4ea';
            toast.style.color = '#137333';
            toast.innerHTML = '🚀 AI Server Aktif';
        } else {
            toast.style.background = '#fef7e0';
            toast.style.color = '#b06000';
            toast.innerHTML = '⚠️ Mode Offline (Akurasi Menurun)';
        }
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => { toast.style.display = 'none'; }, 300);
        }, 4000);
    }

    /**
     * Full pipeline: segment image → calculate proportions → render overlay.
     * Tries YOLO backend first, falls back to browser SeeFood TFLite model.
     * @param {HTMLImageElement|HTMLCanvasElement} imgElement
     * @param {HTMLCanvasElement} overlayCanvas
     * @returns {Promise<{ mode: string, proportions, pixelCounts, foodPixels, inferenceTime, detections }>}
     */
    async function analyze(imgElement, overlayCanvas) {
        try {
            if (DEBUG) console.log('[SeeFoodSegmenter] Mengirim gambar ke backend YOLO...');
            const yoloResult = await analyzeViaYOLO(imgElement);
            if (DEBUG) console.log('[SeeFoodSegmenter] YOLO backend berhasil:', yoloResult);

            if (overlayCanvas && yoloResult.annotated_image) {
                await renderYOLOOverlay(yoloResult.annotated_image, overlayCanvas);
            }

            showModeNotification('yolo');

            return {
                mode: 'yolo',
                proportions: yoloResult.proportions,
                pixelCounts: yoloResult.pixel_counts,
                foodPixels: (yoloResult.pixel_counts.makanan_pokok || 0) + 
                            (yoloResult.pixel_counts.lauk_pauk || 0) + 
                            (yoloResult.pixel_counts.sayur || 0) + 
                            (yoloResult.pixel_counts.buah || 0),
                totalPixels: Object.values(yoloResult.pixel_counts).reduce((a, b) => a + b, 0),
                inferenceTime: yoloResult.inference_time,
                detections: yoloResult.detections
            };
        } catch (err) {
            console.warn('[SeeFoodSegmenter] Gagal menggunakan YOLO backend, fallback ke TFLite lokal:', err);
            
            showModeNotification('tflite');

            // Fallback to local SeeFood TFLite model
            await loadModel();
            
            const segResult = await segment(imgElement);
            const propResult = calcProportions(segResult.maskData);

            if (overlayCanvas) {
                renderOverlay(
                    segResult.maskData,
                    segResult.width, segResult.height,
                    overlayCanvas, imgElement
                );
            }

            return {
                mode: 'tflite',
                proportions: propResult.proportions,
                pixelCounts: propResult.pixelCounts,
                foodPixels: propResult.foodPixels,
                totalPixels: propResult.totalPixels,
                inferenceTime: segResult.inferenceTime,
                detections: []
            };
        }
    }

    // ════════════════════════════════════════════════════════
    //  PUBLIC API
    // ════════════════════════════════════════════════════════

    return {
        loadModel,
        segment,
        calcProportions,
        renderOverlay,
        analyze,
        getStatus:    () => ({ status, lastError }),
        getLabels:    () => SEEFOOD_LABELS,
        getIndexMap:  () => INDEX_TO_PIRINGKU || buildIndexMap(),
        PIRINGKU_COLORS,
        MODEL_INPUT_SIZE
    };
})();

/**
 * ============================================================
 * SEGMENTATION EDITOR — Interactive Correction Tool
 * ============================================================
 */
window.SegmentationEditor = (function () {
    'use strict';

    let photoCanvas, segCanvas, corrCanvas, minimapCanvas, minimapViewport, wrapper;
    let ctx, minimapCtx;
    let isDrawing = false;
    let isPanning = false;
    let currentMode = 'makanan_pokok';
    let brushSize = 30;
    let scale = 1;
    let panX = 0, panY = 0;
    let startX = 0, startY = 0;
    
    let undoStack = [];
    
    const CAT_COLORS = {
        'makanan_pokok': '#FFC107',
        'lauk_pauk': '#F44336',
        'sayur': '#4CAF50',
        'buah': '#9C27B0'
    };

    function init(pCanvas, sCanvas, cCanvas) {
        photoCanvas = pCanvas;
        segCanvas = sCanvas;
        corrCanvas = cCanvas;
        wrapper = document.getElementById('segCanvasWrapper');
        minimapCanvas = document.getElementById('minimapCanvas');
        minimapViewport = document.getElementById('minimapViewport');
        
        // Match sizes
        corrCanvas.width = photoCanvas.width || 400;
        corrCanvas.height = photoCanvas.height || 400;
        ctx = corrCanvas.getContext('2d', { willReadFrequently: true });
        
        if (minimapCanvas) {
            minimapCtx = minimapCanvas.getContext('2d');
        }

        bindEvents();
        updateCursor();
        saveState();
        updateMinimap();
    }

    function bindEvents() {
        // Pointer events for drawing & panning
        corrCanvas.addEventListener('pointerdown', handlePointerDown);
        window.addEventListener('pointermove', handlePointerMove);
        window.addEventListener('pointerup', handlePointerUp);
        
        wrapper.addEventListener('pointerdown', (e) => {
            if (scale > 1 && e.target !== corrCanvas) {
                isPanning = true;
                startX = e.clientX - panX;
                startY = e.clientY - panY;
            }
        });
    }

    function handlePointerDown(e) {
        if (scale > 1 && e.button !== 0 && !e.touches) {
            isPanning = true;
            startX = e.clientX - panX;
            startY = e.clientY - panY;
            return;
        }
        isDrawing = true;
        draw(e);
    }

    function handlePointerMove(e) {
        if (isPanning) {
            panX = e.clientX - startX;
            panY = e.clientY - startY;
            applyTransform();
            return;
        }
        if (!isDrawing) return;
        draw(e);
    }

    function handlePointerUp() {
        if (isDrawing) {
            isDrawing = false;
            saveState(); // push to undo stack
        }
        isPanning = false;
    }

    function getCanvasPos(canvas, e) {
        const rect = canvas.getBoundingClientRect();
        // When CSS transform scale is applied, getBoundingClientRect() returns
        // the visual (transformed) rect. We need to convert pointer coords
        // to the untransformed canvas coordinate space.
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        
        let clientX = e.clientX;
        let clientY = e.clientY;
        if (e.touches && e.touches.length > 0) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        } else if (e.changedTouches && e.changedTouches.length > 0) {
            clientX = e.changedTouches[0].clientX;
            clientY = e.changedTouches[0].clientY;
        }

        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        };
    }

    function draw(e) {
        const pos = getCanvasPos(corrCanvas, e);

        ctx.beginPath();
        ctx.arc(pos.x, pos.y, brushSize / 2, 0, Math.PI * 2);

        if (currentMode === 'eraser') {
            ctx.globalCompositeOperation = 'destination-out';
            ctx.fillStyle = 'rgba(0,0,0,1)';
            ctx.fill();
            ctx.globalCompositeOperation = 'source-over';
        } else {
            ctx.globalCompositeOperation = 'source-over';
            ctx.fillStyle = CAT_COLORS[currentMode];
            ctx.fill();
        }
        updateMinimap();
    }

    function updateCursor() {
        const color = currentMode === 'eraser' ? 'rgba(255,255,255,0.8)' : CAT_COLORS[currentMode];
        const svg = `<svg width="${brushSize}" height="${brushSize}" xmlns="http://www.w3.org/2000/svg">
            <circle cx="${brushSize/2}" cy="${brushSize/2}" r="${brushSize/2 - 1}" fill="${color}" stroke="#fff" stroke-width="1.5" opacity="0.7"/>
        </svg>`;
        const url = `data:image/svg+xml;base64,${btoa(svg)}`;
        const offset = brushSize / 2;
        corrCanvas.style.cursor = `url(${url}) ${offset} ${offset}, crosshair`;
    }

    function setMode(mode) {
        currentMode = mode;
        updateCursor();
    }

    function setBrushSize(size) {
        brushSize = parseInt(size);
        updateCursor();
    }

    function setZoom(factor) {
        scale = Math.max(1, Math.min(scale * factor, 4));
        if (scale === 1) {
            resetZoom();
        } else {
            wrapper.classList.add('zoomed');
            document.getElementById('minimapContainer').style.display = 'block';
            applyTransform();
        }
    }

    function resetZoom() {
        scale = 1;
        panX = 0;
        panY = 0;
        wrapper.classList.remove('zoomed');
        document.getElementById('minimapContainer').style.display = 'none';
        applyTransform();
    }

    function applyTransform() {
        const maxPanX = (scale - 1) * wrapper.clientWidth / 2;
        const maxPanY = (scale - 1) * wrapper.clientHeight / 2;
        
        if(panX > maxPanX) panX = maxPanX;
        if(panX < -maxPanX) panX = -maxPanX;
        if(panY > maxPanY) panY = maxPanY;
        if(panY < -maxPanY) panY = -maxPanY;

        const transformStr = `translate(${panX}px, ${panY}px) scale(${scale})`;
        segCanvas.style.transform = transformStr;
        corrCanvas.style.transform = transformStr;
        if(photoCanvas) photoCanvas.style.transform = transformStr;
        
        updateMinimap();
    }

    function updateMinimap() {
        if (!minimapCanvas || scale === 1 || !photoCanvas) return;
        
        minimapCtx.clearRect(0, 0, minimapCanvas.width, minimapCanvas.height);
        minimapCtx.drawImage(photoCanvas, 0, 0, minimapCanvas.width, minimapCanvas.height);
        minimapCtx.globalAlpha = 0.5;
        minimapCtx.drawImage(segCanvas, 0, 0, minimapCanvas.width, minimapCanvas.height);
        minimapCtx.globalAlpha = 1.0;
        minimapCtx.drawImage(corrCanvas, 0, 0, minimapCanvas.width, minimapCanvas.height);

        const vw = minimapCanvas.width / scale;
        const vh = minimapCanvas.height / scale;
        const panRatioX = panX / (wrapper.clientWidth * scale);
        const panRatioY = panY / (wrapper.clientHeight * scale);
        
        let vx = (minimapCanvas.width - vw) / 2 - (panRatioX * minimapCanvas.width);
        let vy = (minimapCanvas.height - vh) / 2 - (panRatioY * minimapCanvas.height);

        minimapViewport.style.width = `${vw}px`;
        minimapViewport.style.height = `${vh}px`;
        minimapViewport.style.left = `${vx}px`;
        minimapViewport.style.top = `${vy}px`;
    }

    function saveState() {
        if (undoStack.length >= 10) undoStack.shift();
        undoStack.push(corrCanvas.toDataURL());
    }

    function undo() {
        if (undoStack.length > 1) {
            undoStack.pop();
            const previousState = undoStack[undoStack.length - 1];
            const img = new Image();
            img.onload = () => {
                ctx.clearRect(0, 0, corrCanvas.width, corrCanvas.height);
                ctx.drawImage(img, 0, 0);
                updateMinimap();
            };
            img.src = previousState;
        } else if (undoStack.length === 1) {
            ctx.clearRect(0, 0, corrCanvas.width, corrCanvas.height);
            updateMinimap();
        }
    }

    function clear() {
        if(!ctx) return;
        ctx.clearRect(0, 0, corrCanvas.width, corrCanvas.height);
        undoStack = [];
        saveState();
        updateMinimap();
    }

    function getCanvasData() {
        if (!ctx) return null;
        return ctx.getImageData(0, 0, corrCanvas.width, corrCanvas.height);
    }
    
    function getColors() {
        return CAT_COLORS;
    }

    return {
        init,
        setMode,
        setBrushSize,
        setZoom,
        resetZoom,
        undo,
        clear,
        getCanvasData,
        getColors
    };
})();
