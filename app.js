(function() {
    const $ = id => document.getElementById(id);
    const $$ = selector => document.querySelectorAll(selector);
    
    // --- State Management ---
    const STATE_KEY = 'NutriLabel_State';
    let state = {
        user: null,
        limitSugar: 50,
        limitCalorie: 2000,
        limitSodium: 2300,
        limitSaturatedFat: 22,
        history: []
    };
    let chartInstance = null;
    let chartNSInstance = null;
    let cropperInstance = null;

    // ── Timezone-safe Date Helper ──────────────────────────
    function getLocalDateStr(dateOrStr) {
        const d = dateOrStr ? new Date(dateOrStr) : new Date();
        return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
    }

    // ── Normalize History Entry Schema ─────────────────────
    function normalizeEntry(entry) {
        return {
            ...entry,
            id: entry.id || 'entry_' + Date.now(),
            name: entry.name || entry.foodName || 'Makanan',
            cal: Math.round(entry.cal || entry.calories || 0),
            fat: parseFloat((entry.fat || entry.fat_g || 0).toFixed(1)),
            protein: parseFloat((entry.protein || entry.protein_g || 0).toFixed(1)),
            sugar: parseFloat((entry.sugar || 0).toFixed(1)),
            sodium: parseFloat((entry.sodium || 0).toFixed(1)),
            fiber: parseFloat((entry.fiber || 0).toFixed(1)),
            carbohydrates_g: parseFloat((entry.carbohydrates_g || entry.carb || 0).toFixed(1)),
            date: entry.date || new Date().toISOString(),
            grade: entry.grade || null
        };
    }

    // ── Confetti Celebration ───────────────────────────────
    function launchConfetti() {
        const container = $('confettiContainer');
        if (!container) return;
        container.innerHTML = '';
        const colors = ['#48C78E','#8ED96C','#F4C45A','#EF7768','#74b9ff','#a29bfe','#fd79a8','#fdcb6e'];
        const shapes = ['circle','rect','star'];
        for (let i = 0; i < 50; i++) {
            const piece = document.createElement('div');
            piece.className = `confetti-piece ${shapes[Math.floor(Math.random()*shapes.length)]}`;
            piece.style.left = `${Math.random() * 100}%`;
            piece.style.background = colors[Math.floor(Math.random()*colors.length)];
            piece.style.animationDelay = `${Math.random() * 0.8}s`;
            piece.style.animationDuration = `${2 + Math.random() * 2}s`;
            container.appendChild(piece);
        }
        setTimeout(() => { container.innerHTML = ''; }, 4000);
    }

    // ── Haptic Feedback ────────────────────────────────────
    function haptic(style) {
        if (!navigator.vibrate) return;
        if (style === 'light') navigator.vibrate(10);
        else if (style === 'medium') navigator.vibrate(25);
        else if (style === 'success') navigator.vibrate([15, 50, 15]);
    }

    // ── Animated Value Update ──────────────────────────────
    function animateValue(el) {
        if (!el) return;
        el.classList.remove('value-updated');
        void el.offsetWidth; // force reflow
        el.classList.add('value-updated');
    }
    let clinicalEngine = null;
    const OCR_API_URL = 'http://127.0.0.1:8000/api/ocr';
    const OCR_FETCH_TIMEOUT_MS = 90000;
    const CROPPER_OPTS = {
        viewMode: 1,
        autoCropArea: 0.88,
        movable: true,
        zoomable: true,
        rotatable: false,
        scalable: false,
        guides: true,
        center: true,
        highlight: true,
        background: true,
        responsive: true,
    };

    async function fetchOcrWithTimeout(url, options) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), OCR_FETCH_TIMEOUT_MS);
        try {
            return await fetch(url, { ...options, signal: controller.signal });
        } finally {
            clearTimeout(timer);
        }
    }

    function loadState() {
        const saved = localStorage.getItem(STATE_KEY);
        if(saved) {
            const parsed = JSON.parse(saved);
            // Migrate old state shape
            state.user = parsed.user || null;
            state.limitSugar = parsed.limitSugar || 50;
            state.limitCalorie = parsed.limitCalorie || 2000;
            state.limitSodium = parsed.limitSodium || 2300;
            state.limitSaturatedFat = parsed.limitSaturatedFat || 22;
            // Normalize all history entries on load to fix schema mismatches
            state.history = (parsed.history || []).map(h => normalizeEntry(h));
        }
        updateDashboard();
    }

    function saveState() {
        localStorage.setItem(STATE_KEY, JSON.stringify(state));
        updateDashboard();
    }

    // --- Navigation & Views ---
    function switchView(viewId) {
        $$('.view').forEach(v => v.classList.remove('active'));
        $$('.nav-item').forEach(n => n.classList.remove('active'));
        
        $(viewId).classList.add('active');
        
        // Map sub-views to their parent nav button
        let navTarget = viewId;
        if (viewId === 'viewOcrScanner' || viewId === 'viewIsiPiringku' || viewId === 'viewCrop' || viewId === 'viewCorrection' || viewId === 'viewLibrary') {
            navTarget = 'viewScanner';
        }
        const navBtn = document.querySelector(`[data-view="${navTarget}"]`);
        if(navBtn) navBtn.classList.add('active');

        if(viewId === 'viewOcrScanner') startCamera();
        else stopCamera();
    }

    // --- Camera ---
    let stream = null;
    let cameraQualityTimer = null;
    async function startCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
            $('cameraFeed').srcObject = stream;
            $('scanLaser').style.display = 'block';
            startCameraQualityLoop();
        } catch(e) {
            console.warn(e);
            alert("Harap berikan izin akses kamera pada browser.");
        }
    }

    function stopCamera() {
        if(stream) {
            stream.getTracks().forEach(t => t.stop());
            stream = null;
            $('scanLaser').style.display = 'none';
            $('scanLoading').style.display = 'none';
        }
        if(cameraQualityTimer) {
            clearInterval(cameraQualityTimer);
            cameraQualityTimer = null;
        }
    }

    function updateQualityUi(prefix, level, title, hint) {
        const dot = $(`${prefix}QualityDot`);
        const titleEl = $(`${prefix}QualityTitle`);
        const hintEl = $(`${prefix}QualityHint`);
        if(dot) dot.className = `quality-dot ${level}`;
        if(titleEl) titleEl.textContent = title;
        if(hintEl) hintEl.textContent = hint;
    }

    function estimateCanvasQuality(canvas) {
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        const w = canvas.width, h = canvas.height;
        if(!w || !h) return { level: 'warn', title: 'Kamera belum siap', hint: 'Tunggu preview kamera muncul.' };
        const data = ctx.getImageData(0, 0, w, h).data;
        let sum = 0, sumSq = 0, bright = 0, dark = 0;
        for(let i = 0; i < data.length; i += 4) {
            const y = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
            sum += y; sumSq += y * y;
            if(y > 245) bright++;
            if(y < 35) dark++;
        }
        const n = data.length / 4;
        const mean = sum / n;
        const std = Math.sqrt(Math.max(0, sumSq / n - mean * mean));
        const glare = bright / n;
        const lowLight = dark / n;
        if(glare > 0.16) return { level: 'bad', title: 'Pantulan terlalu kuat', hint: 'Miringkan kemasan atau pindah dari lampu langsung.' };
        if(mean < 55 || lowLight > 0.25) return { level: 'bad', title: 'Terlalu gelap', hint: 'Tambah cahaya, lalu arahkan ulang ke tabel gizi.' };
        if(std < 28) return { level: 'warn', title: 'Kontras rendah', hint: 'Dekatkan kamera dan pastikan teks label terlihat tajam.' };
        return { level: 'good', title: 'Siap dipindai', hint: 'Tabel gizi sudah cukup jelas. Penuhi bingkai lalu pindai.' };
    }

    function startCameraQualityLoop() {
        if(cameraQualityTimer) clearInterval(cameraQualityTimer);
        cameraQualityTimer = setInterval(() => {
            const video = $('cameraFeed');
            if(!video || !video.videoWidth) return;
            const canvas = $('snapshotCanvas');
            const targetW = 160;
            const targetH = Math.max(120, Math.round(video.videoHeight / video.videoWidth * targetW));
            canvas.width = targetW;
            canvas.height = targetH;
            canvas.getContext('2d').drawImage(video, 0, 0, targetW, targetH);
            const q = estimateCanvasQuality(canvas);
            updateQualityUi('camera', q.level, q.title, q.hint);
        }, 900);
    }

    // --- Setup Upload Gambar ---
    $('uploadImg').addEventListener('change', function(e) {
        const file = e.target.files[0];
        if(!file) return;

        const reader = new FileReader();
        reader.onload = function(event) {
            const cropImg = $('imageToCrop');
            cropImg.src = event.target.result;
            
            switchView('viewCrop');

            if(cropperInstance) cropperInstance.destroy();
            cropperInstance = new Cropper(cropImg, CROPPER_OPTS);
            
            $('uploadImg').value = ''; // Reset uploader status
        };
        reader.readAsDataURL(file);
    });

    // --- Kamera ke Cropper ---
    $('btnCapture').addEventListener('click', () => {
        if(!stream) {
            alert("Kamera tidak aktif!");
            return;
        }

        const video = $('cameraFeed');
        const canvas = $('snapshotCanvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);

        const imgData = canvas.toDataURL('image/jpeg');
        const cropImg = $('imageToCrop');
        cropImg.src = imgData;

        switchView('viewCrop');

        if(cropperInstance) cropperInstance.destroy();
        cropperInstance = new Cropper(cropImg, CROPPER_OPTS);
    });

    function canvasToBlob(canvas, type = 'image/jpeg', quality = 0.98) {
        return new Promise(resolve => canvas.toBlob(resolve, type, quality));
    }

    function setNumberInput(id, value) {
        const el = $(id);
        if (!el || value === undefined || value === null || Number.isNaN(Number(value))) return;
        el.value = Number(value);
    }

    // --- Nutrisi keyword list untuk bbox coloring (subset KAMUS_NUTRISI) ---
    const NUTRISI_KEYWORDS = [
        'energi','kalori','energy','calories','lemak','fat','protein','karbohidrat',
        'carbohydrate','gula','sugar','natrium','sodium','garam','salt','kolesterol',
        'cholesterol','serat','fiber','vitamin','kalsium','calcium','zat besi','iron',
        'takaran saji','serving size','informasi nilai gizi','nutrition',
    ];
    function isNutrisiLabel(text) {
        const t = text.toLowerCase();
        return NUTRISI_KEYWORDS.some(k => t.includes(k));
    }
    function isNumericValue(text) {
        return /\d/.test(text) && /[gm%]|kkal|kcal|mg/i.test(text);
    }

    function renderOcrBboxOverlay(payload) {
        const detections = payload.ocr_detections;
        const card = $('ocrDetectionCard');
        if (!detections || !detections.length || !window._lastCropDataURL) {
            if (card) card.style.display = 'none';
            return;
        }
        card.style.display = 'block';

        // Render crop image to preview canvas
        const previewCanvas = $('ocrPreviewCanvas');
        const bboxCanvas = $('ocrBboxCanvas');
        const img = new Image();
        img.onload = function() {
            previewCanvas.width = img.width;
            previewCanvas.height = img.height;
            bboxCanvas.width = img.width;
            bboxCanvas.height = img.height;
            previewCanvas.getContext('2d').drawImage(img, 0, 0);

            // Draw bounding boxes
            const ctx = bboxCanvas.getContext('2d');
            ctx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height);

            // Compute scale: detections bbox coords are relative to the ROI image
            // that was processed by OCR. The crop image from frontend IS the ROI.
            // No additional scaling needed if previewCanvas matches crop dimensions.

            detections.forEach(det => {
                const [bx, by, bw, bh] = det.bbox;
                let color, label;
                if (isNutrisiLabel(det.text)) {
                    color = '#4CAF50'; label = '✅ nutrisi';
                } else if (isNumericValue(det.text)) {
                    color = '#FFC107'; label = '🔢 nilai';
                } else if (det.conf < 0.6) {
                    color = '#FF80AB'; label = '❌ rendah';
                } else {
                    color = '#FF80AB'; label = '❌ lainnya';
                }
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.strokeRect(bx, by, bw, bh);
                // Small label background
                ctx.fillStyle = color;
                const fontSize = Math.max(10, Math.min(14, bh * 0.4));
                ctx.font = `bold ${fontSize}px Outfit, sans-serif`;
                const textW = ctx.measureText(det.text.slice(0, 20)).width + 6;
                ctx.globalAlpha = 0.85;
                ctx.fillRect(bx, Math.max(0, by - fontSize - 4), textW, fontSize + 4);
                ctx.globalAlpha = 1;
                ctx.fillStyle = '#fff';
                ctx.fillText(det.text.slice(0, 20), bx + 3, Math.max(fontSize, by - 4));
            });
        };
        img.src = window._lastCropDataURL;

        // Render detection list
        const listEl = $('ocrDetectionList');
        if (listEl) {
            listEl.innerHTML = detections.map(det => {
                let icon, cat;
                if (isNutrisiLabel(det.text)) { icon = '✅'; cat = 'nutrisi'; }
                else if (isNumericValue(det.text)) { icon = '🔢'; cat = 'nilai'; }
                else { icon = '❌'; cat = 'ignored'; }
                return `<div style="padding:3px 0; border-bottom:1px solid rgba(0,0,0,0.04);">`
                    + `<span style="color:var(--text-muted);">[${det.conf.toFixed(2)}]</span> `
                    + `<strong>${det.text}</strong>`
                    + ` → <span style="font-size:0.72rem;">${icon} ${cat}</span></div>`;
            }).join('');
        }

        // Anchor & duration status
        const anchorEl = $('ocrAnchorStatus');
        if (anchorEl) anchorEl.textContent = payload.pass1_anchor_found ? '✅ Ditemukan' : '⚠️ Tidak ditemukan';
        const durasiEl = $('ocrDurasi');
        if (durasiEl) durasiEl.textContent = payload.durasi_detik || '—';
    }

    function parsePaddleOCR(payload) {
        const data = payload && payload.data ? payload.data : {};
        const q = payload && payload.quality ? payload.quality : null;
        const sf = payload && payload.semantic_filter ? payload.semantic_filter : null;
        if(q) {
            const level = q.usable ? 'good' : (q.quality_score >= 0.35 ? 'warn' : 'bad');
            let hint = `Quality score ${Math.round(q.quality_score * 100)}%. `;
            if(q.warnings && q.warnings.length) hint += `Catatan: ${q.warnings.join(', ')}. `;
            if(sf && sf.enabled) hint += `Filter nutrisi aktif, ${sf.removed} teks non-gizi diabaikan.`;
            updateQualityUi('ocr', level, q.usable ? 'Kualitas scan cukup' : 'Kualitas scan perlu diperbaiki', hint);
            const card = $('ocrQualityCard');
            if(card) card.style.display = 'block';
        }
        setNumberInput('inpTakaran', payload ? payload.takaran_saji_g : null);
        setNumberInput('inpEnergi', data.kalori ? data.kalori.per_sajian : null);
        setNumberInput('inpGula', data.gula ? data.gula.per_sajian : null);
        setNumberInput('inpLemak', data.lemak_jenuh ? data.lemak_jenuh.per_sajian : null);
        setNumberInput('inpSodium', data.natrium ? data.natrium.per_sajian : null);
        setNumberInput('inpProtein', data.protein ? data.protein.per_sajian : null);
        setNumberInput('inpFiber', data.serat ? data.serat.per_sajian : null);

        $('inpFoodName').value = 'Produk hasil OCR';

        // Render bounding box overlay
        renderOcrBboxOverlay(payload);

        switchView('viewCorrection');
    }

    // --- OCR Ekstrak dari Crop ---
    $('btnConfirmCrop').addEventListener('click', async () => {
        if(!cropperInstance) return;

        $('scanLoading').style.display = 'block';
        $('cropActionBtns').style.display = 'none';
        
        const croppedCanvas = cropperInstance.getCroppedCanvas();

        // Simpan crop DataURL untuk bbox overlay nanti
        window._lastCropDataURL = croppedCanvas.toDataURL('image/jpeg', 0.92);

        try {
            const blob = await canvasToBlob(croppedCanvas);
            if (!blob) throw new Error('Gagal membuat gambar hasil crop.');

            const formData = new FormData();
            formData.append('file', blob, 'nutrilabel-crop.jpg');

            const response = await fetchOcrWithTimeout(OCR_API_URL, {
                method: 'POST',
                body: formData
            });
            if (!response.ok) {
                throw new Error(`OCR API error ${response.status}`);
            }

            const payload = await response.json();
            parsePaddleOCR(payload);
        } catch(e) {
            console.error(e);
            $('scanLoading').style.display = 'none';
            $('cropActionBtns').style.display = 'flex';
            alert("Gagal memproses OCR: " + e.message + "\nPastikan server FastAPI berjalan di http://127.0.0.1:8000");
            switchView('viewCorrection');
        } finally {
            $('scanLoading').style.display = 'none';
            $('cropActionBtns').style.display = 'flex';
        }
    });

    // --- Toggle Bounding Box Visibility ---
    $('btnToggleBbox').addEventListener('click', function() {
        const bboxCanvas = $('ocrBboxCanvas');
        if (!bboxCanvas) return;
        if (bboxCanvas.style.display === 'none') {
            bboxCanvas.style.display = 'block';
            this.textContent = '👁 Sembunyikan Kotak';
        } else {
            bboxCanvas.style.display = 'none';
            this.textContent = '👁 Tampilkan Kotak';
        }
    });

    function parseOCR(text) {
        const extract = (patterns) => {
            for (const p of patterns) {
                const m = text.match(p);
                if (m) return m[1].replace(',','.');
            }
            return '';
        };

        $('inpTakaran').value = extract([/(?:takaran saji|serving size)\s*:?\s*(\d+[\.,]?\d*)/i]);
        $('inpEnergi').value = extract([/(?:energi total|energi|energy|kalori|calories)\s*:?\s*(\d+[\.,]?\d*)/i]);
        $('inpGula').value = extract([/(?:gula|sugars?|sugar total)\s*:?\s*(\d+[\.,]?\d*)/i]);
        $('inpLemak').value = extract([/(?:lemak jenuh|saturated fat)\s*:?\s*(\d+[\.,]?\d*)/i]);
        $('inpSodium').value = extract([/(?:natrium|sodium|garam|salt)\s*:?\s*(\d+[\.,]?\d*)/i]);
        $('inpProtein').value = extract([/(?:protein)\s*:?\s*(\d+[\.,]?\d*)/i]);
        $('inpFiber').value = extract([/(?:serat|fiber|dietary fiber)\s*:?\s*(\d+[\.,]?\d*)/i]);
        $('inpFoodName').value = 'Makanan Terdeteksi';

        switchView('viewCorrection');
    }

    // --- Grade (use NutriScore module for official Nutri-Score colors + local pastel for dashboard) ---
    const GRADE_COLORS = window.NutriScore.GRADE_COLORS;
    const GRADE_COLORS_PASTEL = { 'A':'#48C78E', 'B':'#8ED96C', 'C':'#F4C45A', 'D':'#EF7768', 'E':'#D94848' };
    const GRADE_DESC = window.NutriScore.GRADE_LABELS;

    function generateExpertExplanation(grade, nP, pP, m) {
        const verdicts = {
            'A': 'Pilihan bergizi tinggi! 🌟',
            'B': 'Pilihan yang cukup baik! 👍',
            'C': 'Perlu dibatasi porsinya 🟡',
            'D': 'Tinggi kandungan kurang sehat 🟠',
            'E': 'Konsumsi dengan sangat bijak ⚠️'
        };
        
        let bullets = [];
        if (m.s >= 7) bullets.push("Kandungan gula SANGAT tinggi, waspada lonjakan gula darah.");
        else if (m.s >= 4) bullets.push("Kandungan gula lumayan tinggi.");
        
        if (m.f >= 7) bullets.push("Lemak jenuh terlalu pekat, waspada kesehatan jantung.");
        else if (m.f >= 4) bullets.push("Lemak jenuh cukup mendominasi dan perlu dikontrol.");
        
        if (m.na >= 6) bullets.push("Kadar garam (sodium) tinggi, rentan memicu hipertensi.");
        
        if (m.fb >= 3 && m.pr >= 3) bullets.push("Bagusnya, kombinasi serta dan protein bantu rasa kenyang lebih lama.");
        else if (m.fb >= 3) bullets.push("Poin plus: tinggi serat untuk membantu pencernaan.");
        else if (m.pr >= 3) bullets.push("Poin plus: terdapat protein lumayan untuk asupan otot.");
        
        if (bullets.length === 0) {
            if(nP <= 3) bullets.push("Secara total, komponen gizi kurang baik di kategori rendah (bagus).");
            else bullets.push("Gizi negatif merata tanpa ada angka yang terlalu ekstrem.");
        }
        
        const tips = {
            'A':'Keren! Pertahankan asupan sebaik ini sehari-hari.',
            'B':'Boleh jadi referensi rutin, sambil cek batas kalori total.',
            'C':'Pastikan kombinasikan dengan buah/sayur utuh.',
            'D':'Pertimbangkan cari alternatif produk "Less Sugar/Fat".',
            'E':'Jadikan ini cemilan iseng belaka, bukan kebiasaan!'
        };
        
        const bulletHtml = bullets.slice(0,3).map(b => `<li style="margin-bottom:4px;">${b}</li>`).join('');
        return `
            <div style="font-weight:800; color:var(--grade-${grade}); margin-bottom:8px; font-size:1.05rem;">Grade ${grade} — ${verdicts[grade]}</div>
            <ul style="margin-left:18px; margin-bottom:12px; color:var(--text-muted); font-size:0.9rem;">
                ${bulletHtml}
            </ul>
            <div style="background:rgba(0,0,0,0.03); padding:10px 12px; border-radius:10px; font-size:0.85rem;">
                <strong style="color:var(--text-dark);">💡 Tips:</strong> <span style="color:var(--text-muted);">${tips[grade]}</span>
            </div>
        `;
    }

    let currentCalculated = null;

    $('btnCalculate').addEventListener('click', () => {
        const isBev = $('inpIsBev').value === 'true';
        const takaran = parseFloat($('inpTakaran').value) || 0;
        const energi = parseFloat($('inpEnergi').value) || 0;
        const gula = parseFloat($('inpGula').value) || 0;
        const lemak = parseFloat($('inpLemak').value) || 0;
        const sodium = parseFloat($('inpSodium').value) || 0;
        const protein = parseFloat($('inpProtein').value) || 0;
        const fiber = parseFloat($('inpFiber').value) || 0;
        const fvnPct = parseFloat($('inpFvn') ? $('inpFvn').value : 0) || 0;
        const name = $('inpFoodName').value || 'Produk';

        // NEW INPUTS
        const isPowder = $('inpIsPowder') && $('inpIsPowder').checked;
        const waterMl = parseFloat($('inpWaterMl') ? $('inpWaterMl').value : 0) || 0;
        const porsiFraksi = parseFloat($('inpPorsiFraksi') ? $('inpPorsiFraksi').value : 1) || 1;

        if(takaran <= 0) { alert("Harap isi Takaran Saji yang valid (gram/ml)!"); return; }

        let mp = 100 / takaran;
        let consumedServing = takaran * porsiFraksi;

        if (isPowder) {
            mp = 100 / (takaran + waterMl);
            consumedServing = (takaran + waterMl) * porsiFraksi;
        }

        // Normalize to per 100g or 100ml
        const nutrition100 = {
            energy_kj: energi * mp * 4.184,
            sugar_g: gula * mp,
            saturated_fat_g: lemak * mp,
            sodium_mg: sodium * mp,
            fiber_g: fiber * mp,
            protein_g: protein * mp,
            fvn_percent: fvnPct
        };

        const micronutrients = {
            vitamin_c_mg: (parseFloat($('inpVitC') && $('inpVitC').value) || 0) * mp,
            calcium_mg: (parseFloat($('inpCalcium') && $('inpCalcium').value) || 0) * mp,
            iron_mg: (parseFloat($('inpIron') && $('inpIron').value) || 0) * mp,
            folate_mcg: (parseFloat($('inpFolate') && $('inpFolate').value) || 0) * mp,
            vitamin_b1_mg: (parseFloat($('inpVitB1') && $('inpVitB1').value) || 0) * mp,
            vitamin_b6_mg: (parseFloat($('inpVitB6') && $('inpVitB6').value) || 0) * mp,
            vitamin_b12_mcg: (parseFloat($('inpVitB12') && $('inpVitB12').value) || 0) * mp
        };

        // Clinical evaluation
        let clinicalEval = null;
        if (clinicalEngine) {
            clinicalEval = clinicalEngine.evaluateMeal(nutrition100, consumedServing);
        }

        // Use NutriScore module
        const context = { actualConsumed: clinicalEval, porsiFraksi };
        const result = window.NutriScore.calculateNutriScore(nutrition100, isBev || isPowder, micronutrients, context);
        const finalGrade = result.grade;
        const score = result.score;
        const nP = result.n_points;
        const pP = result.p_points;
        const formulaStr = `${nP.total} - ${pP.total} = ${score}`;

        // Update UI
        const badge = $('finalGradeBadge');
        badge.textContent = finalGrade;
        badge.style.background = result.color;
        badge.className = 'grade-letter pulse';
        $('gradeShowcase').style.color = result.color;
        $('finalGradeTitle').textContent = result.label;
        $('resScore').textContent = score;

        // Nutrient Highlights (using personalized limits)
        const hlCalc = (val, max) => Math.min(200, (val/max)*100).toFixed(0);
        const hlColor = (pct) => pct <= 20 ? 'nh-green' : (pct <= 40 ? 'nh-yellow' : 'nh-red');
        const nHls = [
            { id: 'hlEnergi', pct: hlCalc(energi * porsiFraksi, state.limitCalorie) },
            { id: 'hlSatFat', pct: hlCalc(lemak * porsiFraksi, state.limitSaturatedFat) },
            { id: 'hlSugar', pct: hlCalc(gula * porsiFraksi, state.limitSugar) },
            { id: 'hlSodium', pct: hlCalc(sodium * porsiFraksi, state.limitSodium) }
        ];
        nHls.forEach(h => {
            const el = $(h.id);
            if(el) { el.textContent = `${h.pct}%`; el.className = `nh-val ${hlColor(h.pct)}`; }
        });

        // Detail Breakdown
        const setTxt = (id, val) => { if($(id)) $(id).textContent = val };
        setTxt('ptsEnergi', nP.energy); setTxt('ptsSatFat', nP.saturated_fat);
        setTxt('ptsSugar', nP.sugar); setTxt('ptsSodium', nP.sodium);
        setTxt('sumN', nP.total);
        setTxt('ptsFiber', pP.fiber); setTxt('ptsProtein', pP.protein);
        setTxt('ptsFvn', pP.fvn); setTxt('sumP', pP.total);
        setTxt('formulaDetail', formulaStr);

        // Expert Explanation with NutriScore warnings + clinical warnings
        const explainUi = $('expertExplanationContent');
        if(explainUi) {
            explainUi.innerHTML = generateExpertExplanation(
                finalGrade, nP.total, pP.total,
                { e: nP.energy, f: nP.saturated_fat, s: nP.sugar, na: nP.sodium, fb: pP.fiber, pr: pP.protein }
            );

            // Append clinical warnings
            if (clinicalEval && clinicalEval.clinical_warnings.length > 0) {
                const statusColors = { SAFE:'#1a9641', WARNING:'#fdae61', DANGER:'#d7191c' };
                let clinicalHtml = `<div style="margin-top:16px; padding:14px; border-radius:14px; background:${statusColors[clinicalEval.status]}15; border:1px solid ${statusColors[clinicalEval.status]}30;">`;
                clinicalHtml += `<h4 style="font-size:0.9rem; font-weight:800; color:${statusColors[clinicalEval.status]}; margin-bottom:8px;">🩺 Evaluasi Klinis — ${clinicalEval.status}</h4>`;
                clinicalHtml += '<ul style="margin-left:16px; font-size:0.85rem; color:var(--text-muted);">';
                clinicalEval.clinical_warnings.forEach(w => clinicalHtml += `<li style="margin-bottom:4px;">${w}</li>`);
                clinicalHtml += '</ul></div>';
                explainUi.innerHTML += clinicalHtml;
            }
        }

        // Porsi Info & Powder Info
        const porsiInfo = $('porsiInfo');
        if (porsiInfo) {
            if (porsiFraksi < 1.0) porsiInfo.textContent = `Porsi dikonsumsi: ${porsiFraksi}x sajian`;
            else porsiInfo.textContent = '';
        }
        const powderInfo = $('powderInfo');
        if (powderInfo) {
            if (isPowder) powderInfo.textContent = `💧 Mode serbuk: Dilarutkan dalam ${waterMl}ml air`;
            else powderInfo.textContent = '';
        }

        // Micro points display
        const rowMicro = $('rowMicro');
        const ptsMicro = $('ptsMicro');
        if (rowMicro && ptsMicro && result.micro_points) {
            if (result.micro_points.total > 0) {
                ptsMicro.textContent = result.micro_points.total;
                rowMicro.style.display = 'flex';
            } else {
                rowMicro.style.display = 'none';
            }
        }

        // Smart tips display
        const smartTipsCard = $('smartTipsCard');
        const smartTipsContent = $('smartTipsContent');
        if (smartTipsCard && smartTipsContent && result.tips && result.tips.length > 0) {
            smartTipsContent.innerHTML = '<ul style="margin-left:18px; padding-left:0;">' + result.tips.map(t => `<li style="margin-bottom:4px;">${t}</li>`).join('') + '</ul>';
            smartTipsCard.style.display = 'block';
        } else if (smartTipsCard) {
            smartTipsCard.style.display = 'none';
        }

        currentCalculated = normalizeEntry({
            id: Date.now().toString(), name,
            grade: finalGrade, sugar: gula * porsiFraksi, cal: energi * porsiFraksi, fat: lemak * porsiFraksi,
            sodium: sodium * porsiFraksi, protein: protein * porsiFraksi, fiber: fiber * porsiFraksi, servingSize_g: consumedServing,
            date: new Date().toISOString(),
            nutriScore: { grade: finalGrade, score, n_points: nP, p_points: pP, warnings: result.warnings, tips: result.tips },
            clinicalEval: clinicalEval
        });

        $('modalResult').classList.add('active');
    });

    $('btnCloseModal').addEventListener('click', () => {
        $('modalResult').classList.remove('active');
    });

    $('btnConfirmEat').addEventListener('click', () => {
        if(currentCalculated) {
            state.history.unshift(currentCalculated);
            saveState();
            haptic('success');
            launchConfetti();
            // Show progress toast
            const todayStr = getLocalDateStr();
            const todayCal = state.history.filter(h => getLocalDateStr(h.date) === todayStr).reduce((a,h) => a + (h.cal||0), 0);
            const maxCal = state.limitCalorie || 2000;
            const pct = Math.min(100, Math.round((todayCal/maxCal)*100));
            showToast(`✅ Tercatat! Asupan hari ini: ${todayCal} kkal (${pct}% target)`);
        }
        $('modalResult').classList.remove('active');
        switchView('viewDashboard');
    });

    // --- User Profile / Onboarding ---

    // Step navigation
    $('btnToStep2').addEventListener('click', () => {
        const age = parseInt($('userAge').value);
        const weight = parseFloat($('userWeight').value);
        const height = parseFloat($('userHeight').value);
        if(!age || !weight || !height) { alert("Harap lengkapi semua data fisik!"); return; }
        $('onboardStep1').style.display = 'none';
        $('onboardStep2').style.display = 'block';
        $$('.step-dot').forEach(d => d.classList.remove('active'));
        document.querySelector('.step-dot[data-step="2"]').classList.add('active');
    });

    $('btnBackStep1').addEventListener('click', () => {
        $('onboardStep2').style.display = 'none';
        $('onboardStep1').style.display = 'flex';
        $$('.step-dot').forEach(d => d.classList.remove('active'));
        document.querySelector('.step-dot[data-step="1"]').classList.add('active');
    });

    // Show/hide diabetes sub-checks
    $('chkDiabetes').addEventListener('change', (e) => {
        $('diabetesSubChecks').style.display = e.target.checked ? 'block' : 'none';
        if(!e.target.checked) {
            $('chkNephropathy').checked = false;
            $('chkDyslipidemia').checked = false;
        }
    });

    $('btnSaveProfile').addEventListener('click', () => {
        const gender = $('userGender').value;
        const age = parseInt($('userAge').value);
        const weight = parseFloat($('userWeight').value);
        const height = parseFloat($('userHeight').value);
        const activityLevel = $('userActivityLevel') ? $('userActivityLevel').value : 'moderate';

        if(!age || !weight || !height) { alert("Harap lengkapi semua data fisik!"); return; }

        const heightM = height / 100;
        const bmi = weight / (heightM * heightM);
        let bmiStatus = 'Normal';
        if(bmi < 18.5) bmiStatus = 'Kurus';
        else if(bmi >= 25 && bmi < 29.9) bmiStatus = 'Gemuk';
        else if(bmi >= 30) bmiStatus = 'Obesitas';

        let bmr = (10 * weight) + (6.25 * height) - (5 * age);
        if(gender === 'male') bmr += 5; else bmr -= 161;

        const tdee = bmr * 1.2;
        const isDiabetic = $('chkDiabetes').checked;
        const hasNephropathy = $('chkNephropathy').checked;
        const hasDyslipidemia = $('chkDyslipidemia').checked;
        const isHypertensive = $('chkHypertension').checked;

        // Calculate hydration target
        const hydrationTarget = window.HydrationEngine
            ? window.HydrationEngine.calculateDailyTarget(weight, activityLevel)
            : { target_ml: Math.round(weight * 35), target_liters: parseFloat((weight * 35 / 1000).toFixed(1)), target_glasses: Math.ceil(weight * 35 / 250) };

        state.user = {
            gender, age, weight_kg: weight, height_cm: height,
            bmi, bmiStatus, bmr, tdee,
            isDiabetic, hasNephropathy, hasDyslipidemia, isHypertensive,
            activity_level: activityLevel,
            hydration_target_ml: hydrationTarget.target_ml,
            hydration_target_liters: hydrationTarget.target_liters,
            hydration_target_glasses: hydrationTarget.target_glasses
        };

        // Use ClinicalHealthRules for personalized limits
        clinicalEngine = window.ClinicalHealthRules.create({
            age, gender, weight_kg: weight, height_cm: height,
            bmr, tdee, isDiabetic, hasNephropathy, hasDyslipidemia, isHypertensive
        });
        const limits = clinicalEngine.getDailyLimits();
        state.user.daily_limits = limits;

        state.limitCalorie = limits.max_snack_calories;
        state.limitSugar = limits.max_sugar_g;
        state.limitSodium = limits.max_sodium_mg;
        state.limitSaturatedFat = limits.max_saturated_fat_g;

        saveState();

        $('mainHeader').style.display = 'flex';
        $('mainNav').style.display = 'flex';
        switchView('viewDashboard');
    });

    // --- Dashboard Updaters ---
    function updateDashboard() {
        const todayStr = getLocalDateStr();
        const todayItems = state.history.filter(h => getLocalDateStr(h.date) === todayStr);

        const sumSugar = todayItems.reduce((acc, h) => acc + (h.sugar||0), 0);
        const sumCal = todayItems.reduce((acc, h) => acc + (h.cal||h.calories||0), 0);
        const sumFat = todayItems.reduce((acc, h) => acc + (h.fat||h.fat_g||0), 0);
        const sumProtein = todayItems.reduce((acc, h) => acc + (h.protein||h.protein_g||0), 0);
        const sumCarb = todayItems.reduce((acc, h) => acc + (h.carbohydrates_g||h.carb||0), 0);

        // Update Combined Dashboard Card (Realfood + Snacks + Hydration)
        const limits = (state.user && state.user.daily_limits) || {
            max_calories_kkal: 2000,
            max_sugar_g: 50,
            max_saturated_fat_g: 22,
            max_sodium_mg: 2000,
            max_protein_g: 60,
            max_carbs_g: 300,
            max_fat_g: 67
        };
        const maxCal = limits.max_calories_kkal || 2000;
        const maxProtein = limits.max_protein_g || 60;
        const maxFat = limits.max_fat_g || 67;
        const maxCarb = limits.max_carbs_g || 300;

        if ($('combinedCal')) {
            $('combinedCal').textContent = `${sumCal.toFixed(0)} / ${maxCal.toFixed(0)} kkal`;
        }
        
        let waterConsumed = 0;
        let waterTarget = 2000;
        if (window.HydrationEngine) {
            waterTarget = window.HydrationEngine.calculateDailyTarget(state.user ? state.user.weight_kg || 60 : 60, state.user ? state.user.activity_level || 'moderate' : 'moderate').target_ml;
            waterConsumed = window.HydrationEngine.getTodayTotal(waterTarget);
        }
        if ($('combinedWater')) {
            $('combinedWater').textContent = `${waterConsumed} / ${waterTarget} mL`;
        }
        
        if ($('combinedProteinText')) {
            $('combinedProteinText').textContent = `${sumProtein.toFixed(1)}g / ${maxProtein}g`;
            const pctPro = Math.min(100, (sumProtein / maxProtein) * 100);
            $('combinedProteinBar').style.width = `${pctPro}%`;
        }
        if ($('combinedFatText')) {
            $('combinedFatText').textContent = `${sumFat.toFixed(1)}g / ${maxFat}g`;
            const pctFat = Math.min(100, (sumFat / maxFat) * 100);
            $('combinedFatBar').style.width = `${pctFat}%`;
        }
        if ($('combinedCarbText')) {
            $('combinedCarbText').textContent = `${sumCarb.toFixed(1)}g / ${maxCarb}g`;
            const pctCarb = Math.min(100, (sumCarb / maxCarb) * 100);
            $('combinedCarbBar').style.width = `${pctCarb}%`;
        }

        // Reinitialize clinical engine from saved profile
        if (state.user && !clinicalEngine) {
            const u = state.user;
            clinicalEngine = window.ClinicalHealthRules.create({
                age: u.age, gender: u.gender, weight_kg: u.weight_kg || u.weight,
                height_cm: u.height_cm || u.height, bmr: u.bmr, tdee: u.tdee,
                isDiabetic: u.isDiabetic || false, hasNephropathy: u.hasNephropathy || false,
                hasDyslipidemia: u.hasDyslipidemia || false, isHypertensive: u.isHypertensive || false
            });
        }

        // Clinical badges
        const badgesEl = $('clinicalBadges');
        if (badgesEl && state.user) {
            const u = state.user;
            let badges = '';
            if (u.isDiabetic) badges += '<span class="clinical-badge badge-diabetes">🩺 Mode Diabetes</span>';
            if (u.isHypertensive) badges += '<span class="clinical-badge badge-hypertension">💊 Mode Hipertensi</span>';
            if (u.hasNephropathy) badges += '<span class="clinical-badge badge-kidney">⚕️ Mode Ginjal</span>';
            if (u.hasDyslipidemia) badges += '<span class="clinical-badge badge-lipid">🫀 Dislipidemia</span>';

            // Blood pressure target
            if ((u.isDiabetic || u.isHypertensive) && clinicalEngine) {
                const bpTarget = clinicalEngine.getBloodPressureTarget();
                badges += `<div class="bp-target-info">${bpTarget}</div>`;
            }

            if (badges) { badgesEl.innerHTML = badges; badgesEl.style.display = 'flex'; }
            else { badgesEl.style.display = 'none'; }
        }

        if(state.user) {
            const b = state.user.bmi;
            $('bmiBadge').textContent = `BMI: ${b.toFixed(1)} (${state.user.bmiStatus})`;
            if(b < 18.5 || b >= 25) $('bmiBadge').style.color = "var(--grade-D)";
            else $('bmiBadge').style.color = "var(--grade-A)";
        }

        // UI Circle Progress
        const maxS = state.limitSugar;
        const pct = Math.min(100, (sumSugar / maxS) * 100);
        
        $('dashboardSugar').textContent = sumSugar.toFixed(1);
        $('sugarUnitLabel').textContent = `/ ${maxS.toFixed(0)}g`;

        const circle = $('sugarCircleVal');
        
        // Cirle circumference = 2 * PI * r = 2 * 3.14 * 40 = 251.2
        const dashOffset = 251.2 - (251.2 * pct / 100);
        circle.style.strokeDashoffset = dashOffset;

        let scColor = 'var(--grade-A)';
        let scMsg = 'Gula seimbang! Pertahankan 🥑';
        if(pct >= 100) { scColor = 'var(--grade-D)'; scMsg = 'Bahaya! Gula Melebihi Batas Harian 🚨'; }
        else if(pct > 75) { scColor = 'var(--grade-C)'; scMsg = 'Hati-hati, sudah mendekati batas ⚠️'; }
        
        circle.style.stroke = scColor;
        $('sugarMessage').textContent = scMsg;
        if(pct > 75) $('sugarMessage').style.color = scColor;
        else $('sugarMessage').style.color = 'var(--text-dark)';

        // UI Quick Stats
        const maxC = state.limitCalorie;
        $('dashboardCal').innerHTML = `${sumCal.toFixed(0)} <span style="font-size:0.75rem; color:var(--text-muted);">/ ${maxC.toFixed(0)} kkal</span>`;
        const maxF = state.limitSaturatedFat;
        $('dashboardFat').innerHTML = `${sumFat.toFixed(1)} <span style="font-size:0.75rem; color:var(--text-muted);">/ ${maxF.toFixed(0)} g</span>`;

        // Sodium & Protein stats
        const sumSodium = todayItems.reduce((a,h) => a + (h.sodium||0), 0);
        if ($('dashboardSodium')) {
            $('dashboardSodium').innerHTML = `${sumSodium.toFixed(0)} <span style="font-size:0.75rem; color:var(--text-muted);">/ ${state.limitSodium} mg</span>`;
        }
        if ($('dashboardProtein')) {
            const maxPro = (state.user && state.user.daily_limits && state.user.daily_limits.max_protein_g)
                ? state.user.daily_limits.max_protein_g.toFixed(0) + 'g'
                : '—';
            $('dashboardProtein').innerHTML = `${sumProtein.toFixed(1)} <span style="font-size:0.75rem; color:var(--text-muted);">/ ${maxPro}</span>`;
        }

        // UI Dashboard History
        const list = $('dashboardHistory');
        if(todayItems.length === 0) {
            list.innerHTML = `<div class="empty-state">Belum ada makanan hari ini. Yuk mulai scan! 📸</div>`;
        } else {
            list.innerHTML = todayItems.slice(0, 3).map((h, idx) => {
                const badgeIcon = h.grade || '🍽️';
                const badgeBg = h.grade ? (GRADE_COLORS_PASTEL[h.grade] || 'rgba(0,0,0,0.05)') : 'rgba(52,152,219,0.2)';
                const calVal = h.cal || 0;
                return `
                <div class="history-item" data-history-index="${state.history.indexOf(h)}">
                    <div class="history-icon-badge" style="background:${badgeBg}; box-shadow: 0 4px 12px ${badgeBg}60;">${badgeIcon}</div>
                    <div class="history-details">
                        <div class="history-title">${h.name}</div>
                        <div class="history-meta">+${calVal}kkal ${h.sugar ? '• ' + h.sugar + 'g gula' : ''}</div>
                    </div>
                    <button class="history-delete-btn" onclick="event.stopPropagation(); window._deleteHistoryItem(${state.history.indexOf(h)}, this)" title="Hapus">🗑️</button>
                </div>`;
            }).join('');
        }

        updateStreakAndChart();

        // Global list in History Tab & Gabungan Statistik
        const viewHist = $('viewHistory');
        if(viewHist) {
            // Calculate Gabungan Statistik for Today
            const totalC = todayItems.reduce((acc, h) => acc + (h.cal||0), 0);
            const totalP = todayItems.reduce((acc, h) => acc + (h.protein||0), 0);
            const totalL = todayItems.reduce((acc, h) => acc + (h.fat||0), 0);
            const totalK = todayItems.reduce((acc, h) => acc + (h.carbohydrates_g||0), 0);
            
            let hydrationText = "0 mL";
            if (window.HydrationEngine) {
                const waterTarget = window.HydrationEngine.calculateDailyTarget(state.user ? state.user.weight_kg || 60 : 60, state.user ? state.user.activity_level || 'moderate' : 'moderate').target_ml;
                const waterConsumed = window.HydrationEngine.getTodayTotal(waterTarget);
                hydrationText = `${waterConsumed} mL / ${waterTarget} mL`;
            }

            const statsHtml = `
            <h2 style="margin-top:12px;">📊 Gabungan Statistik Hari Ini</h2>
            <p style="color:var(--text-muted); font-size:0.85rem; margin-bottom:16px;">Ringkasan seluruh nutrisi dan hidrasi Anda hari ini.</p>
            <div class="glass-card" style="margin-bottom:24px;">
                <div style="display:flex; justify-content:space-between; align-items:flex-end; border-bottom:1px solid rgba(0,0,0,0.05); padding-bottom:12px; margin-bottom:12px;">
                    <div>
                        <div style="font-size:0.8rem; color:var(--text-muted); font-weight:700;">TOTAL KALORI</div>
                        <div style="font-size:1.8rem; font-weight:800; color:var(--primary); line-height:1;">${totalC.toFixed(0)} <span style="font-size:0.9rem; color:var(--text-dark);">kkal</span></div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.8rem; color:var(--text-muted); font-weight:700;">HIDRASI</div>
                        <div style="font-size:1.1rem; font-weight:800; color:#0984e3;">${hydrationText} 💧</div>
                    </div>
                </div>
                <div style="display:flex; justify-content:space-between; text-align:center;">
                    <div style="flex:1;">
                        <div style="font-size:1.1rem; font-weight:800; color:var(--text-dark);">${totalP.toFixed(1)}g</div>
                        <div style="font-size:0.75rem; color:var(--text-muted); font-weight:600;">PROTEIN</div>
                    </div>
                    <div style="flex:1; border-left:1px solid rgba(0,0,0,0.05); border-right:1px solid rgba(0,0,0,0.05);">
                        <div style="font-size:1.1rem; font-weight:800; color:var(--text-dark);">${totalL.toFixed(1)}g</div>
                        <div style="font-size:0.75rem; color:var(--text-muted); font-weight:600;">LEMAK</div>
                    </div>
                    <div style="flex:1;">
                        <div style="font-size:1.1rem; font-weight:800; color:var(--text-dark);">${totalK.toFixed(1)}g</div>
                        <div style="font-size:0.75rem; color:var(--text-muted); font-weight:600;">KARBO</div>
                    </div>
                </div>
            </div>
            `;

            if(state.history.length === 0) {
                viewHist.innerHTML = statsHtml + `<h2 style="margin-bottom:16px;">Semua Riwayat</h2><div class="empty-state">Belum ada catatan makanan. Yuk mulai! 🍽️</div>`;
            } else {
                viewHist.innerHTML = statsHtml + `<h2 style="margin-bottom:16px;">Semua Riwayat</h2>` + state.history.slice().reverse().map((h, rIdx) => {
                    const d = new Date(h.date);
                    const isManual = h.type === 'Pencarian Manual';
                    const badgeIcon = isManual ? '📖' : (h.grade || '🍽️');
                    const badgeBg = isManual ? 'rgba(52, 152, 219, 0.2)' : (h.grade ? GRADE_COLORS_PASTEL[h.grade] : 'rgba(0,0,0,0.05)');
                    const kkal = h.cal || 0;
                    const realIdx = state.history.length - 1 - rIdx;
                    
                    return `
                    <div class="history-item" data-history-index="${realIdx}">
                        <div class="history-icon-badge" style="background:${badgeBg};">${badgeIcon}</div>
                        <div class="history-details">
                            <div class="history-title">${h.name}</div>
                            <div class="history-meta">${d.getDate()}/${d.getMonth()+1} • +${kkal}kkal ${h.sugar ? `• ${h.sugar}g Gula` : ''}</div>
                        </div>
                        <button class="history-delete-btn" onclick="event.stopPropagation(); window._deleteHistoryItem(${realIdx}, this)" title="Hapus">🗑️</button>
                    </div>`;
                }).join('');
            }
        }

        // Update greeting card
        updateGreeting(sumCal, limits);
    }

    function updateStreakAndChart() {
        if(!state.user) return;
        const limitS = state.limitSugar;
        
        // 7 days backwards loop
        const days = [];
        const sugarData = [];
        for(let i=6; i>=0; i--) {
            const d = new Date();
            d.setDate(d.getDate() - i);
            const dStr = getLocalDateStr(d);
            const shortD = `${d.getDate()}/${d.getMonth()+1}`;
            
            // Total sum of that day
            const dailySum = state.history
                .filter(h => getLocalDateStr(h.date) === dStr)
                .reduce((acc, h) => acc + (h.sugar||0), 0);

            days.push(shortD);
            sugarData.push(dailySum);
        }

        // Streak Engine
        let streak = 0;
        for(let i=6; i>=0; i--) {
            // Evaluasi mundur bertahap
            if(sugarData[i] > limitS) { streak = 0; } 
            else { streak++; }
        }

        const sTitle = $('streakTitle');
        const sDesc = $('streakDesc');
        const sIcon = $('streakIcon');

        sTitle.textContent = `${streak} Hari Beruntun!`;
        if(streak >= 3) {
            sIcon.textContent = '🔥';
            sDesc.textContent = 'Luar biasa konsisten! Batas gula aman selalu terjaga.';
            sIcon.style.textShadow = '0 4px 12px rgba(244, 196, 90, 0.4)';
        } else if(streak >= 1) {
            sIcon.textContent = '🥑';
            sDesc.textContent = 'Awal yang baik! Ayo jaga tubuh sehatmu besok.';
            sIcon.style.textShadow = '0 4px 12px rgba(72, 199, 142, 0.4)';
        } else {
            sIcon.textContent = '🤕';
            sTitle.textContent = 'Gula Bocor!';
            sDesc.textContent = 'Jatah harianmu kemarin jebol. Ayo perbaiki hari ini.';
            sIcon.style.textShadow = '0 4px 12px rgba(239, 119, 104, 0.4)';
        }

        // --- Chart.js ---
        const ctx = document.getElementById('weeklyChart').getContext('2d');
        if(chartInstance) chartInstance.destroy();

        chartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: days,
                datasets: [
                    {
                        label: 'Gula Dikonsumsi (g)',
                        data: sugarData,
                        backgroundColor: sugarData.map(val => val > limitS ? '#EF7768' : '#48C78E'),
                        borderRadius: 4,
                        maxBarThickness: 14
                    },
                    {
                        label: 'Limit Gula',
                        data: Array(7).fill(limitS),
                        type: 'line',
                        borderColor: '#F4C45A',
                        borderWidth: 2,
                        pointRadius: 0,
                        borderDash: [4, 4],
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true, grid: { display:false }, ticks: { font: { size: 9, family: "'Outfit', sans-serif" } } },
                    x: { grid: { display:false }, ticks: { font: { size: 9, family: "'Outfit', sans-serif" } } }
                }
            }
        });

        // --- Nutri-Score Weekly Distribution Chart ---
        const nsCanvas = document.getElementById('weeklyNSChart');
        if (nsCanvas) {
            if (chartNSInstance) chartNSInstance.destroy();
            const gradeCounts = { A:0, B:0, C:0, D:0, E:0 };
            const weekAgo = new Date(); weekAgo.setDate(weekAgo.getDate() - 7);
            state.history.filter(h => new Date(h.date) >= weekAgo).forEach(h => {
                if (h.grade && gradeCounts.hasOwnProperty(h.grade)) gradeCounts[h.grade]++;
            });
            chartNSInstance = new Chart(nsCanvas.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: ['A','B','C','D','E'],
                    datasets: [{
                        data: [gradeCounts.A, gradeCounts.B, gradeCounts.C, gradeCounts.D, gradeCounts.E],
                        backgroundColor: ['#48C78E','#8ED96C','#F4C45A','#EF7768','#D94848'],
                        borderRadius: 6, maxBarThickness: 28
                    }]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { beginAtZero: true, ticks: { stepSize: 1, font: { size: 9, family: "'Outfit'" } }, grid: { display: false } },
                        x: { grid: { display: false }, ticks: { font: { size: 11, family: "'Outfit'", weight: 700 } } }
                    }
                }
            });
        }
    }

    // --- Scan Choice ---
    $('btnChoicePackaged').addEventListener('click', () => switchView('viewOcrScanner'));
    $('btnChoicePlate').addEventListener('click', async () => {
        switchView('viewIsiPiringku');
        const badge = $('segModelStatus');
        if(badge) {
            badge.textContent = '⏳ Mengecek AI...';
            badge.style.color = 'var(--text-muted)';
        }

        let yoloActive = false;
        try {
            const healthRes = await fetch('http://127.0.0.1:8000/api/health');
            if (healthRes.ok) {
                const healthData = await healthRes.json();
                if (healthData.yolo_model === 'ready') {
                    yoloActive = true;
                    if(badge) {
                        badge.textContent = '⚡ YOLOv8-Seg (GPU)';
                        badge.style.color = 'var(--grade-A)';
                    }
                }
            }
        } catch (e) {
            console.warn("FastAPI backend not running or no YOLO model found:", e);
        }

        if (!yoloActive && !window.seefoodModel && window.SeeFoodSegmenter) {
            try {
                await window.SeeFoodSegmenter.loadModel((status, msg) => {
                    if(badge) {
                        badge.textContent = status === 'ready' ? '✅ TFLite Lokal' : (status === 'error' ? '❌ Gagal' : '⏳ Memuat TFLite...');
                        badge.style.color = status === 'ready' ? 'var(--grade-A)' : (status === 'error' ? 'var(--grade-E)' : 'var(--text-muted)');
                    }
                });
            } catch (err) {
                console.error("Failed to load SeeFood segmenter", err);
            }
        } else if (!yoloActive && window.seefoodModel) {
            if(badge) {
                badge.textContent = '✅ TFLite Lokal';
                badge.style.color = 'var(--grade-A)';
            }
        }
    });
    
    if ($('btnChoiceLibrary')) {
        $('btnChoiceLibrary').addEventListener('click', () => {
            switchView('viewLibrary');
            renderLibrarySuggestions();
        });
    }

    // --- SeeFood Segmentation UI Logic ---
    function updatePlateSlider(key, value) {
        const slider = $('slider' + key);
        if(slider) {
            slider.value = value;
            $('val' + key).textContent = value.toFixed(0) + '%';
            const seg = $('plate' + key);
            if(seg) seg.style.flex = value;
        }
    }

    if($('btnPlateCamera')) {
        $('btnPlateCamera').addEventListener('click', () => $('plateCameraInput').click());
        $('btnPlateUpload').addEventListener('click', () => $('plateImgUpload').click());
        $('btnRetakePhoto').addEventListener('click', () => {
            $('segPhotoInput').style.display = 'block';
            $('segPreviewContainer').style.display = 'none';
            $('piringkuResult').style.display = 'none';
            $('segDebugPanel').style.display = 'none';
            $('sliderAutoLabel').style.display = 'none';
            $('yoloDetectionsContainer').style.display = 'none';
            if (window.SegmentationEditor) window.SegmentationEditor.clear();
        });

    // --- Interactive Segmentation Correction ---
    if ($('btnToggleCorrection')) {
        $('btnToggleCorrection').addEventListener('click', (e) => {
            const btn = e.target;
            const toolbar = $('correctionToolbar');
            const cCanvas = $('correctionCanvas');
            const hint = $('corrHint');
            
            btn.classList.toggle('active');
            if (btn.classList.contains('active')) {
                toolbar.classList.add('active');
                cCanvas.classList.add('active');
                if (!window._corrHintShown) {
                    hint.style.display = 'block';
                    window._corrHintShown = true;
                    const hideHint = () => { hint.style.display = 'none'; cCanvas.removeEventListener('pointerdown', hideHint); };
                    cCanvas.addEventListener('pointerdown', hideHint);
                    setTimeout(() => { hint.style.display = 'none'; }, 3000);
                }
            } else {
                toolbar.classList.remove('active');
                cCanvas.classList.remove('active');
            }
        });

        $$('.btn-corr-tool').forEach(btn => {
            btn.addEventListener('click', (e) => {
                $$('.btn-corr-tool').forEach(b => b.classList.remove('active'));
                const target = e.currentTarget;
                target.classList.add('active');
                const cat = target.dataset.cat;
                window.SegmentationEditor.setMode(cat);
                
                let label = "Penghapus";
                if(cat === 'makanan_pokok') label = "Karbo (Kuning)";
                else if(cat === 'lauk_pauk') label = "Lauk (Merah)";
                else if(cat === 'sayur') label = "Sayur (Hijau)";
                else if(cat === 'buah') label = "Buah (Ungu)";
                $('corrModeLabel').textContent = `Mode: ${label}`;
            });
        });

        $('corrBrushSize').addEventListener('input', (e) => {
            window.SegmentationEditor.setBrushSize(e.target.value);
        });

        $('btnCorrZoomIn').addEventListener('click', () => window.SegmentationEditor.setZoom(1.5));
        $('btnCorrZoomOut').addEventListener('click', () => window.SegmentationEditor.setZoom(1/1.5));
        $('btnCorrZoomReset').addEventListener('click', () => window.SegmentationEditor.resetZoom());
        $('btnCorrUndo').addEventListener('click', () => window.SegmentationEditor.undo());

        $('btnCorrRecalculate').addEventListener('click', () => {
            const btn = $('btnCorrRecalculate');
            btn.textContent = '⏳ Menghitung...';
            btn.disabled = true;
            
            setTimeout(() => {
                // ── MODE MURNI MANUAL ──────────────────────────────
                // Canvas koreksi (correctionCanvas) dibaca dari NOL.
                // Tidak ada fallback ke hasil AI (segCanvas).
                // Hanya pixel yang dilukis user yang dihitung.
                const cCtx = $('correctionCanvas').getContext('2d');
                const w = $('correctionCanvas').width;
                const h = $('correctionCanvas').height;
                const cData = cCtx.getImageData(0, 0, w, h).data;
                
                let counts = { makanan_pokok:0, lauk_pauk:0, sayur:0, buah:0 };
                let totalPainted = 0;
                
                const hexToRgb = hex => {
                    const r = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
                    return r ? [parseInt(r[1], 16), parseInt(r[2], 16), parseInt(r[3], 16)] : [0,0,0];
                };
                const colors = window.SegmentationEditor.getColors();
                const targets = {
                    'makanan_pokok': hexToRgb(colors['makanan_pokok']),
                    'lauk_pauk': hexToRgb(colors['lauk_pauk']),
                    'sayur': hexToRgb(colors['sayur']),
                    'buah': hexToRgb(colors['buah'])
                };
                
                const matchColor = (r, g, b, alpha, tol) => {
                    if (alpha < 50) return null;
                    for (const cat in targets) {
                        const tc = targets[cat];
                        if (Math.abs(r - tc[0]) < tol && Math.abs(g - tc[1]) < tol && Math.abs(b - tc[2]) < tol) {
                            return cat;
                        }
                    }
                    return null;
                };

                // Hanya baca correctionCanvas — murni hasil lukisan user
                for (let i = 0; i < cData.length; i += 4) {
                    const cat = matchColor(cData[i], cData[i+1], cData[i+2], cData[i+3], 40);
                    if (cat) {
                        counts[cat]++;
                        totalPainted++;
                    }
                }
                
                const isModified = totalPainted > 0;
                
                if (isModified) {
                    const foodPixels = counts.makanan_pokok + counts.lauk_pauk + counts.sayur + counts.buah;
                    const prop = {
                        makanan_pokok_percent: foodPixels ? (counts.makanan_pokok / foodPixels) * 100 : 0,
                        sayur_percent: foodPixels ? (counts.sayur / foodPixels) * 100 : 0,
                        lauk_pauk_percent: foodPixels ? (counts.lauk_pauk / foodPixels) * 100 : 0,
                        buah_percent: foodPixels ? (counts.buah / foodPixels) * 100 : 0
                    };
                    
                    updatePlateSlider('Carb', prop.makanan_pokok_percent);
                    updatePlateSlider('Veggie', prop.sayur_percent);
                    updatePlateSlider('Protein', prop.lauk_pauk_percent);
                    updatePlateSlider('Fruit', prop.buah_percent);
                    
                    $('sliderAutoLabel').innerHTML = '✏️ Manual Penuh';
                    $('sliderAutoLabel').style.background = '#F57C00';
                    $('sliderAutoLabel').style.color = '#fff';
                    $('sliderAutoLabel').style.display = 'inline-block';
                    
                    // Update quick result
                    $('segQuickResult').innerHTML = `
                        <div class="qr-item"><span>Karbo</span><strong>${prop.makanan_pokok_percent.toFixed(0)}%</strong></div>
                        <div class="qr-item"><span>Lauk</span><strong>${prop.lauk_pauk_percent.toFixed(0)}%</strong></div>
                        <div class="qr-item"><span>Sayur</span><strong>${prop.sayur_percent.toFixed(0)}%</strong></div>
                        <div class="qr-item"><span>Buah</span><strong>${prop.buah_percent.toFixed(0)}%</strong></div>
                    `;
                    
                    $('btnAnalyzePlate').click();
                } else {
                    // Tidak ada yang dilukis — peringatkan user
                    showToast('⚠️ Belum ada area yang dilukis! Gunakan kuas untuk menandai area makanan terlebih dahulu.');
                }
                
                btn.innerHTML = '✅ Hitung Ulang';
                btn.disabled = false;
            }, 500);
        });
        
        
        if ($('btnToggleAI')) {
            $('btnToggleAI').addEventListener('click', (e) => {
                const sCanvas = $('segCanvas');
                const ctx = sCanvas.getContext('2d');
                const img = new Image();
                
                if (e.target.dataset.hidden === 'true') {
                    // Tampilkan kembali AI
                    img.onload = () => {
                        ctx.clearRect(0, 0, sCanvas.width, sCanvas.height);
                        ctx.drawImage(img, 0, 0, sCanvas.width, sCanvas.height);
                    };
                    img.src = window._annotatedPhotoDataURL;
                    e.target.innerHTML = '👁 Sembunyikan AI';
                    e.target.dataset.hidden = 'false';
                } else {
                    // Sembunyikan AI (tampilkan foto bersih)
                    img.onload = () => {
                        ctx.clearRect(0, 0, sCanvas.width, sCanvas.height);
                        ctx.drawImage(img, 0, 0, sCanvas.width, sCanvas.height);
                    };
                    img.src = window._cleanPhotoDataURL;
                    e.target.innerHTML = '👁 Tampilkan AI';
                    e.target.dataset.hidden = 'true';
                }
            });
        }
    }

        const handleSegPhotoUpload = async (e) => {
            const file = e.target.files[0];
            if(!file) return;
            e.target.value = ''; // Reset
            
            if(!window.SeeFoodSegmenter) {
                alert('Model AI belum selesai dimuat, harap tunggu.');
                return;
            }

            const badge = $('segModelStatus');
            const isYolo = badge && badge.textContent.includes('YOLO');
            const subEl = $('segLoadingSub');
            if (subEl) {
                subEl.textContent = isYolo ? 'Menggunakan YOLOv11-Seg (GPU)' : 'Menggunakan DeepLab-V3 + MobileNet-V2';
            }

            $('segPhotoInput').style.display = 'none';
            $('segPreviewContainer').style.display = 'none';
            $('piringkuResult').style.display = 'none';
            $('segLoading').style.display = 'block';
            $('segDebugPanel').style.display = 'none';
            
            const reader = new FileReader();
            reader.onload = async (event) => {
                window._cleanPhotoDataURL = event.target.result;
                const img = new Image();
                img.onload = async () => {
                    try {
                        const canvas = $('segCanvas');
                        const res = await window.SeeFoodSegmenter.analyze(img, canvas);
                        window._lastSegResult = res; // Save for correction
                        window._annotatedPhotoDataURL = canvas.toDataURL(); // Simpan hasil YOLO/AI
                        
                        // Reset button state
                        const btnToggle = $('btnToggleAI');
                        if (btnToggle) {
                            btnToggle.dataset.hidden = 'false';
                            btnToggle.innerHTML = '👁 Sembunyikan AI';
                        }
                        
                        if (window.SegmentationEditor) {
                            window.SegmentationEditor.init(img, canvas, $('correctionCanvas'));
                            window.SegmentationEditor.clear();
                            $('correctionToolbar').classList.remove('active');
                            $('btnToggleCorrection').classList.remove('active');
                            $('correctionCanvas').classList.remove('active');
                            $('corrHint').style.display = 'none';
                        }
                        
                        // Auto-fill sliders
                        const prop = res.proportions;
                        updatePlateSlider('Carb', prop.makanan_pokok_percent);
                        updatePlateSlider('Veggie', prop.sayur_percent);
                        updatePlateSlider('Protein', prop.lauk_pauk_percent);
                        updatePlateSlider('Fruit', prop.buah_percent);
                        
                        $('sliderAutoLabel').style.display = 'inline-block';
                        
                        // Quick result text
                        $('segQuickResult').innerHTML = `
                            <div class="qr-item"><span>Karbo</span><strong>${prop.makanan_pokok_percent.toFixed(0)}%</strong></div>
                            <div class="qr-item"><span>Lauk</span><strong>${prop.lauk_pauk_percent.toFixed(0)}%</strong></div>
                            <div class="qr-item"><span>Sayur</span><strong>${prop.sayur_percent.toFixed(0)}%</strong></div>
                            <div class="qr-item"><span>Buah</span><strong>${prop.buah_percent.toFixed(0)}%</strong></div>
                        `;
                        
                        // YOLO details rendering
                        if (res.mode === 'yolo' && res.detections && res.detections.length > 0) {
                            const detectionsListEl = $('yoloDetectionsList');
                            detectionsListEl.innerHTML = '';
                            
                            const sortedDetections = [...res.detections].sort((a, b) => b.confidence - a.confidence);
                            sortedDetections.forEach(det => {
                                let badgeColor = '#9e9e9e';
                                if (det.category === 'makanan_pokok') badgeColor = '#FFC107';
                                else if (det.category === 'lauk_pauk') badgeColor = '#F44336';
                                else if (det.category === 'sayur') badgeColor = '#4CAF50';
                                else if (det.category === 'buah') badgeColor = '#9C27B0';
                                
                                const itemHtml = `
                                    <div class="yolo-det-item">
                                        <span class="yolo-det-label">
                                            <span class="yolo-det-dot" style="background:${badgeColor};"></span>
                                            ${det.class_name}
                                        </span>
                                        <span class="yolo-det-conf">
                                            ${(det.confidence * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                `;
                                detectionsListEl.insertAdjacentHTML('beforeend', itemHtml);
                            });
                            
                            $('yoloDetectionsContainer').style.display = 'block';
                            $('sliderAutoLabel').textContent = '🤖 YOLOv8-Seg';
                            $('sliderAutoLabel').style.background = '#48C78E';
                            $('sliderAutoLabel').style.color = '#fff';
                        } else {
                            $('yoloDetectionsContainer').style.display = 'none';
                            $('sliderAutoLabel').textContent = '🤖 Auto-detected';
                            $('sliderAutoLabel').style.background = '';
                            $('sliderAutoLabel').style.color = '';
                        }

                        // Debug Panel
                        $('segDebugOutput').textContent = JSON.stringify(res.pixelCounts, null, 2) + 
                            `\nTotal Pixels: ${res.totalPixels}\nFood Pixels: ${res.foodPixels}\nTime: ${res.inferenceTime}s`;
                        $('segDebugPanel').style.display = 'block';
                        
                        $('btnAnalyzePlate').click(); // trigger analysis
                        
                    } catch(err) {
                        console.error(err);
                        alert('Gagal menganalisis foto. Error: ' + err.message + '\n\nSilakan atur proporsi secara manual.');
                        $('segPhotoInput').style.display = 'block';
                    } finally {
                        $('segLoading').style.display = 'none';
                        $('segPreviewContainer').style.display = 'block';
                    }
                };
                img.src = event.target.result;
            };
            reader.readAsDataURL(file);
        };

        $('plateCameraInput').addEventListener('change', handleSegPhotoUpload);
        $('plateImgUpload').addEventListener('change', handleSegPhotoUpload);
    }

    // --- Isi Piringku Sliders ---
    ['Carb','Veggie','Protein','Fruit'].forEach(key => {
        const slider = $('slider' + key);
        if (slider) {
            slider.addEventListener('input', () => {
                const val = slider.value;
                $('val' + key).textContent = val + '%';
                // Update plate visual
                const seg = $('plate' + key);
                if (seg) seg.style.flex = val;
            });
        }
    });

    // --- Isi Piringku Analysis ---
    $('btnAnalyzePlate').addEventListener('click', async () => {
        const seg = {
            makanan_pokok_percent: parseInt($('sliderCarb').value) || 0,
            sayur_percent: parseInt($('sliderVeggie').value) || 0,
            lauk_pauk_percent: parseInt($('sliderProtein').value) || 0,
            buah_percent: parseInt($('sliderFruit').value) || 0
        };

        const profile = state.user || {};
        const result = window.IsiPiringku.validateIsiPiringku(seg, profile);
        const resEl = $('piringkuResult');

        let html = `<div style="margin-bottom:12px;">
            <h3 style="font-size:1.1rem; font-weight:800;">${result.is_balanced ? '✅ Seimbang!' : '⚠️ Belum Seimbang'}</h3>
        </div>`;

        // Ideal vs Actual bars
        html += '<div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:16px;">';
        const labels = { makanan_pokok:'🍚 Karbo', sayur:'🥬 Sayur', lauk_pauk:'🍗 Lauk', buah:'🍎 Buah' };
        for (const [k, lbl] of Object.entries(labels)) {
            const d = result.ideal_vs_actual[k];
            const isOk = Math.abs(d.actual - d.ideal) < 10;
            html += `<div style="background:${isOk?'rgba(72,199,142,0.08)':'rgba(239,119,104,0.08)'}; padding:10px; border-radius:10px;">
                <div style="font-size:0.75rem; font-weight:600; color:var(--text-muted);">${lbl}</div>
                <div style="font-size:1rem; font-weight:800;">${d.actual.toFixed(0)}% <span style="font-size:0.75rem; font-weight:500; color:var(--text-muted);">/ ${d.ideal.toFixed(0)}%</span></div>
            </div>`;
        }
        html += '</div>';

        // Messages
        html += '<div style="font-size:0.85rem; line-height:1.6;">';
        result.messages.forEach(m => html += `<p style="margin-bottom:6px;">${m}</p>`);
        html += '</div>';

        // Clinical notes
        if (result.clinical_notes.length > 0) {
            html += '<div style="margin-top:12px; padding:12px; border-radius:12px; background:rgba(142,68,173,0.08); border:1px solid rgba(142,68,173,0.15);">';
            html += '<h4 style="font-size:0.85rem; font-weight:700; margin-bottom:6px;">🩺 Catatan Klinis</h4>';
            result.clinical_notes.forEach(n => html += `<p style="font-size:0.8rem; margin-bottom:4px;">${n}</p>`);
            html += '</div>';
        }

        resEl.innerHTML = html;
        resEl.style.display = 'block';

        // --- Fetch Nutrition Prediction ---
        fetchNutritionPrediction(seg, result);
    });

    // ═══════════════════════════════════════════════════════
    //  NUTRITION PREDICTION ENGINE
    // ═══════════════════════════════════════════════════════

    let lastNutritionResult = null;

    async function fetchNutritionPrediction(proportions, piringkuResult) {
        const nutritionEl = $('nutritionPredictionResult');
        if (!nutritionEl) return;

        // Build labels from YOLO detections if available, else use category names
        const detections = window._lastYoloDetections || [];
        const labelMap = {
            makanan_pokok: 'nasi putih',
            lauk_pauk: 'ayam goreng',
            sayur: 'sayur campuran',
            buah: 'buah segar'
        };

        // Map YOLO detections to labels
        if (detections.length > 0) {
            detections.forEach(d => {
                if (d.category && d.class_name) {
                    labelMap[d.category] = d.class_name;
                }
            });
        }

        const detected_labels = [];
        for (const [cat, pctKey] of [['makanan_pokok', 'makanan_pokok_percent'], ['lauk_pauk', 'lauk_pauk_percent'], ['sayur', 'sayur_percent'], ['buah', 'buah_percent']]) {
            const pct = proportions[pctKey] || 0;
            if (pct > 0) {
                detected_labels.push({
                    label: labelMap[cat] || cat,
                    category: cat,
                    proportion: pct / 100
                });
            }
        }

        if (detected_labels.length === 0) {
            nutritionEl.style.display = 'none';
            return;
        }

        try {
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 8000);

            const response = await fetch('http://127.0.0.1:8000/api/food-nutrition', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ detected_labels }),
                signal: controller.signal
            });
            clearTimeout(timeout);

            if (!response.ok) throw new Error(`Server error: ${response.status}`);
            const data = await response.json();

            if (data.status === 'success') {
                lastNutritionResult = data;
                renderNutritionResult(data);
            } else {
                throw new Error(data.message || 'Unknown error');
            }
        } catch (err) {
            console.warn('[NutritionPredictor] Backend not available, using offline estimation:', err.message);
            // Offline fallback: rough calorie estimation from proportions
            const offlineResult = estimateNutritionOffline(detected_labels);
            lastNutritionResult = offlineResult;
            renderNutritionResult(offlineResult);
        }
    }

    // Offline fallback nutrition estimation (when backend is unavailable)
    function estimateNutritionOffline(detected_labels) {
        const OFFLINE_DB = {
            makanan_pokok: { cal_per_100g: 130, protein: 2.7, carbs: 28, fat: 0.3, name: 'Nasi/Karbo' },
            lauk_pauk: { cal_per_100g: 220, protein: 20, carbs: 3, fat: 14, name: 'Lauk Protein' },
            sayur: { cal_per_100g: 35, protein: 2, carbs: 6, fat: 0.5, name: 'Sayuran' },
            buah: { cal_per_100g: 50, protein: 0.6, carbs: 12, fat: 0.2, name: 'Buah-buahan' }
        };
        const SERVING_G = { makanan_pokok: 150, lauk_pauk: 75, sayur: 100, buah: 100 };

        let totalCal = 0, totalProtein = 0, totalCarbs = 0, totalFat = 0;
        const components = [];

        for (const item of detected_labels) {
            const cat = item.category;
            const db = OFFLINE_DB[cat];
            if (!db) continue;
            const serving = SERVING_G[cat] || 100;
            const scaledG = serving * item.proportion * 2;
            const cal = (db.cal_per_100g / 100) * scaledG;
            const pro = (db.protein / 100) * scaledG;
            const carb = (db.carbs / 100) * scaledG;
            const fat = (db.fat / 100) * scaledG;

            totalCal += cal;
            totalProtein += pro;
            totalCarbs += carb;
            totalFat += fat;

            components.push({
                label: item.label,
                category: cat,
                calories: Math.round(cal),
                protein_g: parseFloat(pro.toFixed(1)),
                carbohydrates_g: parseFloat(carb.toFixed(1)),
                fat_g: parseFloat(fat.toFixed(1)),
                confidence: 0.5,
                confidence_level: 'low',
                matched_food: db.name,
                estimated_grams: Math.round(scaledG)
            });
        }

        return {
            status: 'success',
            source: 'offline',
            total: {
                calories: Math.round(totalCal),
                protein_g: parseFloat(totalProtein.toFixed(1)),
                carbohydrates_g: parseFloat(totalCarbs.toFixed(1)),
                fat_g: parseFloat(totalFat.toFixed(1))
            },
            components,
            confidence_level: 'low'
        };
    }

    function renderNutritionResult(data) {
        const el = $('nutritionPredictionResult');
        if (!el || !data) return;

        const t = data.total || {};
        $('ntTotalCal').textContent = t.calories || 0;
        $('ntTotalProtein').textContent = t.protein_g || 0;
        $('ntTotalCarb').textContent = t.carbohydrates_g || 0;
        $('ntTotalFat').textContent = t.fat_g || 0;

        // Confidence badge
        const confBadge = $('nutritionConfBadge');
        const confLevel = data.confidence_level || 'low';
        const confLabels = { high: 'Estimasi Akurat', medium: 'Estimasi Sedang', low: 'Estimasi Kasar' };
        const confColors = { high: '#2E7D32', medium: '#F57F17', low: '#C62828' };
        const confBgs = { high: 'rgba(72,199,142,0.15)', medium: 'rgba(244,196,90,0.2)', low: 'rgba(239,119,104,0.15)' };
        confBadge.textContent = confLabels[confLevel] || 'Estimasi';
        confBadge.style.background = confBgs[confLevel] || confBgs.low;
        confBadge.style.color = confColors[confLevel] || confColors.low;

        // Per-component breakdown
        const compList = $('nutritionComponentList');
        const catEmojis = { makanan_pokok: '🍚', lauk_pauk: '🍗', sayur: '🥬', buah: '🍎' };
        const confBadgeClass = { high: 'nc-conf-high', medium: 'nc-conf-medium', low: 'nc-conf-low' };
        let compHtml = '';
        (data.components || []).forEach(c => {
            const emoji = catEmojis[c.category] || '🍽️';
            const bClass = confBadgeClass[c.confidence_level] || 'nc-conf-low';
            compHtml += `<div class="nutrition-comp-row">
                <div class="nc-label">${emoji} ${c.matched_food || c.label}
                    <span class="nc-conf-badge ${bClass}">${c.estimated_grams || '~'}g</span>
                </div>
                <div class="nc-values">${c.calories} kkal | P:${c.protein_g}g C:${c.carbohydrates_g}g F:${c.fat_g}g</div>
            </div>`;
        });
        compList.innerHTML = compHtml;

        // Pre-fill editable fields
        $('ntEditCal').value = t.calories || '';
        $('ntEditProtein').value = t.protein_g || '';
        $('ntEditCarb').value = t.carbohydrates_g || '';
        $('ntEditFat').value = t.fat_g || '';

        // Health guidance
        renderNutritionGuidance(t);

        el.style.display = 'block';
    }

    function renderNutritionGuidance(totals) {
        const guidanceEl = $('nutritionGuidance');
        const contentEl = $('nutritionGuidanceContent');
        if (!guidanceEl || !contentEl || !state.user) return;

        const tdee = state.user.tdee || 2000;
        const cal = totals.calories || 0;
        const calPct = Math.round((cal / tdee) * 100);
        const messages = [];

        if (calPct > 50) {
            messages.push(`⚠️ Makanan ini sekitar <strong>${calPct}%</strong> dari total kebutuhan kalori harianmu (${Math.round(tdee)} kkal). Pertimbangkan porsi yang lebih kecil.`);
        } else if (calPct > 30) {
            messages.push(`💡 Makanan ini mencakup <strong>${calPct}%</strong> dari kebutuhan harian (${Math.round(tdee)} kkal). Porsi yang wajar untuk satu kali makan utama.`);
        } else {
            messages.push(`✅ Makanan ini hanya <strong>${calPct}%</strong> dari kebutuhan harianmu. Porsi ringan dan seimbang!`);
        }

        // Diabetes-specific
        if (state.user.isDiabetic && totals.carbohydrates_g > 60) {
            messages.push(`🩺 Perhatian diabetes: Karbohidrat ${totals.carbohydrates_g}g cukup tinggi. Monitor gula darah setelah makan.`);
        }

        // Hypertension-specific
        if (state.user.isHypertensive) {
            messages.push(`🧂 Pastikan makanan ini rendah garam. Perhatikan saus dan bumbu tambahan.`);
        }

        if (messages.length > 0) {
            contentEl.innerHTML = messages.join('<br><br>');
            guidanceEl.style.display = 'block';
        } else {
            guidanceEl.style.display = 'none';
        }
    }

    // Save nutrition to history
    if ($('btnSaveNutrition')) {
        $('btnSaveNutrition').addEventListener('click', () => {
            if (!lastNutritionResult) { alert('Belum ada data nutrisi!'); return; }

            // Use edited values if user corrected
            const cal = parseFloat($('ntEditCal').value) || lastNutritionResult.total.calories || 0;
            const protein = parseFloat($('ntEditProtein').value) || lastNutritionResult.total.protein_g || 0;
            const carbs = parseFloat($('ntEditCarb').value) || lastNutritionResult.total.carbohydrates_g || 0;
            const fat = parseFloat($('ntEditFat').value) || lastNutritionResult.total.fat_g || 0;

            const entry = normalizeEntry({
                name: '🍽️ Menu Sehat',
                grade: cal > 500 ? 'C' : (cal > 250 ? 'B' : 'A'),
                sugar: 0,
                cal: Math.round(cal),
                fat: parseFloat(fat.toFixed(1)),
                sodium: 0,
                protein: parseFloat(protein.toFixed(1)),
                fiber: 0,
                carbohydrates_g: parseFloat(carbs.toFixed(1)),
                date: new Date().toISOString(),
                type: 'meal',
                nutrition: { calories: cal, protein_g: protein, carbohydrates_g: carbs, fat_g: fat }
            });

            state.history.push(entry);
            saveState();
            haptic('success');
            launchConfetti();

            // Visual feedback
            $('btnSaveNutrition').innerHTML = '✅ Tersimpan!';
            $('btnSaveNutrition').style.background = 'var(--grade-A)';
            setTimeout(() => {
                $('btnSaveNutrition').innerHTML = '🍽️ Simpan Makanan ke Riwayat';
                $('btnSaveNutrition').style.background = '';
            }, 2000);

            // Progress toast
            const todayStr = getLocalDateStr();
            const todayCal = state.history.filter(h => getLocalDateStr(h.date) === todayStr).reduce((a,h) => a + (h.cal||0), 0);
            const maxCal = state.limitCalorie || 2000;
            const pct = Math.min(100, Math.round((todayCal/maxCal)*100));
            showToast(`🍽️ Menu Sehat tercatat! Total hari ini: ${todayCal} kkal (${pct}%)`);
        });
    }

    // Store YOLO detections for nutrition prediction
    window._lastYoloDetections = [];

    // ═══════════════════════════════════════════════════════
    //  HYDRATION SYSTEM INTEGRATION
    // ═══════════════════════════════════════════════════════

    let hydrationChartInstance = null;

    function getHydrationTarget() {
        return (state.user && state.user.hydration_target_ml) || 2000;
    }

    function updateHydrationView() {
        if (!window.HydrationEngine) return;
        const HE = window.HydrationEngine;
        const target = getHydrationTarget();
        const consumed = HE.getTodayTotal(target);
        const status = HE.getHydrationStatus(consumed, target);

        // Target text
        if ($('hydrationTargetText') && state.user) {
            const liters = (target / 1000).toFixed(1);
            const glasses = Math.ceil(target / HE.GLASS_ML);
            $('hydrationTargetText').textContent = `Target: ${liters}L (~${glasses} gelas)`;
        }

        // Consumed mL
        if ($('hydrationConsumedML')) $('hydrationConsumedML').textContent = consumed;

        // Water blob fill level
        updateWaterBlobFill(status.percent);

        // Emoji
        if ($('hydrationEmoji')) $('hydrationEmoji').textContent = status.emoji;

        // Mood
        if ($('hydrationMoodEmoji')) $('hydrationMoodEmoji').textContent = status.emoji;
        if ($('hydrationMoodText')) $('hydrationMoodText').textContent = status.text;
        if ($('hydrationMood')) {
            $('hydrationMood').style.background = status.bg;
            $('hydrationMood').style.color = status.color;
        }

        // Glow effect at high percentage
        const svgEl = $('waterBlobSvg');
        if (svgEl) {
            svgEl.classList.toggle('glowing', status.percent >= 80);
        }

        // Render today's log
        renderHydrationLog();

        // Update 7-day chart
        renderHydrationChart();
    }

    function updateWaterBlobFill(percent) {
        const fillPath = $('waterFillPath');
        if (!fillPath) return;

        const clampedPct = Math.max(0, Math.min(100, percent));
        // Water level: 0% = bottom (y=200), 100% = top (y=20)
        const waterY = 200 - (clampedPct / 100) * 180;
        const waveH = Math.max(2, 8 - (clampedPct / 20));

        const wavePath1 = `M 0 ${waterY} Q 50 ${waterY - waveH} 100 ${waterY} Q 150 ${waterY + waveH} 200 ${waterY} L 200 220 L 0 220 Z`;
        const wavePath2 = `M 0 ${waterY} Q 50 ${waterY + waveH} 100 ${waterY} Q 150 ${waterY - waveH} 200 ${waterY} L 200 220 L 0 220 Z`;

        // Update the animate element's values
        const animateEl = fillPath.querySelector('animate');
        if (animateEl) {
            animateEl.setAttribute('values', `${wavePath1};${wavePath2};${wavePath1}`);
        }
        fillPath.setAttribute('d', wavePath1);
    }

    function triggerBounceAnimation() {
        const container = $('waterBlobContainer');
        if (!container) return;
        container.classList.remove('bounce');
        void container.offsetWidth; // Force reflow
        container.classList.add('bounce');
        setTimeout(() => container.classList.remove('bounce'), 700);
    }

    function handleHydrationLog(ml) {
        if (!window.HydrationEngine || !ml || ml <= 0) return;
        const target = getHydrationTarget();
        window.HydrationEngine.logIntake(ml, target);
        triggerBounceAnimation();
        updateHydrationView();
        updateHydrationMiniWidget();
        updateDashboard();
        
        // Cute appreciation message
        const cheers = ["Segar! +{ml}mL 🌊", "Gluk gluk! Air masuk! 💧", "Great job! +{ml}mL ✨", "Tubuhmu berterima kasih! 🥰"];
        const randomCheer = cheers[Math.floor(Math.random() * cheers.length)].replace('{ml}', ml);
        if (typeof showToast === 'function') {
            showToast(randomCheer);
        }
    }

    function renderHydrationLog() {
        if (!window.HydrationEngine) return;
        const logList = $('hydrationLogList');
        if (!logList) return;

        const target = getHydrationTarget();
        const { log } = (function() {
            const logs = window.HydrationEngine.getAllLogs();
            const todayStr = new Date().toISOString().split('T')[0];
            const todayLog = logs.find(l => l.date === todayStr);
            return { log: todayLog || { entries: [] } };
        })();

        if (!log.entries || log.entries.length === 0) {
            logList.innerHTML = '<p style="font-size:0.8rem; color:var(--text-muted); text-align:center; padding:16px;">Belum ada log hari ini. Yuk minum air! 💧</p>';
            return;
        }

        logList.innerHTML = log.entries.slice().reverse().map(e => `
            <div class="hydration-log-item">
                <span style="font-weight:600;">🕐 ${e.time}</span>
                <span style="font-weight:800; color:var(--text-dark);">+${e.ml} mL</span>
            </div>
        `).join('');
    }

    function renderHydrationChart() {
        if (!window.HydrationEngine || typeof Chart === 'undefined') return;
        const canvas = $('hydrationWeeklyChart');
        if (!canvas) return;

        const data = window.HydrationEngine.get7DayHistory();

        if (hydrationChartInstance) {
            hydrationChartInstance.data.labels = data.labels;
            hydrationChartInstance.data.datasets[0].data = data.consumed;
            hydrationChartInstance.data.datasets[1].data = data.targets;
            hydrationChartInstance.update('none');
            return;
        }

        hydrationChartInstance = new Chart(canvas, {
            type: 'bar',
            data: {
                labels: data.labels,
                datasets: [
                    {
                        label: 'Dikonsumsi',
                        data: data.consumed,
                        backgroundColor: 'rgba(116, 185, 255, 0.7)',
                        borderColor: '#0984e3',
                        borderWidth: 1,
                        borderRadius: 6,
                        borderSkipped: false
                    },
                    {
                        label: 'Target',
                        data: data.targets,
                        backgroundColor: 'rgba(0,0,0,0.04)',
                        borderColor: 'rgba(0,0,0,0.08)',
                        borderWidth: 1,
                        borderRadius: 6,
                        borderSkipped: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'bottom',
                        labels: { font: { family: 'Outfit', size: 11, weight: '600' }, boxWidth: 12, boxHeight: 12, borderRadius: 3, useBorderRadius: true }
                    }
                },
                scales: {
                    x: { grid: { display: false }, ticks: { font: { family: 'Outfit', size: 10 } } },
                    y: { beginAtZero: true, grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { font: { family: 'Outfit', size: 10 }, callback: v => (v / 1000).toFixed(1) + 'L' } }
                }
            }
        });
    }

    function updateHydrationMiniWidget() {
        if (!window.HydrationEngine || !state.user) return;
        const HE = window.HydrationEngine;
        const target = getHydrationTarget();
        const consumed = HE.getTodayTotal(target);
        const status = HE.getHydrationStatus(consumed, target);

        // Mini circle progress
        const circle = $('hydrationMiniCircle');
        if (circle) {
            const pct = Math.min(100, status.percent);
            const dashOffset = 125.6 - (125.6 * pct / 100);
            circle.style.strokeDashoffset = dashOffset;
            circle.style.stroke = status.color;
        }

        // Mini emoji
        if ($('hydrationMiniEmoji')) $('hydrationMiniEmoji').textContent = status.emoji;

        // Mini text
        if ($('hydrationMiniText')) {
            $('hydrationMiniText').textContent = `${consumed} / ${target} mL`;
        }

        // Mini mood
        if ($('hydrationMiniMood')) {
            $('hydrationMiniMood').textContent = status.text;
            $('hydrationMiniMood').style.color = status.color;
        }
    }

    // ── Hydration Event Listeners ───────────────────────
    function initHydrationEvents() {
        // Quick-add buttons
        $$('.hq-btn[data-ml]').forEach(btn => {
            btn.addEventListener('click', () => {
                const ml = parseInt(btn.dataset.ml);
                if (ml > 0) handleHydrationLog(ml);
            });
        });

        // Custom ML button
        if ($('btnHydrationCustom')) {
            $('btnHydrationCustom').addEventListener('click', () => {
                const ml = parseInt($('hydrationCustomML').value);
                if (ml > 0 && ml <= 2000) {
                    handleHydrationLog(ml);
                    $('hydrationCustomML').value = '';
                } else if (ml > 2000) {
                    alert('Maksimal 2000 mL per entri!');
                }
            });
        }

        // Undo button
        if ($('btnHydrationUndo')) {
            $('btnHydrationUndo').addEventListener('click', () => {
                if (!window.HydrationEngine) return;
                const target = getHydrationTarget();
                const result = window.HydrationEngine.undoLastIntake(target);
                if (result.removed) {
                    updateHydrationView();
                    updateHydrationMiniWidget();
                    updateDashboard();
                }
            });
        }

        // Urine Color Scale
        $$('.urine-color-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                if (!window.HydrationEngine) return;
                const idx = parseInt(btn.dataset.idx);
                const advice = window.HydrationEngine.getUrineAdvice(idx);

                // Active state
                $$('.urine-color-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');

                // Show result
                const resultEl = $('urineResult');
                const iconEl = $('urineResultIcon');
                const textEl = $('urineResultText');

                if (resultEl && iconEl && textEl) {
                    const severityIcons = { good: '✅', warning: '⚠️', danger: '🚨', info: 'ℹ️' };
                    iconEl.textContent = severityIcons[advice.severity] || '💧';
                    textEl.innerHTML = `<strong>${advice.label}</strong><br>${advice.text}`;
                    resultEl.className = `urine-result severity-${advice.severity}`;
                    resultEl.style.display = 'flex';
                }
            });
        });
    }

    // ── Hydration Reminder Timer ───────────────────────
    function initHydrationReminder() {
        setInterval(() => {
            if (!state.user || !window.HydrationEngine) return;
            
            const hour = new Date().getHours();
            if (hour < 8 || hour > 21) return; // Hanya aktif jam 8 pagi - 9 malam

            const logs = window.HydrationEngine.getAllLogs();
            const todayStr = new Date().toISOString().split('T')[0];
            const todayLog = logs.find(l => l.date === todayStr);

            let lastDrinkTime = new Date();
            lastDrinkTime.setHours(8, 0, 0, 0);

            if (todayLog && todayLog.entries && todayLog.entries.length > 0) {
                const lastEntry = todayLog.entries[todayLog.entries.length - 1];
                const [h, m] = lastEntry.time.split(':');
                lastDrinkTime = new Date();
                lastDrinkTime.setHours(parseInt(h), parseInt(m), 0, 0);
            }

            const now = new Date();
            const diffHours = (now - lastDrinkTime) / (1000 * 60 * 60);

            if (diffHours >= 2) {
                const lastReminder = localStorage.getItem('last_hydration_reminder');
                if (!lastReminder || (now.getTime() - parseInt(lastReminder) > 60 * 60 * 1000)) {
                    showToast("Waktunya minum air! 💧 Tubuhmu butuh hidrasi, ayo minum 1 gelas sekarang.");
                    localStorage.setItem('last_hydration_reminder', now.getTime().toString());
                }
            }
        }, 60 * 1000); // Cek setiap menit
    }

    function showToast(msg) {
        let toast = $('appToast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'appToast';
            toast.style.cssText = 'position:fixed; top:20px; left:50%; transform:translateX(-50%); background:rgba(255,255,255,0.95); backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px); padding:16px 20px; border-radius:16px; box-shadow:0 10px 30px rgba(0,0,0,0.12); z-index:9999; font-weight:700; color:var(--text-dark); text-align:center; width:90%; max-width:400px; border:1px solid rgba(72,199,142,0.3); font-size:0.9rem; line-height:1.4;';
            document.body.appendChild(toast);
        }
        toast.classList.remove('hiding');
        toast.innerHTML = msg;
        toast.style.display = 'block';
        clearTimeout(window._toastTimer);
        window._toastTimer = setTimeout(() => {
            toast.classList.add('hiding');
            setTimeout(() => { toast.style.display = 'none'; toast.classList.remove('hiding'); }, 300);
        }, 4000);
    }

    // ── Library / Perpustakaan Gizi Logic ──────────────────────
    let librarySearchTimeout = null;
    let selectedLibraryFood = null;

    function renderLibrarySuggestions() {
        const container = $('librarySearchResults');
        if (!container) return;
        
        const suggestions = [
            { name: "Nasi Putih", calories: 130, protein_g: 2.7, fat_g: 0.3, carbohydrates_g: 28, source: "nutrition_csv" },
            { name: "Dada Ayam Panggang", calories: 165, protein_g: 31, fat_g: 3.6, carbohydrates_g: 0, source: "nutrition_csv" },
            { name: "Telur Rebus", calories: 155, protein_g: 13, fat_g: 11, carbohydrates_g: 1.1, source: "nutrition_csv" },
            { name: "Pisang Mas", calories: 89, protein_g: 1.1, fat_g: 0.3, carbohydrates_g: 22.8, source: "nutrition_csv" },
            { name: "Apel", calories: 52, protein_g: 0.3, fat_g: 0.2, carbohydrates_g: 13.8, source: "nutrition_csv" },
            { name: "Oatmeal Matang", calories: 68, protein_g: 2.4, fat_g: 1.4, carbohydrates_g: 12, source: "nutrition_csv" },
            { name: "Susu Sapi UHT Low Fat", calories: 42, protein_g: 3.4, fat_g: 1.3, carbohydrates_g: 4.7, source: "nutrition_csv" }
        ];

        container.innerHTML = `
            <div class="library-suggestions-header" style="margin-top: 16px; margin-bottom: 12px; font-weight: 700; color: var(--text-dark); display: flex; align-items: center; gap: 8px; font-size: 0.95rem;">
                <span>💡 Rekomendasi Gizi Harian (100g)</span>
            </div>
            <div style="display: flex; flex-direction: column; gap: 10px;"></div>
        `;
        const listDiv = container.querySelector('div:last-child');
        
        suggestions.forEach(food => {
            const card = document.createElement('div');
            card.className = 'food-result-card';
            card.style.cursor = 'pointer';
            
            card.innerHTML = `
                <div class="frc-info">
                    <h4>${food.name}</h4>
                    <span class="frc-source indo">🇮🇩 Rekomendasi</span>
                    <div class="frc-macros">
                        <span>P: ${food.protein_g}g</span>
                        <span>L: ${food.fat_g}g</span>
                        <span>K: ${food.carbohydrates_g}g</span>
                    </div>
                </div>
                <div class="frc-cal">${Math.round(food.calories)} <span style="font-size:0.6rem; color:var(--text-muted);">kkal/100g</span></div>
            `;
            card.addEventListener('click', () => openManualLogModal(food));
            listDiv.appendChild(card);
        });
    }

    if ($('foodSearchInput')) {
        $('foodSearchInput').addEventListener('input', (e) => {
            clearTimeout(librarySearchTimeout);
            const query = e.target.value.trim();
            const resultsContainer = $('librarySearchResults');
            const loader = $('librarySearchLoader');

            if (query.length < 2) {
                if (loader) loader.style.display = 'none';
                renderLibrarySuggestions();
                return;
            }

            if (loader) loader.style.display = 'block';
            resultsContainer.innerHTML = '';

            librarySearchTimeout = setTimeout(async () => {
                try {
                    const res = await fetch(`http://127.0.0.1:8000/api/search-food?q=${encodeURIComponent(query)}`);
                    if (res.ok) {
                        const data = await res.json();
                        renderLibraryResults(data.results);
                    } else {
                        resultsContainer.innerHTML = '<p style="text-align:center; color:var(--text-muted);">Gagal terhubung ke database lokal. Coba jalankan backend FastAPI.</p>';
                    }
                } catch (err) {
                    console.error("Search error:", err);
                    // Offline fallback: cari dari suggestions lokal
                    renderLibraryOfflineSearch(query);
                } finally {
                    if (loader) loader.style.display = 'none';
                }
            }, 400); // 400ms debounce
        });
    }

    function renderLibraryResults(results) {
        const container = $('librarySearchResults');
        container.innerHTML = '';
        if (!results || results.length === 0) {
            container.innerHTML = '<p style="text-align:center; color:var(--text-muted);">Tidak ditemukan makanan dengan nama tersebut.</p>';
            return;
        }

        results.forEach(food => {
            const card = document.createElement('div');
            card.className = 'food-result-card';
            
            const isIndo = food.source === 'nutrition_csv';
            const badgeClass = isIndo ? 'frc-source indo' : 'frc-source';
            const badgeText = isIndo ? '🇮🇩 Lokal' : '🌍 Internasional';

            card.innerHTML = `
                <div class="frc-info">
                    <h4>${food.name}</h4>
                    <span class="${badgeClass}">${badgeText}</span>
                    <div class="frc-macros">
                        <span>P: ${food.protein_g}g</span>
                        <span>L: ${food.fat_g}g</span>
                        <span>K: ${food.carbohydrates_g}g</span>
                    </div>
                </div>
                <div class="frc-cal">${Math.round(food.calories)} <span style="font-size:0.6rem; color:var(--text-muted);">kkal/100g</span></div>
            `;
            
            card.addEventListener('click', () => openManualLogModal(food));
            container.appendChild(card);
        });
    }

    // Pencarian offline dari suggestions lokal saat backend tidak tersedia
    function renderLibraryOfflineSearch(query) {
        const OFFLINE_FOODS = [
            { name: "Nasi Putih", calories: 130, protein_g: 2.7, fat_g: 0.3, carbohydrates_g: 28, source: "nutrition_csv" },
            { name: "Nasi Merah", calories: 111, protein_g: 2.6, fat_g: 0.9, carbohydrates_g: 23, source: "nutrition_csv" },
            { name: "Nasi Goreng", calories: 163, protein_g: 4.4, fat_g: 5.6, carbohydrates_g: 24, source: "nutrition_csv" },
            { name: "Dada Ayam Panggang", calories: 165, protein_g: 31, fat_g: 3.6, carbohydrates_g: 0, source: "nutrition_csv" },
            { name: "Ayam Goreng", calories: 260, protein_g: 25, fat_g: 16, carbohydrates_g: 4, source: "nutrition_csv" },
            { name: "Telur Rebus", calories: 155, protein_g: 13, fat_g: 11, carbohydrates_g: 1.1, source: "nutrition_csv" },
            { name: "Telur Dadar", calories: 185, protein_g: 12, fat_g: 14, carbohydrates_g: 1, source: "nutrition_csv" },
            { name: "Tempe Goreng", calories: 200, protein_g: 14, fat_g: 11, carbohydrates_g: 10, source: "nutrition_csv" },
            { name: "Tahu Goreng", calories: 150, protein_g: 11, fat_g: 9, carbohydrates_g: 5, source: "nutrition_csv" },
            { name: "Ikan Goreng", calories: 195, protein_g: 22, fat_g: 10, carbohydrates_g: 3, source: "nutrition_csv" },
            { name: "Sayur Bayam", calories: 23, protein_g: 2.9, fat_g: 0.4, carbohydrates_g: 3.6, source: "nutrition_csv" },
            { name: "Wortel", calories: 41, protein_g: 0.9, fat_g: 0.2, carbohydrates_g: 10, source: "nutrition_csv" },
            { name: "Kangkung Tumis", calories: 35, protein_g: 3, fat_g: 1, carbohydrates_g: 5, source: "nutrition_csv" },
            { name: "Pisang Mas", calories: 89, protein_g: 1.1, fat_g: 0.3, carbohydrates_g: 22.8, source: "nutrition_csv" },
            { name: "Apel", calories: 52, protein_g: 0.3, fat_g: 0.2, carbohydrates_g: 13.8, source: "nutrition_csv" },
            { name: "Jeruk", calories: 47, protein_g: 0.9, fat_g: 0.1, carbohydrates_g: 12, source: "nutrition_csv" },
            { name: "Mangga", calories: 60, protein_g: 0.8, fat_g: 0.4, carbohydrates_g: 15, source: "nutrition_csv" },
            { name: "Oatmeal Matang", calories: 68, protein_g: 2.4, fat_g: 1.4, carbohydrates_g: 12, source: "nutrition_csv" },
            { name: "Roti Gandum", calories: 247, protein_g: 10, fat_g: 3.4, carbohydrates_g: 44, source: "nutrition_csv" },
            { name: "Mie Goreng", calories: 130, protein_g: 3.6, fat_g: 3, carbohydrates_g: 21, source: "nutrition_csv" },
            { name: "Susu Sapi UHT", calories: 61, protein_g: 3.2, fat_g: 3.3, carbohydrates_g: 4.8, source: "nutrition_csv" },
            { name: "Susu Sapi UHT Low Fat", calories: 42, protein_g: 3.4, fat_g: 1.3, carbohydrates_g: 4.7, source: "nutrition_csv" },
            { name: "Yogurt Plain", calories: 59, protein_g: 10, fat_g: 0.4, carbohydrates_g: 3.6, source: "nutrition_csv" },
            { name: "Kacang Merah Rebus", calories: 127, protein_g: 8.7, fat_g: 0.5, carbohydrates_g: 22, source: "nutrition_csv" },
            { name: "Tahu Putih", calories: 76, protein_g: 8, fat_g: 4.8, carbohydrates_g: 1.9, source: "nutrition_csv" },
            { name: "Daging Sapi Rendang", calories: 193, protein_g: 20, fat_g: 10, carbohydrates_g: 6, source: "nutrition_csv" },
            { name: "Soto Ayam", calories: 90, protein_g: 8, fat_g: 4, carbohydrates_g: 7, source: "nutrition_csv" },
            { name: "Gado-gado", calories: 135, protein_g: 7, fat_g: 8, carbohydrates_g: 12, source: "nutrition_csv" },
        ];

        const q = query.toLowerCase();
        const filtered = OFFLINE_FOODS.filter(f => f.name.toLowerCase().includes(q));
        
        const container = $('librarySearchResults');
        if (filtered.length === 0) {
            container.innerHTML = '<p style="text-align:center; color:var(--text-muted); padding:20px;">Tidak ditemukan di data lokal. Coba jalankan backend untuk pencarian lengkap.</p>';
            return;
        }

        container.innerHTML = '<div style="font-size:0.72rem; color:var(--text-muted); text-align:center; margin-bottom:8px; padding:6px; background:rgba(255,193,7,0.1); border-radius:8px;">⚠️ Mode offline — hasil terbatas. Jalankan backend untuk 3.700+ makanan.</div>';
        const listDiv = document.createElement('div');
        listDiv.style.cssText = 'display:flex; flex-direction:column; gap:10px;';
        container.appendChild(listDiv);

        filtered.forEach(food => {
            const card = document.createElement('div');
            card.className = 'food-result-card';
            card.innerHTML = `
                <div class="frc-info">
                    <h4>${food.name}</h4>
                    <span class="frc-source indo">🇮🇩 Lokal (Offline)</span>
                    <div class="frc-macros">
                        <span>P: ${food.protein_g}g</span>
                        <span>L: ${food.fat_g}g</span>
                        <span>K: ${food.carbohydrates_g}g</span>
                    </div>
                </div>
                <div class="frc-cal">${Math.round(food.calories)} <span style="font-size:0.6rem; color:var(--text-muted);">kkal/100g</span></div>
            `;
            card.addEventListener('click', () => openManualLogModal(food));
            listDiv.appendChild(card);
        });
    }

    function openManualLogModal(food) {
        selectedLibraryFood = food;
        $('mlogTitle').textContent = food.name;
        
        const isIndo = food.source === 'nutrition_csv';
        $('mlogBadge').textContent = isIndo ? '🇮🇩 Database Lokal' : '🌍 Database Internasional';
        $('mlogBadge').style.color = isIndo ? '#d35400' : 'var(--text-muted)';
        
        $('mlogGrams').value = 100;
        updateManualLogCalculations(100);

        $$('.btn-preset-gram').forEach(b => b.classList.remove('active'));
        const preset100 = Array.from($$('.btn-preset-gram')).find(b => b.dataset.gram == "100");
        if (preset100) preset100.classList.add('active');

        $('manualLogOverlay').style.display = 'flex';
    }

    function updateManualLogCalculations(grams) {
        if (!selectedLibraryFood || isNaN(grams) || grams <= 0) return;
        const scale = grams / 100.0;
        
        $('mlogCal').textContent = Math.round(selectedLibraryFood.calories * scale);
        $('mlogPro').textContent = (selectedLibraryFood.protein_g * scale).toFixed(1) + 'g';
        $('mlogFat').textContent = (selectedLibraryFood.fat_g * scale).toFixed(1) + 'g';
        $('mlogCarb').textContent = (selectedLibraryFood.carbohydrates_g * scale).toFixed(1) + 'g';
    }

    if ($('mlogGrams')) {
        $('mlogGrams').addEventListener('input', (e) => {
            const val = parseFloat(e.target.value);
            updateManualLogCalculations(val);
            $$('.btn-preset-gram').forEach(b => b.classList.remove('active'));
        });
    }

    $$('.btn-preset-gram').forEach(btn => {
        btn.addEventListener('click', (e) => {
            $$('.btn-preset-gram').forEach(b => b.classList.remove('active'));
            e.currentTarget.classList.add('active');
            const gram = e.currentTarget.dataset.gram;
            $('mlogGrams').value = gram;
            updateManualLogCalculations(parseFloat(gram));
        });
    });

    if ($('btnCloseManualLog')) {
        $('btnCloseManualLog').addEventListener('click', () => {
            $('manualLogOverlay').style.display = 'none';
            selectedLibraryFood = null;
        });
    }

    if ($('btnSaveManualLog')) {
        $('btnSaveManualLog').addEventListener('click', () => {
            if (!selectedLibraryFood) return;
            const grams = parseFloat($('mlogGrams').value);
            if (isNaN(grams) || grams <= 0) {
                alert("Masukkan berat/porsi yang valid.");
                return;
            }

            const scale = grams / 100.0;
            const cal = Math.round(selectedLibraryFood.calories * scale);
            const pro = parseFloat((selectedLibraryFood.protein_g * scale).toFixed(1));
            const fat = parseFloat((selectedLibraryFood.fat_g * scale).toFixed(1));
            const carb = parseFloat((selectedLibraryFood.carbohydrates_g * scale).toFixed(1));

            const entry = normalizeEntry({
                id: 'log_' + Date.now(),
                date: new Date().toISOString(),
                name: selectedLibraryFood.name + ` (${grams}g)`,
                type: 'Pencarian Manual',
                cal: cal,
                protein: pro,
                fat: fat,
                carbohydrates_g: carb,
                sugar: 0,
                sodium: 0,
                fiber: 0
            });

            state.history.push(entry);
            saveState();
            haptic('success');
            launchConfetti();
            updateDashboard();

            $('manualLogOverlay').style.display = 'none';
            selectedLibraryFood = null;
            $('foodSearchInput').value = '';
            // Reset ke suggestions setelah simpan
            renderLibrarySuggestions();
            
            // Progress toast
            const todayStr = getLocalDateStr();
            const todayCal = state.history.filter(h => getLocalDateStr(h.date) === todayStr).reduce((a,h) => a + (h.cal||0), 0);
            const maxCal = state.limitCalorie || 2000;
            const pct = Math.min(100, Math.round((todayCal/maxCal)*100));
            showToast(`✅ ${entry.name} tercatat! Total: ${todayCal} kkal (${pct}%)`);
            switchView('viewDashboard');
        });
    }

    // ── Greeting System ────────────────────────────────────
    function updateGreeting(todayCal, limits) {
        const hour = new Date().getHours();
        const maxCal = (limits && limits.max_calories_kkal) || state.limitCalorie || 2000;
        const pct = Math.min(100, Math.round((todayCal / maxCal) * 100));

        let emoji, title, sub, gradient;
        if (hour >= 5 && hour < 11) {
            emoji = '🌅'; title = 'Selamat Pagi!';
            sub = todayCal > 0 ? `Sarapan tercatat ${todayCal} kkal. Lanjutkan hari sehatmu!` : 'Mulai hari dengan sarapan sehat dan air putih 💧';
            gradient = 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)';
        } else if (hour >= 11 && hour < 15) {
            emoji = '☀️'; title = 'Selamat Siang!';
            sub = pct >= 50 ? `Sudah ${pct}% target! Pilih makan siang bernutrisi 🥗` : 'Waktunya makan siang. Jangan skip ya!';
            gradient = 'linear-gradient(135deg, #f6d365 0%, #fda085 100%)';
        } else if (hour >= 15 && hour < 18) {
            emoji = '🌤️'; title = 'Selamat Sore!';
            sub = pct >= 70 ? `Hebat! Target kalori hampir penuh (${pct}%)` : 'Cemilan sehat? Buah atau kacang lebih baik 🍎';
            gradient = 'linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%)';
        } else if (hour >= 18 && hour < 21) {
            emoji = '🌙'; title = 'Selamat Malam!';
            sub = pct >= 80 ? `Target tercapai ${pct}%! Makan malam yang ringan ya 🌙` : 'Waktunya makan malam. Pilih menu ringan dan bergizi.';
            gradient = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
        } else {
            emoji = '🌜'; title = 'Istirahat yang Cukup!';
            sub = `Hindari makan terlalu larut. Tidur cukup = tubuh sehat 😴`;
            gradient = 'linear-gradient(135deg, #2c3e50 0%, #4ca1af 100%)';
        }

        const card = $('greetingCard');
        if (card) {
            card.style.background = gradient;
            $('greetingEmoji').textContent = emoji;
            $('greetingTitle').textContent = state.user ? `${title}` : title;
            $('greetingSub').textContent = sub;
            $('greetingProgressFill').style.width = `${pct}%`;
            $('greetingProgressText').textContent = `${pct}% kalori`;
        }
    }

    // ── Delete History Entry ──────────────────────────────
    window._deleteHistoryItem = function(index, btnEl) {
        if (index < 0 || index >= state.history.length) return;
        const item = state.history[index];
        const itemEl = btnEl.closest('.history-item');
        
        // Animate out
        if (itemEl) {
            itemEl.classList.add('deleting');
            haptic('light');
        }
        
        setTimeout(() => {
            state.history.splice(index, 1);
            saveState();
            showToast(`🗑️ ${item.name || 'Makanan'} dihapus dari riwayat`);
        }, 400);
    };

    // --- Init ---
    let initReady = false;
    document.addEventListener('DOMContentLoaded', () => {
        if(initReady) return; initReady = true;
        const n = new Date();
        const m = ['Jan','Feb','Mar','Apr','Mei','Jun','Jul','Agu','Sep','Okt','Nov','Des'];
        $('headerDate').textContent = `${n.getDate()} ${m[n.getMonth()]} ${n.getFullYear()}`;
        
        $$('[data-view]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const viewId = e.currentTarget.dataset.view;
                haptic('light');
                switchView(viewId);
                // Refresh hydration view when navigating to it
                if (viewId === 'viewHydration') {
                    updateHydrationView();
                }
                // Tampilkan suggestions saat masuk ke library
                if (viewId === 'viewLibrary') {
                    const foodInput = $('foodSearchInput');
                    if (foodInput) foodInput.value = '';
                    renderLibrarySuggestions();
                }
            });
        });

        $('btnSeeAll').addEventListener('click', () => switchView('viewHistory'));

        loadState();
        initHydrationEvents();
        initHydrationReminder();

        if(!state.user) {
            $('mainHeader').style.display = 'none';
            $('mainNav').style.display = 'none';
            if ($('greetingCard')) $('greetingCard').style.display = 'none';
            switchView('viewOnboarding');
        } else {
            $('mainHeader').style.display = 'flex';
            $('mainNav').style.display = 'flex';
            switchView('viewDashboard');
            // Init hydration mini widget with stored data
            updateHydrationMiniWidget();
        }
    });
})();
