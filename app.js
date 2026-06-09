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
            state.history = (parsed.history || []).map(h => ({
                ...h,
                sugar: h.sugar || 0, cal: h.cal || 0, fat: h.fat || 0,
                sodium: h.sodium || 0, protein: h.protein || 0, fiber: h.fiber || 0
            }));
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
        if (viewId === 'viewOcrScanner' || viewId === 'viewIsiPiringku' || viewId === 'viewCrop' || viewId === 'viewCorrection') {
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
            alert("Gagal menghubungi OCR lokal. Pastikan server FastAPI jalan di http://127.0.0.1:8000, lalu coba lagi atau isi manual.");
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

        currentCalculated = {
            id: Date.now().toString(), name,
            grade: finalGrade, sugar: gula * porsiFraksi, cal: energi * porsiFraksi, fat: lemak * porsiFraksi,
            sodium: sodium * porsiFraksi, protein: protein * porsiFraksi, fiber: fiber * porsiFraksi, servingSize_g: consumedServing,
            date: new Date().toISOString(),
            nutriScore: { grade: finalGrade, score, n_points: nP, p_points: pP, warnings: result.warnings, tips: result.tips },
            clinicalEval: clinicalEval
        };

        $('modalResult').classList.add('active');
    });

    $('btnCloseModal').addEventListener('click', () => {
        $('modalResult').classList.remove('active');
    });

    $('btnConfirmEat').addEventListener('click', () => {
        if(currentCalculated) {
            state.history.unshift(currentCalculated);
            saveState();
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

        state.user = {
            gender, age, weight_kg: weight, height_cm: height,
            bmi, bmiStatus, bmr, tdee,
            isDiabetic, hasNephropathy, hasDyslipidemia, isHypertensive
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
        const todayStr = new Date().toISOString().split('T')[0];
        const todayItems = state.history.filter(h => h.date.startsWith(todayStr));

        const sumSugar = todayItems.reduce((acc, h) => acc + (h.sugar||0), 0);
        const sumCal = todayItems.reduce((acc, h) => acc + (h.cal||0), 0);
        const sumFat = todayItems.reduce((acc, h) => acc + (h.fat||0), 0);

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
        const sumProtein = todayItems.reduce((a,h) => a + (h.protein||0), 0);
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
            list.innerHTML = `<div class="empty-state">Belum ada makanan hari ini.</div>`;
        } else {
            list.innerHTML = todayItems.slice(0, 3).map(h => `
                <div class="history-item">
                    <div class="history-icon-badge" style="background:${GRADE_COLORS_PASTEL[h.grade]}; box-shadow: 0 4px 12px ${GRADE_COLORS_PASTEL[h.grade]}60;">${h.grade}</div>
                    <div class="history-details">
                        <div class="history-title">${h.name}</div>
                        <div class="history-meta">+${h.cal}kkal • ${h.sugar}g gula</div>
                    </div>
                </div>
            `).join('');
        }

        updateStreakAndChart();

        // Global list in History Tab
        const viewHist = $('viewHistory');
        if(viewHist) {
            if(state.history.length === 0) {
                viewHist.innerHTML = `<h2 style="margin-bottom:16px;">Semua Riwayat</h2><div class="empty-state">Kosong</div>`;
            } else {
                viewHist.innerHTML = `<h2 style="margin-bottom:16px;">Semua Riwayat</h2>` + state.history.map(h => {
                    const d = new Date(h.date);
                    return `
                    <div class="history-item">
                        <div class="history-icon-badge" style="background:${GRADE_COLORS_PASTEL[h.grade]}; box-shadow: 0 4px 12px ${GRADE_COLORS_PASTEL[h.grade]}60;">${h.grade}</div>
                        <div class="history-details">
                            <div class="history-title">${h.name}</div>
                            <div class="history-meta">${d.getDate()}/${d.getMonth()+1} • +${h.cal}kkal • ${h.sugar}g Gula</div>
                        </div>
                    </div>`;
                }).join('');
            }
        }
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
            const dStr = d.toISOString().split('T')[0];
            const shortD = `${d.getDate()}/${d.getMonth()+1}`;
            
            // Total sum of that day
            const dailySum = state.history
                .filter(h => h.date.startsWith(dStr))
                .reduce((acc, h) => acc + h.sugar, 0);

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
            if (!window._lastSegResult) return;
            
            const btn = $('btnCorrRecalculate');
            btn.textContent = '⏳ Menghitung...';
            btn.disabled = true;
            
            setTimeout(() => {
                const pCtx = $('segCanvas').getContext('2d');
                const cCtx = $('correctionCanvas').getContext('2d');
                const w = $('segCanvas').width;
                const h = $('segCanvas').height;
                const pData = pCtx.getImageData(0,0,w,h).data;
                const cData = cCtx.getImageData(0,0,w,h).data;
                
                let counts = { makanan_pokok:0, lauk_pauk:0, sayur:0, buah:0 };
                let isModified = false;
                
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

                for(let i = 0; i < pData.length; i += 4) {
                    let cat = matchColor(cData[i], cData[i+1], cData[i+2], cData[i+3], 40);
                    if (cat) {
                        counts[cat]++;
                        isModified = true;
                    } else {
                        cat = matchColor(pData[i], pData[i+1], pData[i+2], pData[i+3], 10);
                        if (cat) counts[cat]++;
                    }
                }
                
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
                    
                    $('sliderAutoLabel').innerHTML = '✏️ Dikoreksi Manual';
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
    $('btnAnalyzePlate').addEventListener('click', () => {
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
    });

    // --- Init ---
    let initReady = false;
    document.addEventListener('DOMContentLoaded', () => {
        if(initReady) return; initReady = true;
        const n = new Date();
        const m = ['Jan','Feb','Mar','Apr','Mei','Jun','Jul','Agu','Sep','Okt','Nov','Des'];
        $('headerDate').textContent = `${n.getDate()} ${m[n.getMonth()]} ${n.getFullYear()}`;
        
        $$('[data-view]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                switchView(e.currentTarget.dataset.view);
            });
        });

        $('btnSeeAll').addEventListener('click', () => switchView('viewHistory'));

        loadState();

        if(!state.user) {
            $('mainHeader').style.display = 'none';
            $('mainNav').style.display = 'none';
            switchView('viewOnboarding');
        } else {
            $('mainHeader').style.display = 'flex';
            $('mainNav').style.display = 'flex';
            switchView('viewDashboard');
        }
    });
})();
