/**
 * ============================================================
 * CLINICAL HEALTH RULES ENGINE
 * ============================================================
 * Pedoman: PERKENI 2024 (Diabetes), JNC 8 + DASH Diet (Hipertensi)
 * @module ClinicalHealthRules
 * @version 1.0.0
 */
window.ClinicalHealthRules = (function () {
    'use strict';

    /**
     * @param {Object} profile
     * @param {number} profile.age
     * @param {string} profile.gender
     * @param {number} profile.weight_kg
     * @param {number} profile.height_cm
     * @param {number} profile.bmr
     * @param {number} profile.tdee
     * @param {boolean} profile.isDiabetic
     * @param {boolean} profile.hasNephropathy
     * @param {boolean} profile.hasDyslipidemia
     * @param {boolean} profile.isHypertensive
     */
    function ClinicalRules(profile) {
        this.profile = profile;
        this.limits = this.getDailyLimits();
    }

    /** Hitung batas harian berdasarkan profil klinis */
    ClinicalRules.prototype.getDailyLimits = function () {
        const p = this.profile;
        const tdee = p.tdee || 2000;

        // BASE (semua user)
        const limits = {
            max_snack_calories: tdee * 0.20,
            max_sugar_g: (tdee * 0.10) / 4,
            max_sodium_mg: 2300,
            target_fiber_g: { min: 25, max: 38 },
            max_saturated_fat_g: (tdee * 0.10) / 9,
            max_protein_g: null,
            max_cholesterol_mg: null,
            target_carb_percent: null,
            protein_quality_note: null,
            sodium_warning_threshold: 800,
            _sources: ['WHO General Guidelines']
        };

        // OVERRIDE: Diabetes (PERKENI 2024)
        if (p.isDiabetic) {
            limits.max_sugar_g = (tdee * 0.05) / 4;
            limits.target_carb_percent = { min: 45, max: 65 };
            limits.target_fiber_g = { min: 20, max: 35 };
            limits._sources.push('PERKENI 2024');

            if (p.hasDyslipidemia) {
                limits.max_saturated_fat_g = (tdee * 0.07) / 9;
                limits.max_cholesterol_mg = 200;
                limits._sources.push('PERKENI 2024 — Dislipidemia');
            }

            // Nefropati Diabetik
            if (p.hasNephropathy) {
                limits.max_protein_g = p.weight_kg * 0.8;
                limits.protein_quality_note =
                    '65% harus protein hewani (nilai biologi tinggi): telur, daging, ikan, susu';
                limits._sources.push('PERKENI 2024 — Nefropati');
            }
        }

        // OVERRIDE: Hipertensi (JNC 8 + DASH)
        if (p.isHypertensive) {
            limits.max_sodium_mg = 1500;
            limits.sodium_warning_threshold = 600;
            limits._sources.push('JNC 8 + DASH Diet');
        }

        return limits;
    };

    /** Edukasi target tekanan darah */
    ClinicalRules.prototype.getBloodPressureTarget = function () {
        const p = this.profile;
        if (p.isDiabetic || p.hasNephropathy) {
            return 'Target tekanan darah Anda: < 140/90 mmHg (Standar JNC 8 untuk pasien diabetes/gangguan ginjal). Pantau tekanan darah secara rutin.';
        }
        if (p.age >= 60 && !p.isDiabetic && !p.hasNephropathy) {
            return 'Target tekanan darah untuk usia ≥60 tahun tanpa diabetes/CKD: < 150/90 mmHg (JNC 8).';
        }
        return 'Target tekanan darah normal: < 140/90 mmHg.';
    };

    /**
     * Evaluasi makanan berdasarkan profil klinis
     * @param {Object} scanned - Nutrisi per 100g/ml
     * @param {number} servingSize - Ukuran porsi aktual (g/ml)
     */
    ClinicalRules.prototype.evaluateMeal = function (scanned, servingSize) {
        const p = this.profile;
        const L = this.limits;
        const mult = servingSize / 100;

        const actual = {
            sugar: (scanned.sugar_g || 0) * mult,
            sodium: (scanned.sodium_mg || 0) * mult,
            saturated_fat: (scanned.saturated_fat_g || 0) * mult,
            protein: (scanned.protein_g || 0) * mult,
            fiber: (scanned.fiber_g || 0) * mult,
            calories: ((scanned.energy_kj || 0) / 4.184) * mult
        };

        const warnings = [];
        const suggestions = [];
        let status = 'SAFE';

        const pct = {
            sugar: L.max_sugar_g ? ((actual.sugar / L.max_sugar_g) * 100) : 0,
            sodium: L.max_sodium_mg ? ((actual.sodium / L.max_sodium_mg) * 100) : 0,
            saturated_fat: L.max_saturated_fat_g ? ((actual.saturated_fat / L.max_saturated_fat_g) * 100) : 0,
            protein: L.max_protein_g ? ((actual.protein / L.max_protein_g) * 100) : null
        };

        // ── Diabetes checks ──
        if (p.isDiabetic) {
            if (actual.sugar > L.max_sugar_g) {
                status = 'DANGER';
                warnings.push(`🚨 Gula melebihi batas harian! (${actual.sugar.toFixed(1)}g > ${L.max_sugar_g.toFixed(1)}g). Hentikan konsumsi gula tambahan hari ini.`);
            } else if (actual.sugar > L.max_sugar_g * 0.5) {
                status = status === 'DANGER' ? 'DANGER' : 'WARNING';
                warnings.push(`⚠️ Gula dalam produk ini ${actual.sugar.toFixed(1)}g — sudah ${pct.sugar.toFixed(0)}% dari batas harian Anda (${L.max_sugar_g.toFixed(1)}g). Pertimbangkan alternatif rendah gula.`);
            }
        }

        // ── Nefropati checks ──
        if (p.isDiabetic && p.hasNephropathy && L.max_protein_g) {
            if (actual.protein > L.max_protein_g * 0.30) {
                status = status === 'DANGER' ? 'DANGER' : 'WARNING';
                warnings.push(`⚠️ PERHATIAN GINJAL: Protein ${actual.protein.toFixed(1)}g sudah ${pct.protein.toFixed(0)}% dari batas harian Anda (${L.max_protein_g.toFixed(1)}g/hari = 0.8g × ${p.weight_kg}kg).`);
            }
        }

        // ── Hipertensi checks ──
        if (p.isHypertensive) {
            if ((scanned.sodium_mg || 0) > L.sodium_warning_threshold) {
                status = status === 'DANGER' ? 'DANGER' : 'WARNING';
                warnings.push(`🧂 TINGGI GARAM — Produk ini mengandung ${(scanned.sodium_mg||0).toFixed(0)}mg sodium/100g. Berisiko untuk tekanan darah Anda.`);
            }
            if (actual.sodium > L.max_sodium_mg * 0.40) {
                status = status === 'DANGER' ? 'DANGER' : 'WARNING';
                warnings.push(`🧂 Sodium sudah ${pct.sodium.toFixed(0)}% dari batas harian Anda (${L.max_sodium_mg}mg).`);
            }
        }

        // ── Dislipidemia checks ──
        if (p.hasDyslipidemia || p.isDiabetic) {
            if (actual.saturated_fat > L.max_saturated_fat_g * 0.5) {
                status = status === 'DANGER' ? 'DANGER' : 'WARNING';
                warnings.push(`🧈 Lemak jenuh ${actual.saturated_fat.toFixed(1)}g — ${pct.saturated_fat.toFixed(0)}% dari batas harian Anda. Perhatikan asupan lemak jenuh hari ini.`);
            }
        }

        // ── General sugar check for non-diabetic ──
        if (!p.isDiabetic) {
            if (actual.sugar > L.max_sugar_g) {
                status = 'DANGER';
                warnings.push(`🚨 Gula melebihi batas harian Anda! (${actual.sugar.toFixed(1)}g dari ${L.max_sugar_g.toFixed(1)}g).`);
            } else if (actual.sugar > L.max_sugar_g * 0.5) {
                status = status === 'DANGER' ? 'DANGER' : 'WARNING';
                warnings.push(`⚠️ Gula ${actual.sugar.toFixed(1)}g sudah ${pct.sugar.toFixed(0)}% dari batas harian (${L.max_sugar_g.toFixed(1)}g).`);
            }
        }

        // ── Suggestions ──
        if (status === 'DANGER') {
            suggestions.push('Pertimbangkan untuk tidak mengonsumsi produk ini, atau kurangi porsi hingga setengah.');
            suggestions.push('Pilih alternatif dengan Nutri-Score A atau B.');
        } else if (status === 'WARNING') {
            suggestions.push('Boleh dikonsumsi dengan porsi terbatas. Perhatikan asupan selanjutnya hari ini.');
        } else {
            suggestions.push('Produk ini aman untuk dikonsumsi sesuai porsi saji.');
        }

        return {
            status,
            clinical_warnings: warnings,
            suggestions,
            percentage_of_daily: pct,
            actual_consumed: actual
        };
    };

    /** Factory function */
    function create(profile) {
        return new ClinicalRules(profile);
    }

    return { create };
})();
