/**
 * ============================================================
 * ISI PIRINGKU — Dietary Meal Proportion Validator
 * ============================================================
 * Ref: Ridhani et al. (2021) "Isi Piringku Dietary Meal Proportion
 * Estimator Applications Using SeeFood Image Segmentations"
 * Sanitas Vol.12 No.2, 115-130.
 *
 * @module IsiPiringku
 * @version 1.0.0
 */
window.IsiPiringku = (function () {
    'use strict';

    // Label mapping SeeFood → Isi Piringku (Tabel 1 Ridhani et al.)
    const LABEL_MAP = {
        MAKANAN_POKOK: [
            'starch/grains: noodles/pasta', 'starch/grains: rice/grains/cereals',
            'starch/grains: baked_goods', 'starch/grains: starchy_vegetables',
            'starch/grains: other'
        ],
        LAUK_PAUK: [
            'protein: eggs', 'protein: beans/nuts', 'protein: meat',
            'protein: poultry', 'protein: seafood', 'dairy',
            'herbs/spices', 'fats/oils/sauces', 'soups/stews'
        ],
        SAYUR: [
            'vegetables: stem_vegetables', 'vegetables: leafy_greens',
            'vegetables: non-starchy_roots', 'vegetables: other'
        ],
        BUAH: ['fruits'],
        IGNORED: ['background', 'food container', 'dining tools', 'beverages', 'snack', 'sweet/desserts', 'other_food']
    };

    // Default thresholds (Ridhani et al., 2021)
    const DEFAULT_THRESHOLDS = {
        makanan_pokok: { ideal: 33.33, warn_above: 40, warn_below: 26 },
        sayur:         { ideal: 33.33, warn_above: 45, warn_below: 26 },
        lauk_pauk:     { ideal: 16.66, warn_above: 25, warn_below: 13 },
        buah:          { ideal: 16.66, warn_above: 25, warn_below: 13 }
    };

    /**
     * Get clinical-adjusted thresholds
     * @param {Object} profile - User clinical profile
     */
    function getThresholds(profile) {
        const t = JSON.parse(JSON.stringify(DEFAULT_THRESHOLDS));
        if (profile && profile.isDiabetic) {
            t.makanan_pokok.warn_above = 35; // lebih ketat
        }
        if (profile && profile.isHypertensive) {
            t.sayur.warn_below = 30; // lebih ketat
        }
        if (profile && profile.hasNephropathy) {
            t.lauk_pauk.warn_above = 20; // batasi protein
        }
        return t;
    }

    /**
     * Validate Isi Piringku proportions
     * @param {Object} seg - { makanan_pokok_percent, lauk_pauk_percent, sayur_percent, buah_percent }
     * @param {Object} [userProfile] - Optional clinical profile
     */
    function validateIsiPiringku(seg, userProfile) {
        const t = getThresholds(userProfile);
        const messages = [];
        const clinical_notes = [];
        let isBalanced = true;

        const ideal_vs_actual = {
            makanan_pokok: { ideal: t.makanan_pokok.ideal, actual: seg.makanan_pokok_percent || 0 },
            sayur:         { ideal: t.sayur.ideal,         actual: seg.sayur_percent || 0 },
            lauk_pauk:     { ideal: t.lauk_pauk.ideal,     actual: seg.lauk_pauk_percent || 0 },
            buah:          { ideal: t.buah.ideal,           actual: seg.buah_percent || 0 }
        };

        // Makanan Pokok
        if (ideal_vs_actual.makanan_pokok.actual > t.makanan_pokok.warn_above) {
            isBalanced = false;
            messages.push(`🍚 Porsi karbohidrat terlalu banyak (${ideal_vs_actual.makanan_pokok.actual.toFixed(1)}%). Idealnya sekitar ${t.makanan_pokok.ideal.toFixed(0)}%.`);
            if (userProfile && userProfile.isDiabetic) {
                clinical_notes.push('🩺 Diabetes: Kurangi porsi karbohidrat untuk mengontrol gula darah. Pilih sumber karbohidrat kompleks (nasi merah, ubi).');
            }
        } else if (ideal_vs_actual.makanan_pokok.actual < t.makanan_pokok.warn_below) {
            messages.push(`🍚 Porsi karbohidrat kurang (${ideal_vs_actual.makanan_pokok.actual.toFixed(1)}%). Tambahkan sumber energi.`);
        }

        // Sayur
        if (ideal_vs_actual.sayur.actual < t.sayur.warn_below) {
            isBalanced = false;
            messages.push(`🥬 Porsi sayuran kurang (${ideal_vs_actual.sayur.actual.toFixed(1)}%). Idealnya minimal ${t.sayur.warn_below}%.`);
            if (userProfile && userProfile.isHypertensive) {
                clinical_notes.push('💊 Hipertensi: Tingkatkan sayuran kaya kalium (bayam, brokoli, kangkung) sesuai DASH Diet.');
            }
        }

        // Lauk Pauk
        if (ideal_vs_actual.lauk_pauk.actual < t.lauk_pauk.warn_below) {
            isBalanced = false;
            messages.push(`🍗 Porsi lauk pauk kurang (${ideal_vs_actual.lauk_pauk.actual.toFixed(1)}%). Tambahkan sumber protein.`);
        } else if (ideal_vs_actual.lauk_pauk.actual > t.lauk_pauk.warn_above) {
            isBalanced = false;
            messages.push(`🍗 Porsi lauk pauk berlebih (${ideal_vs_actual.lauk_pauk.actual.toFixed(1)}%). Kurangi sedikit.`);
            if (userProfile && userProfile.hasNephropathy) {
                clinical_notes.push('⚕️ Nefropati: Batasi protein. Pilih protein hewani berkualitas tinggi (telur, ikan) dalam porsi kecil.');
            }
        }

        // Buah
        if (ideal_vs_actual.buah.actual < t.buah.warn_below) {
            isBalanced = false;
            messages.push(`🍎 Porsi buah kurang (${ideal_vs_actual.buah.actual.toFixed(1)}%). Tambahkan buah segar.`);
        }

        if (isBalanced) {
            messages.unshift('✅ Komposisi piringmu sudah seimbang! Pertahankan pola makan ini.');
        } else {
            messages.unshift('⚠️ Komposisi piringmu belum seimbang. Perhatikan saran berikut:');
        }

        return { is_balanced: isBalanced, messages, ideal_vs_actual, clinical_notes };
    }

    /**
     * Calculate proportions from pixel counts
     * @param {Object} pixels - { makanan_pokok, lauk_pauk, sayur, buah, ignored }
     * @param {number} totalPixels
     */
    function calculateProportions(pixels, totalPixels) {
        const foodPixels = totalPixels - (pixels.ignored || 0);
        if (foodPixels <= 0) return { makanan_pokok_percent:0, lauk_pauk_percent:0, sayur_percent:0, buah_percent:0 };
        return {
            makanan_pokok_percent: ((pixels.makanan_pokok || 0) / foodPixels) * 100,
            lauk_pauk_percent:     ((pixels.lauk_pauk || 0) / foodPixels) * 100,
            sayur_percent:         ((pixels.sayur || 0) / foodPixels) * 100,
            buah_percent:          ((pixels.buah || 0) / foodPixels) * 100
        };
    }

    return { validateIsiPiringku, calculateProportions, LABEL_MAP, DEFAULT_THRESHOLDS, getThresholds };
})();
