/**
 * ============================================================
 * NUTRI-SCORE ENGINE — FSA/Ofcom European Algorithm
 * ============================================================
 * Referensi: Rayner et al. (2005), Santé Publique France (2021)
 * @module NutriScore
 * @version 2.0.0
 */
window.NutriScore = (function () {
    'use strict';

    const GRADE_COLORS = { 'A':'#1a9641','B':'#a6d96a','C':'#ffffbf','D':'#fdae61','E':'#d7191c' };
    const GRADE_LABELS = { 'A':'Sangat Sehat! ✨','B':'Pilihan Baik 👍','C':'Konsumsi Secukupnya ⚠️','D':'Kurang Sehat 🔴','E':'Sangat Tidak Sehat! 🚨' };

    // N-Points thresholds (0-10)
    const ENERGY_T = [335,670,1005,1340,1675,2010,2345,2680,3015,3350];
    const SUGAR_T  = [4.5,9,13.5,18,22.5,27,31,36,40,45];
    const SATFAT_T = [1,2,3,4,5,6,7,8,9,10];
    const SODIUM_T = [90,180,270,360,450,540,630,720,810,900];

    // Beverage specific thresholds (0-10)
    const BEV_ENERGY_T = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270];
    const BEV_SUGAR_T = [0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12, 13.5];

    // P-Points thresholds (0-5)
    const FIBER_T   = [0.9,1.9,2.8,3.7,4.7];
    const PROTEIN_T = [1.6,3.2,4.8,6.4,8.0];

    function calcPts(value, thresholds) {
        let pts = 0;
        for (let i = 0; i < thresholds.length; i++) {
            if (value > thresholds[i]) pts = i + 1; else break;
        }
        return pts;
    }

    function calcFvnPts(pct) {
        if (pct > 80) return 5;
        if (pct > 60) return 4;
        if (pct > 40) return 3;
        if (pct > 20) return 2;
        if (pct > 0) return 1;
        return 0;
    }

    function calcMicroPts(micro) {
        if (!micro) return { vitamin_c: 0, calcium: 0, iron: 0, b_complex: 0, folate: 0, total: 0 };
        let p = { vitamin_c: 0, calcium: 0, iron: 0, b_complex: 0, folate: 0, total: 0 };
        
        if (micro.vitamin_c_mg >= 160) p.vitamin_c = 2;
        else if (micro.vitamin_c_mg >= 80) p.vitamin_c = 1;
        
        if (micro.calcium_mg >= 400) p.calcium = 1;
        if (micro.iron_mg >= 7) p.iron = 1;
        if (micro.vitamin_b1_mg > 0 && micro.vitamin_b6_mg > 0 && micro.vitamin_b12_mcg > 0) p.b_complex = 1;
        if (micro.folate_mcg >= 200) p.folate = 1;
        
        let pts = p.vitamin_c + p.calcium + p.iron + p.b_complex + p.folate;
        p.total = Math.min(pts, 3);
        return p;
    }

    /**
     * @param {Object} n - { energy_kj, sugar_g, saturated_fat_g, sodium_mg, fiber_g, protein_g, fvn_percent }
     * @param {boolean} isLiquid
     * @param {Object} micronutrients
     * @param {Object} context - { actualConsumed, porsiFraksi }
     */
    function calculateNutriScore(n, isLiquid, micronutrients = {}, context = {}) {
        isLiquid = isLiquid || false;
        const ePts = calcPts(n.energy_kj||0, isLiquid ? BEV_ENERGY_T : ENERGY_T);
        const sPts = calcPts(n.sugar_g||0, isLiquid ? BEV_SUGAR_T : SUGAR_T);
        const fPts = calcPts(n.saturated_fat_g||0, SATFAT_T);
        const naPts = calcPts(n.sodium_mg||0, SODIUM_T);
        const totalN = ePts + sPts + fPts + naPts;

        const fvnPts = calcFvnPts(n.fvn_percent||0);
        const fibPts = calcPts(n.fiber_g||0, FIBER_T);
        let proPts = calcPts(n.protein_g||0, PROTEIN_T);

        // Protein modification: if N>=11 AND fvn<80%, ignore protein
        let proApplied = proPts;
        if (totalN >= 11 && (n.fvn_percent||0) < 80) proApplied = 0;

        const totalP = fvnPts + fibPts + proApplied;
        const microPtsObj = calcMicroPts(micronutrients);
        const finalTotalP = totalP + microPtsObj.total;
        const score = totalN - finalTotalP;

        let grade;
        if (isLiquid) {
            if (score<=-1) grade='A'; else if (score<=2) grade='B';
            else if (score<=6) grade='C'; else if (score<=9) grade='D'; else grade='E';
        } else {
            if (score<=-1) grade='A'; else if (score<=2) grade='B';
            else if (score<=10) grade='C'; else if (score<=18) grade='D'; else grade='E';
        }

        const n_points = { energy:ePts, sugar:sPts, saturated_fat:fPts, sodium:naPts, total:totalN };
        const p_points = { fvn:fvnPts, fiber:fibPts, protein:proPts, protein_applied:proApplied, total:totalP };

        return {
            grade, score,
            color: GRADE_COLORS[grade],
            label: GRADE_LABELS[grade],
            n_points, p_points,
            micro_points: microPtsObj,
            warnings: generateWarnings(n_points, p_points, grade),
            tips: generateTips(grade, context.actualConsumed, context.porsiFraksi, microPtsObj)
        };
    }

    function generateWarnings(nP, pP, grade) {
        const w = [];
        if (nP.sugar>=7) w.push('🍬 Kandungan gula SANGAT tinggi. Waspada lonjakan gula darah mendadak.');
        else if (nP.sugar>=4) w.push('🍬 Kandungan gula cukup tinggi. Perhatikan porsi konsumsi.');
        if (nP.sodium>=7) w.push('⚠️ TINGGI GARAM — Berisiko untuk tekanan darah tinggi.');
        else if (nP.sodium>=4) w.push('🧂 Kandungan garam cukup tinggi. Batasi asupan garam hari ini.');
        if (nP.saturated_fat>=7) w.push('🧈 Lemak jenuh sangat tinggi. Batasi konsumsi harian.');
        else if (nP.saturated_fat>=4) w.push('🧈 Lemak jenuh cukup tinggi. Perhatikan total asupan hari ini.');
        if (nP.energy>=8) w.push('⚡ Kalori sangat padat. Pertimbangkan porsi lebih kecil.');
        if (pP.fiber>=3) w.push('🥬 Kandungan serat baik. Membantu pencernaan dan kenyang lebih lama.');
        if (pP.protein>=3) w.push('💪 Sumber protein yang baik.');
        return w;
    }

    function generateTips(grade, actualConsumed, porsiFraksi, microPtsObj) {
        const t = [];
        const frac = parseFloat(porsiFraksi) || 1.0;
        const hasMicro = microPtsObj && microPtsObj.total > 0;
        
        switch(grade) {
            case 'A':
                t.push('Pilihan sangat sehat! Produk ini bisa menjadi bagian rutin diet harianmu.');
                if (hasMicro) t.push('Sangat bagus karena mengandung ekstra vitamin/mineral esensial!');
                break;
            case 'B':
                t.push('Pilihan yang baik. Boleh dikonsumsi secara rutin dalam porsi wajar.');
                if (frac < 1.0) t.push('Porsi sudah diatur dengan baik.');
                break;
            case 'C':
                if (hasMicro) {
                    t.push('Meski grade C, produk ini kaya vitamin. Konsumsi wajar tetap aman.');
                } else {
                    t.push('Konsumsi secukupnya. Kombinasikan dengan buah atau sayur segar.');
                }
                break;
            case 'D':
            case 'E':
                if (frac < 1.0) {
                    let sugarConsumed = actualConsumed && actualConsumed.nutrition_consumed ? actualConsumed.nutrition_consumed.sugar_g : null;
                    if (sugarConsumed !== null && sugarConsumed !== undefined) {
                        let fStr = frac === 0.5 ? '½' : frac === 0.25 ? '¼' : frac === 0.75 ? '¾' : frac;
                        t.push(`Porsi dibatasi, bagus! Dengan ${fStr} sajian, gula hanya ${(sugarConsumed).toFixed(1)}g.`);
                    } else {
                        t.push(`Porsi sudah dibatasi (${frac} sajian), bagus! Hindari konsumsi lagi hari ini.`);
                    }
                } else {
                    t.push('Pertimbangkan kurangi jadi setengah porsi atau kurang untuk membatasi asupan gizi negatif.');
                }
                if (grade === 'E') t.push('Jadikan sebagai "cheat day" saja, bukan kebiasaan harian.');
                break;
        }
        return t;
    }

    return { calculateNutriScore, generateWarnings, generateTips, GRADE_COLORS, GRADE_LABELS };
})();
