/**
 * ============================================================
 * HYDRATION ENGINE & TRACKER
 * ============================================================
 * Personalized hydration tracking with gamification.
 * Formula: water_ml = weight_kg × 35 + activity_modifier
 * 
 * @module HydrationEngine
 * @version 1.0.0
 */
window.HydrationEngine = (function () {
    'use strict';

    // ── Configuration ──────────────────────────────────────
    const STORAGE_KEY = 'NutriLabel_Hydration_Logs';
    const GLASS_ML = 250; // Standard glass size

    // Activity modifiers (mL added to base)
    const ACTIVITY_MODIFIERS = {
        sedentary: 0,
        moderate: 500,
        active: 1000
    };

    // Mood states based on hydration percentage
    const MOOD_STATES = [
        { min: 0,   max: 30,  emoji: '😵', text: 'Haus berat', color: '#D94848', bg: 'rgba(217,72,72,0.1)' },
        { min: 31,  max: 60,  emoji: '🙂', text: 'Lumayan terhidrasi', color: '#F4C45A', bg: 'rgba(244,196,90,0.1)' },
        { min: 61,  max: 90,  emoji: '💧', text: 'Tubuhmu happy', color: '#48C78E', bg: 'rgba(72,199,142,0.1)' },
        { min: 91,  max: 120, emoji: '✨', text: 'Hydration Master', color: '#8ED96C', bg: 'rgba(142,217,108,0.15)' },
        { min: 121, max: 9999, emoji: '🌊', text: 'Overhidrasi! Kurangi', color: '#0984e3', bg: 'rgba(116,185,255,0.15)' }
    ];

    // Urine color feedback (scale 1-7, matching the urine colour.jpg chart)
    const URINE_FEEDBACK = [
        {
            level: 1, label: 'Over Hydrated', color: '#F5F5DC',
            text: 'Hidrasi berlebihan! Kurangi sedikit asupan air. Tubuhmu sudah sangat terhidrasi 💧',
            severity: 'info'
        },
        {
            level: 2, label: 'Good', color: '#FAFAD2',
            text: 'Hidrasi sempurna! Tubuhmu bekerja optimal bagai mesin AI tanpa bug 💧',
            severity: 'good'
        },
        {
            level: 3, label: 'Fair', color: '#F0E68C',
            text: 'Hidrasi baik! Pertahankan pola minummu, tubuh cukup terhidrasi 👍',
            severity: 'good'
        },
        {
            level: 4, label: 'Light Dehydrated', color: '#DAA520',
            text: 'Mulai ada tanda dehidrasi ringan. Yuk minum 1–2 gelas air sekarang ⚠️',
            severity: 'warning'
        },
        {
            level: 5, label: 'Dehydrated', color: '#CD853F',
            text: 'Dehidrasi sedang! Segera minum air putih minimal 2 gelas. Jangan tunda ya! 💦',
            severity: 'warning'
        },
        {
            level: 6, label: 'Very Dehydrated', color: '#D2691E',
            text: 'Dehidrasi tingkat lanjut. Segera tingkatkan asupan cairan dan pertimbangkan konsultasi medis bila berlanjut ⚕️',
            severity: 'danger'
        },
        {
            level: 7, label: 'Severe Dehydrated', color: '#8B4513',
            text: 'Dehidrasi berat! Segera minum banyak air dan hubungi tenaga kesehatan jika kondisi tidak membaik ⚕️🚨',
            severity: 'danger'
        }
    ];

    // ── Calculation Engine ─────────────────────────────────

    /**
     * Calculate personalized daily hydration target.
     * @param {number} weight_kg - User weight
     * @param {string} activity_level - 'sedentary' | 'moderate' | 'active'
     * @returns {{ target_ml: number, target_liters: number, target_glasses: number }}
     */
    function calculateDailyTarget(weight_kg, activity_level) {
        const base = (weight_kg || 60) * 35;
        const modifier = ACTIVITY_MODIFIERS[activity_level] || 0;
        const target_ml = base + modifier;
        return {
            target_ml: Math.round(target_ml),
            target_liters: parseFloat((target_ml / 1000).toFixed(1)),
            target_glasses: Math.ceil(target_ml / GLASS_ML)
        };
    }

    /**
     * Get mood state based on hydration percentage and time of day.
     * @param {number} consumed_ml
     * @param {number} target_ml
     * @returns {{ emoji: string, text: string, color: string, bg: string, percent: number }}
     */
    function getHydrationStatus(consumed_ml, target_ml) {
        const pct = target_ml > 0 ? Math.round((consumed_ml / target_ml) * 100) : 0;
        let mood = MOOD_STATES.find(m => pct >= m.min && pct <= m.max) || MOOD_STATES[0];
        
        // Time-aware adjustment
        const hour = new Date().getHours();
        if (pct < 30) {
            if (hour >= 5 && hour < 11) {
                mood = { emoji: '🌅', text: 'Pagi! Yuk mulai minum air', color: '#F4C45A', bg: 'rgba(244,196,90,0.1)' };
            } else if (hour >= 11 && hour < 15) {
                mood = { emoji: '🥵', text: 'Siang panas, jangan lupa minum!', color: '#D94848', bg: 'rgba(217,72,72,0.1)' };
            } else if (hour >= 20) {
                mood = { emoji: '😵', text: 'Aduh, kurang minum hari ini!', color: '#D94848', bg: 'rgba(217,72,72,0.1)' };
            }
        } else if (pct >= 30 && pct < 60 && hour >= 18) {
            mood = { emoji: '🌙', text: 'Malam tiba, kejar target minummu!', color: '#F4C45A', bg: 'rgba(244,196,90,0.1)' };
        }

        return { ...mood, percent: pct };
    }

    /**
     * Get urine color feedback by index (0-6 → level 1-7).
     * @param {number} colorIndex - 0-based index from the color scale
     * @returns {{ level: number, label: string, color: string, text: string, severity: string }}
     */
    function getUrineAdvice(colorIndex) {
        const idx = Math.max(0, Math.min(6, colorIndex));
        return URINE_FEEDBACK[idx];
    }

    // ── LocalStorage Persistence ──────────────────────────

    /**
     * Get all hydration logs from storage.
     * @returns {Array<{date: string, target_ml: number, entries: Array<{time: string, ml: number}>}>}
     */
    function getAllLogs() {
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            return raw ? JSON.parse(raw) : [];
        } catch (e) {
            console.warn('[HydrationEngine] Failed to load logs:', e);
            return [];
        }
    }

    /**
     * Save all logs to storage.
     * @param {Array} logs
     */
    function saveLogs(logs) {
        try {
            // Keep max 30 days of data to prevent bloat
            const trimmed = logs.slice(-30);
            localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
        } catch (e) {
            console.warn('[HydrationEngine] Failed to save logs:', e);
        }
    }

    /**
     * Get today's date string (YYYY-MM-DD) in local timezone.
     * @returns {string}
     */
    function getTodayStr() {
        const d = new Date();
        return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
    }

    /**
     * Get or create today's log entry.
     * @param {number} target_ml - Today's target
     * @returns {{ log: Object, logs: Array, index: number }}
     */
    function getTodayLog(target_ml) {
        const logs = getAllLogs();
        const todayStr = getTodayStr();
        let idx = logs.findIndex(l => l.date === todayStr);
        if (idx === -1) {
            logs.push({ date: todayStr, target_ml: target_ml || 2000, entries: [] });
            idx = logs.length - 1;
        }
        // Update target if changed
        if (target_ml) logs[idx].target_ml = target_ml;
        return { log: logs[idx], logs, index: idx };
    }

    /**
     * Log water intake.
     * @param {number} amount_ml
     * @param {number} target_ml - Current daily target
     * @returns {{ todayTotal: number, entries: Array, percent: number }}
     */
    function logIntake(amount_ml, target_ml) {
        if (!amount_ml || amount_ml <= 0) return null;
        const { log, logs } = getTodayLog(target_ml);
        const now = new Date();
        const timeStr = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
        log.entries.push({ time: timeStr, ml: Math.round(amount_ml) });
        saveLogs(logs);
        const todayTotal = log.entries.reduce((sum, e) => sum + e.ml, 0);
        return {
            todayTotal,
            entries: log.entries,
            percent: target_ml > 0 ? Math.round((todayTotal / target_ml) * 100) : 0
        };
    }

    /**
     * Get today's total consumption.
     * @param {number} target_ml
     * @returns {number}
     */
    function getTodayTotal(target_ml) {
        const { log } = getTodayLog(target_ml);
        return log.entries.reduce((sum, e) => sum + e.ml, 0);
    }

    /**
     * Undo the last intake entry.
     * @param {number} target_ml
     * @returns {{ todayTotal: number, removed: Object|null }}
     */
    function undoLastIntake(target_ml) {
        const { log, logs } = getTodayLog(target_ml);
        const removed = log.entries.pop() || null;
        saveLogs(logs);
        const todayTotal = log.entries.reduce((sum, e) => sum + e.ml, 0);
        return { todayTotal, removed };
    }

    /**
     * Get 7-day rolling history for Chart.js.
     * @returns {{ labels: string[], targets: number[], consumed: number[] }}
     */
    function get7DayHistory() {
        const logs = getAllLogs();
        const labels = [];
        const targets = [];
        const consumed = [];
        const dayNames = ['Min', 'Sen', 'Sel', 'Rab', 'Kam', 'Jum', 'Sab'];

        for (let i = 6; i >= 0; i--) {
            const d = new Date();
            d.setDate(d.getDate() - i);
            const dStr = `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
            const dayLabel = `${dayNames[d.getDay()]} ${d.getDate()}/${d.getMonth() + 1}`;

            const dayLog = logs.find(l => l.date === dStr);
            labels.push(dayLabel);
            targets.push(dayLog ? dayLog.target_ml : 0);
            consumed.push(dayLog ? dayLog.entries.reduce((s, e) => s + e.ml, 0) : 0);
        }

        return { labels, targets, consumed };
    }

    /**
     * Generate SVG water fill path for a given percentage.
     * Creates a smooth wave effect.
     * @param {number} percent - 0 to 100
     * @returns {string} SVG path d attribute
     */
    function generateWavePath(percent) {
        const clampedPct = Math.max(0, Math.min(100, percent));
        // Water level: 0% = bottom (y=200), 100% = top (y=20)
        const waterY = 200 - (clampedPct / 100) * 180;
        const waveHeight = Math.max(2, 8 - (clampedPct / 20)); // Wave dampens at higher fill
        return `M 0 ${waterY} 
                Q 50 ${waterY - waveHeight} 100 ${waterY} 
                Q 150 ${waterY + waveHeight} 200 ${waterY} 
                L 200 220 L 0 220 Z`;
    }

    // ── Public API ────────────────────────────────────────

    return {
        calculateDailyTarget,
        getHydrationStatus,
        getUrineAdvice,
        logIntake,
        getTodayTotal,
        undoLastIntake,
        get7DayHistory,
        generateWavePath,
        getAllLogs,
        URINE_FEEDBACK,
        MOOD_STATES,
        GLASS_ML
    };
})();
