/**
 * Rule verification for the Clinical Rules Engine.
 *
 * Loads frontend/js/clinical-rules.js, runs a set of clinically meaningful
 * profile x food scenarios, and compares the engine output against the
 * documented specification (Table 4 limits and Table 5 status thresholds).
 *
 * This verifies that the *implementation* matches the *specification*. It is
 * NOT a clinical validation (no dietitian reviewed the rules).
 *
 * Usage: node experiments/scripts/verify_clinical_rules.js
 */
const fs = require("fs");
const vm = require("vm");
const path = require("path");

const ROOT = path.resolve(__dirname, "..", "..");
const RULES = path.join(ROOT, "frontend", "js", "clinical-rules.js");
const OUT = path.join(ROOT, "experiments", "results", "clinical_rules");

const sandbox = { window: {} };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(RULES, "utf8"), sandbox);
const engine = sandbox.window.ClinicalHealthRules;

// ---- Independent specification model (from Table 4 / Table 5) -------------
function bmr(profile) {
  const base = 10 * profile.weight_kg + 6.25 * profile.height_cm - 5 * profile.age;
  return profile.gender === "male" ? base + 5 : base - 161;
}
function specLimits(p) {
  const tdee = p.tdee;
  let sugar = (tdee * 0.10) / 4;
  let sodium = 2300;
  let satfat = (tdee * 0.10) / 9;
  let cholesterol = null;
  let protein = null;
  let sodiumWarn = 800;
  if (p.isDiabetic) {
    sugar = (tdee * 0.05) / 4;
    if (p.hasDyslipidemia) {
      satfat = (tdee * 0.07) / 9;
      cholesterol = 200;
    }
    if (p.hasNephropathy) protein = p.weight_kg * 0.8;
  }
  if (p.isHypertensive) {
    sodium = 1500;
    sodiumWarn = 600;
  }
  return {
    max_sugar_g: sugar,
    max_sodium_mg: sodium,
    max_saturated_fat_g: satfat,
    max_cholesterol_mg: cholesterol,
    max_protein_g: protein,
    sodium_warning_threshold: sodiumWarn,
  };
}
function specStatus(p, L, scanned, serving) {
  const mult = serving / 100;
  const sugar = (scanned.sugar_g || 0) * mult;
  const sodium = (scanned.sodium_mg || 0) * mult;
  const satfat = (scanned.saturated_fat_g || 0) * mult;
  const protein = (scanned.protein_g || 0) * mult;
  let status = "SAFE";
  const bump = (s) => { if (s === "DANGER" || status === "DANGER") status = "DANGER"; else status = "WARNING"; };
  if (sugar > L.max_sugar_g) status = "DANGER";
  else if (sugar > L.max_sugar_g * 0.5) bump("WARNING");
  if (p.isDiabetic && p.hasNephropathy && L.max_protein_g && protein > L.max_protein_g * 0.3) bump("WARNING");
  if (p.isHypertensive) {
    if ((scanned.sodium_mg || 0) > L.sodium_warning_threshold) bump("WARNING");
    if (sodium > L.max_sodium_mg * 0.4) bump("WARNING");
  }
  if ((p.hasDyslipidemia || p.isDiabetic) && satfat > L.max_saturated_fat_g * 0.5) bump("WARNING");
  return status;
}

// ---- Scenarios ------------------------------------------------------------
const profiles = {
  P1_general: { age: 30, gender: "male", weight_kg: 70, height_cm: 170, isDiabetic: false, hasNephropathy: false, hasDyslipidemia: false, isHypertensive: false },
  P2_diabetes: { age: 45, gender: "female", weight_kg: 65, height_cm: 158, isDiabetic: true, hasNephropathy: false, hasDyslipidemia: false, isHypertensive: false },
  P3_hypertension: { age: 55, gender: "male", weight_kg: 75, height_cm: 168, isDiabetic: false, hasNephropathy: false, hasDyslipidemia: false, isHypertensive: true },
  P4_diabetes_dyslipidemia: { age: 50, gender: "female", weight_kg: 68, height_cm: 160, isDiabetic: true, hasNephropathy: false, hasDyslipidemia: true, isHypertensive: false },
  P5_diabetes_nephropathy: { age: 60, gender: "male", weight_kg: 60, height_cm: 165, isDiabetic: true, hasNephropathy: true, hasDyslipidemia: false, isHypertensive: false },
  P6_hypertension_dyslipidemia: { age: 58, gender: "female", weight_kg: 72, height_cm: 155, isDiabetic: false, hasNephropathy: false, hasDyslipidemia: true, isHypertensive: true },
};
// scanned values are per 100 g; serving in grams
const foods = [
  { name: "F1_low", serving: 50, sugar_g: 5, sodium_mg: 100, saturated_fat_g: 1, protein_g: 1, fiber_g: 1, energy_kj: 400 },
  { name: "F2_sugary", serving: 50, sugar_g: 60, sodium_mg: 200, saturated_fat_g: 4, protein_g: 3, fiber_g: 1, energy_kj: 1900 },
  { name: "F3_salty", serving: 100, sugar_g: 4, sodium_mg: 900, saturated_fat_g: 2, protein_g: 6, fiber_g: 1, energy_kj: 1000 },
  { name: "F4_fatty", serving: 80, sugar_g: 10, sodium_mg: 300, saturated_fat_g: 20, protein_g: 5, fiber_g: 1, energy_kj: 2200 },
  { name: "F5_protein", serving: 150, sugar_g: 2, sodium_mg: 250, saturated_fat_g: 3, protein_g: 25, fiber_g: 0, energy_kj: 800 },
];

const rows = [];
let limitMatch = 0, limitTotal = 0, statusMatch = 0, statusTotal = 0;

for (const [pname, p0] of Object.entries(profiles)) {
  const p = { ...p0 };
  p.bmr = bmr(p);
  p.tdee = p.bmr * 1.2;
  const specL = specLimits(p);
  const rules = engine.create(p);
  const engL = rules.limits;
  for (const key of Object.keys(specL)) {
    limitTotal++;
    const a = engL[key];
    const b = specL[key];
    const same = a === null || b === null ? a === b : Math.abs(a - b) < 1e-6;
    if (same) limitMatch++;
    else rows.push({ type: "limit", profile: pname, field: key, engine: a, spec: b, match: false });
  }
  for (const food of foods) {
    const scanned = { sugar_g: food.sugar_g, sodium_mg: food.sodium_mg, saturated_fat_g: food.saturated_fat_g, protein_g: food.protein_g, fiber_g: food.fiber_g, energy_kj: food.energy_kj };
    const res = rules.evaluateMeal(scanned, food.serving);
    const exp = specStatus(p, specL, scanned, food.serving);
    statusTotal++;
    const ok = res.status === exp;
    if (ok) statusMatch++;
    rows.push({
      type: "status",
      profile: pname,
      food: food.name,
      tdee: Math.round(p.tdee),
      engine: res.status,
      spec: exp,
      match: ok,
    });
  }
}

fs.mkdirSync(OUT, { recursive: true });
fs.writeFileSync(path.join(OUT, "verification.json"), JSON.stringify({ limitMatch, limitTotal, statusMatch, statusTotal, rows }, null, 2));

const mismatches = rows.filter((r) => !r.match);
const lines = [
  "# Clinical Rules Engine - rule verification",
  "",
  `Specs matched: limits **${limitMatch}/${limitTotal}**, status decisions **${statusMatch}/${statusTotal}**.`,
  "",
  "Implementation (frontend/js/clinical-rules.js) versus the documented specification (Table 4 / Table 5).",
  "This verifies the implementation, not clinical efficacy.",
  "",
  "| Profile | Food | TDEE (kcal) | Engine | Spec | Match |",
  "| :--- | :--- | ---: | :--- | :--- | :--- |",
];
for (const r of rows.filter((x) => x.type === "status")) {
  lines.push(`| ${r.profile} | ${r.food} | ${r.tdee} | ${r.engine} | ${r.spec} | ${r.match ? "yes" : "NO"} |`);
}
lines.push("");
fs.writeFileSync(path.join(OUT, "verification.md"), lines.join("\n") + "\n", "utf8");

console.log(lines.slice(0, 6).join("\n"));
console.log("mismatches:", mismatches.length);
if (mismatches.length) console.log(JSON.stringify(mismatches, null, 2));
