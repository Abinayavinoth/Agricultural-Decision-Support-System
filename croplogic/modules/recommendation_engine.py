# =============================================================================
# CropLogic — Recommendation Engine
# Table 4.2 in dissertation: maps stress × risk → prioritised agronomic advice
# =============================================================================
from typing import List, Dict, Any
from datetime import datetime


# Priority ordering
PRIORITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}

# ─────────────────────────────────────────────────────────────────────────────
# Core rule table (Table 4.2)
# ─────────────────────────────────────────────────────────────────────────────

def generate_recommendations(analysis: Dict, disease_result: Dict = None) -> List[Dict]:
    """
    Generate prioritised recommendations from field analysis output.

    Args:
        analysis       — output of ndvi_analysis.analyse_field()
        disease_result — optional CNN classifier result dict

    Returns:
        Sorted list of recommendation dicts with keys:
            priority, title, description, action_window, icon
    """
    recs = []

    stress = analysis["zscore"]["stress_level"]
    risk   = analysis["risk"]["level"]
    ndvi   = analysis["ndvi_now"]
    beta1  = analysis["trend"]["beta1"]
    temp   = analysis["weather"]["temperature_c"]
    hum    = analysis["weather"]["humidity_pct"]
    rain   = analysis["weather"]["rain_7d_mm"]
    crop   = analysis["field"]["crop"]

    # ── Rule 1: Severe stress + High risk → CRITICAL ──────────────────────────
    if stress == "Severe Stress" and risk == "High":
        recs.append(_rec(
            "CRITICAL", "🚨",
            "Immediate Irrigation + Fungicide Application",
            f"NDVI={ndvi} is severely below baseline (Z={analysis['zscore']['zscore']}). "
            f"Combine emergency irrigation with a targeted fungicide. "
            f"Scout {crop} field within 24h. Consult agronomist immediately.",
            "Within 24 hours"
        ))

    # ── Rule 2: Moderate stress OR Medium risk ─────────────────────────────────
    elif stress in ("Moderate Stress",) or risk == "Medium":
        recs.append(_rec(
            "HIGH", "⚠️",
            "Increase Irrigation Frequency",
            f"Moderate vegetation stress detected (Z={analysis['zscore']['zscore']}). "
            f"Increase irrigation by 25%. Scout {crop} field within 48 hours. "
            f"Monitor NDVI trend — current slope β₁={beta1:.5f}/day.",
            "Within 48 hours"
        ))

    # ── Rule 3: Mild stress ────────────────────────────────────────────────────
    elif stress == "Mild Stress":
        recs.append(_rec(
            "MEDIUM", "📋",
            "Monitor NDVI & Check Soil Moisture",
            f"Mild stress signal (Z={analysis['zscore']['zscore']}). "
            f"Verify soil moisture levels. Continue current schedule but check drainage. "
            f"Next satellite observation due in ≤5 days.",
            "Within 1 week"
        ))

    # ── Rule 4: Healthy ────────────────────────────────────────────────────────
    else:
        recs.append(_rec(
            "LOW", "✅",
            "Continue Current Crop Management",
            f"{crop} shows healthy NDVI ({ndvi}). "
            f"No immediate intervention required. "
            f"Next routine check in 5 days (satellite revisit cycle).",
            "Routine — 5 days"
        ))

    # ── Conditional rules ────────────────────────────────────────────────────

    # High humidity → fungal alert
    if hum > 80:
        recs.append(_rec(
            "MEDIUM", "💧",
            f"High Humidity — Fungal Disease Alert ({hum:.0f}%)",
            f"Relative humidity {hum:.0f}% exceeds 80% threshold. "
            f"Conditions strongly favour fungal pathogens (Phytophthora, Botrytis, Alternaria). "
            f"Apply preventive copper-based or systemic fungicide for {crop}.",
            "Within 48 hours"
        ))

    # Heat stress
    if temp > 38:
        recs.append(_rec(
            "HIGH", "🌡️",
            f"Heat Stress Management — {temp:.1f}°C",
            f"Temperature {temp:.1f}°C exceeds 38°C heat-stress threshold. "
            f"Apply foliar potassium silicate spray. Increase irrigation to cool root zone. "
            f"Avoid nitrogen fertiliser during peak heat. Schedule operations before 09:00.",
            "Within 24 hours"
        ))

    # Declining NDVI trend
    if beta1 < -0.005 and analysis["trend"]["trend_alert"]:
        recs.append(_rec(
            "MEDIUM", "📉",
            f"Declining NDVI Trend (β₁={beta1:.5f}/day, p<0.05)",
            f"Statistically significant downward NDVI trend detected. "
            f"Investigate nutrient deficiency (N, Fe, Mg) or early-stage blight. "
            f"Recommend soil test and visual scouting within 3 days.",
            "Within 3 days"
        ))

    # Drought risk
    if rain < 5.0:
        recs.append(_rec(
            "MEDIUM", "☀️",
            "Drought Stress Risk — 7-day Rainfall <5mm",
            f"Only {rain:.1f}mm cumulative rainfall in past 7 days. "
            f"Ensure irrigation schedule compensates. "
            f"Check soil moisture sensors or probe at 15cm depth. "
            f"Consider mulching to reduce evaporation.",
            "Within 2 days"
        ))

    # Waterlogging risk
    if rain > 80:
        recs.append(_rec(
            "MEDIUM", "🌊",
            f"Waterlogging Risk — {rain:.0f}mm in 7 days",
            f"Heavy rainfall ({rain:.0f}mm/7d) may cause waterlogging and root asphyxiation. "
            f"Open drainage channels. Avoid field machinery to prevent compaction. "
            f"Monitor for yellowing / wilting in low-lying zones.",
            "Immediate"
        ))

    # CNN disease result
    if disease_result and disease_result.get("top1"):
        top = disease_result["top1"]
        if not top.get("healthy") and top.get("confidence", 0) > 0.55:
            cls  = top["class"].replace("___", " — ").replace("_", " ")
            conf = top["confidence"] * 100
            recs.append(_rec(
                "HIGH", "🔬",
                f"Disease Identified: {cls} ({conf:.0f}% confidence)",
                f"CNN classifier detected {cls} with {conf:.0f}% confidence. "
                f"Apply targeted treatment: see disease management protocol for {crop}. "
                f"Remove and destroy heavily infected plant material.",
                "Within 24 hours"
            ))

    # Sort by priority
    recs.sort(key=lambda r: PRIORITY_ORDER.get(r["priority"], 99))
    return recs


def _rec(priority: str, icon: str, title: str,
          description: str, action_window: str) -> Dict:
    return {
        "priority"      : priority,
        "icon"          : icon,
        "title"         : title,
        "description"   : description,
        "action_window" : action_window,
        "generated_at"  : datetime.utcnow().isoformat(),
    }


def format_report(field_name: str, recs: List[Dict]) -> str:
    """Plain-text formatted report for terminal / notebook output."""
    lines = [
        f"{'='*70}",
        f"  CropLogic Recommendations — {field_name}",
        f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"{'='*70}",
    ]
    for i, r in enumerate(recs, 1):
        lines += [
            f"\n[{i}] {r['icon']}  [{r['priority']}] {r['title']}",
            f"     Action window : {r['action_window']}",
            f"     {r['description']}",
        ]
    lines.append(f"\n{'='*70}\n")
    return "\n".join(lines)
