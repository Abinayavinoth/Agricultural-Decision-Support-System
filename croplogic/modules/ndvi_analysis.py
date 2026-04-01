# =============================================================================
# CropLogic — NDVI Analysis Module
# Implements:
#   • Z-score vegetation stress detection  (Eq. 3.5 / 3.6 in dissertation)
#   • Temporal trend β₁ via linear regression  (Eq. 3.3 / 3.4)
#   • Disease risk probability model  (Eq. 3.7)
# =============================================================================
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Any
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import CROP_BASELINES, CROP_TYPE_MAP


# ─────────────────────────────────────────────────────────────────────────────
# NDVI Core Formulas
# ─────────────────────────────────────────────────────────────────────────────

def compute_ndvi(nir: float, red: float) -> float:
    """
    NDVI = (NIR - RED) / (NIR + RED)
    Sentinel-2: NIR = Band 8 (842 nm), RED = Band 4 (665 nm)
    Returns float in [-1, 1], NaN if denominator == 0.
    """
    denom = nir + red
    if denom == 0:
        return float("nan")
    return float((nir - red) / denom)


def field_mean_ndvi(pixel_ndvis: np.ndarray) -> float:
    """
    Spatial average NDVI over a field boundary (Eq. 3.2):
        NDVI_field(t) = (1/|B|) * sum_{p in B} NDVI(p, t)
    Excludes NaN (cloudy) pixels.
    """
    valid = pixel_ndvis[~np.isnan(pixel_ndvis)]
    if len(valid) == 0:
        return float("nan")
    return float(np.mean(valid))


# ─────────────────────────────────────────────────────────────────────────────
# Z-score Stress Detection  (Section 3.3)
# ─────────────────────────────────────────────────────────────────────────────

def compute_zscore(ndvi: float, crop: str) -> Dict:
    """
    Z = (NDVI_t - mu_crop) / sigma_crop

    Returns dict with:
        zscore        — standardised deviation
        stress_level  — Healthy / Mild / Moderate / Severe Stress
        alert         — bool, True if |Z| > tau
        mu, sigma, tau — baseline parameters used
    """
    crop_type = CROP_TYPE_MAP.get(crop, "cereal")
    base      = CROP_BASELINES[crop_type]
    mu, sigma, tau = base["mu"], base["sigma"], base["tau"]

    z    = (ndvi - mu) / sigma
    az   = abs(z)
    alert = az > tau

    if az <= 1.0:   level = "Healthy"
    elif az <= tau: level = "Mild Stress"
    elif az <= 2.0: level = "Moderate Stress"
    else:           level = "Severe Stress"

    return {
        "zscore"      : round(z, 3),
        "stress_level": level,
        "alert"       : alert,
        "mu"          : mu,
        "sigma"       : sigma,
        "tau"         : tau,
    }


def batch_zscore(df: pd.DataFrame, crop: str) -> pd.DataFrame:
    """Apply Z-score analysis to entire NDVI time-series DataFrame."""
    results = df["ndvi"].apply(lambda v: pd.Series(compute_zscore(v, crop)))
    return pd.concat([df, results], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Trend β₁  (Section 3.2)
# ─────────────────────────────────────────────────────────────────────────────

def compute_trend(ndvi_series: np.ndarray, dates: np.ndarray = None,
                  window: int = 28) -> Dict:
    """
    Fits linear model NDVI(t) = β₀ + β₁·t + ε over a rolling window.

    Returns:
        beta1      — slope (NDVI units/day)
        pvalue     — two-tailed p-value for H₀: β₁ = 0
        trend_alert— True if β₁ < 0 and p < 0.05
        r_squared  — fit quality
    """
    n = len(ndvi_series)
    if n < 3:
        return {"beta1": 0.0, "pvalue": 1.0, "trend_alert": False, "r_squared": 0.0}

    use = ndvi_series[-min(window, n):]
    t   = np.arange(len(use), dtype=float)
    slope, intercept, r, p, se = stats.linregress(t, use)

    return {
        "beta1"       : round(float(slope), 6),
        "intercept"   : round(float(intercept), 4),
        "pvalue"      : round(float(p), 4),
        "r_squared"   : round(float(r ** 2), 4),
        "trend_alert" : bool(slope < 0 and p < 0.05),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Disease Risk Score  (Section 3.4 — Eq. 3.7)
# ─────────────────────────────────────────────────────────────────────────────

def compute_disease_risk(ndvi: float, beta1: float, humidity: float,
                          rain_7d: float, temperature: float,
                          crop: str) -> Dict:
    """
    P(risk | x) = f(NDVI, ∇NDVI, crop, weather)

    Rule-based Gradient Boosting proxy (used when trained GBM is unavailable).
    Each factor contributes a weighted partial score; total clipped to [0, 1].

    Returns:
        probability — float [0, 1]
        level       — "Low" / "Medium" / "High"
        factors     — per-factor contributions for explainability
    """
    factors = {}

    # NDVI level
    if ndvi < 0.35:       factors["ndvi"] = 0.30
    elif ndvi < 0.50:     factors["ndvi"] = 0.18
    elif ndvi < 0.60:     factors["ndvi"] = 0.08
    else:                 factors["ndvi"] = 0.0

    # NDVI trend direction
    if beta1 < -0.010:    factors["trend"] = 0.22
    elif beta1 < -0.005:  factors["trend"] = 0.12
    elif beta1 < 0:       factors["trend"] = 0.06
    else:                 factors["trend"] = 0.0

    # Humidity — high RH favours fungal pathogens
    if humidity > 85:     factors["humidity"] = 0.25
    elif humidity > 75:   factors["humidity"] = 0.14
    elif humidity > 65:   factors["humidity"] = 0.06
    else:                 factors["humidity"] = 0.0

    # Cumulative rainfall
    if rain_7d > 60:      factors["rain"] = 0.18
    elif rain_7d > 30:    factors["rain"] = 0.09
    elif rain_7d > 10:    factors["rain"] = 0.04
    else:                 factors["rain"] = 0.0

    # Heat stress above 38°C
    if temperature > 40:  factors["heat"] = 0.10
    elif temperature > 38:factors["heat"] = 0.05
    else:                 factors["heat"] = 0.0

    p     = float(np.clip(sum(factors.values()), 0, 1))

    if p < 0.30:   level = "Low"
    elif p < 0.60: level = "Medium"
    else:           level = "High"

    return {"probability": round(p, 3), "level": level, "factors": factors}


# ─────────────────────────────────────────────────────────────────────────────
# Full Field Analysis — combines all modules
# ─────────────────────────────────────────────────────────────────────────────

def analyse_field(field: Dict, ndvi_df: pd.DataFrame, weather: Dict = None) -> Dict:
    """
    Run complete single-field analysis pipeline.

    Args:
        field     — field metadata dict (name, crop, lat, lon, …)
        ndvi_df   — output of data_fetcher.build_ndvi_series()
        weather   — output of data_fetcher.fetch_weather() or None

    Returns:
        Comprehensive analysis dict consumed by the recommendation engine.
    """
    if ndvi_df.empty:
        return {"error": "No NDVI data available"}

    # Latest values
    latest   = ndvi_df.iloc[-1]
    ndvi_now = float(latest["ndvi"])
    crop     = field["crop"]

    # Z-score
    zr = compute_zscore(ndvi_now, crop)

    # Trend over available data
    tr = compute_trend(ndvi_df["ndvi"].values)

    # Weather defaults
    temp    = 28.0
    hum     = 70.0
    rain_7d = 15.0
    if weather:
        temp    = weather["current"].get("temperature_2m", 28)
        hum     = weather["current"].get("relative_humidity_2m", 70)
        rain_7d = sum(weather["daily"].get("precip_sum", [0] * 7) or [0])

    # Disease risk
    rr = compute_disease_risk(ndvi_now, tr["beta1"], hum, rain_7d, temp, crop)

    # Historical stats
    ndvi_arr = ndvi_df["ndvi"].values
    return {
        "field"       : field,
        "ndvi_now"    : ndvi_now,
        "ndvi_mean_30": round(float(np.mean(ndvi_arr[-30:])), 4),
        "ndvi_min_30" : round(float(np.min(ndvi_arr[-30:])), 4),
        "ndvi_max_30" : round(float(np.max(ndvi_arr[-30:])), 4),
        "zscore"      : zr,
        "trend"       : tr,
        "risk"        : rr,
        "weather"     : {
            "temperature_c" : temp,
            "humidity_pct"  : hum,
            "rain_7d_mm"    : rain_7d,
        },
    }
