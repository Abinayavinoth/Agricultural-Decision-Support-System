# =============================================================================
# CropLogic — Data Fetcher
# Pulls real-time data from FREE APIs:
#   • NASA POWER  → solar radiation → NDVI proxy
#   • Open-Meteo  → weather (no API key needed)
# =============================================================================
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import NASA_POWER_URL, OPEN_METEO_URL, CROP_BASELINES, CROP_TYPE_MAP


# ─────────────────────────────────────────────────────────────────────────────
# NASA POWER API  (completely free, no key)
# Docs: https://power.larc.nasa.gov/docs/
# ─────────────────────────────────────────────────────────────────────────────

def fetch_nasa_power(lat: float, lon: float, days: int = 30) -> Optional[Dict]:
    """
    Fetch daily solar radiation + temperature + precipitation from NASA POWER.
    Returns dict of {parameter: {YYYYMMDD: value}} or None on failure.

    Parameters used:
        ALLSKY_SFC_SW_DWN  — all-sky surface shortwave downward irradiance (MJ/m²/day)
        T2M                — temperature at 2m (°C)
        PRECTOTCORR        — corrected total precipitation (mm/day)
        RH2M               — relative humidity at 2m (%)
    """
    end   = datetime.utcnow()
    start = end - timedelta(days=days)

    params = {
        "parameters" : "ALLSKY_SFC_SW_DWN,T2M,PRECTOTCORR,RH2M",
        "community"  : "AG",
        "longitude"  : lon,
        "latitude"   : lat,
        "start"      : start.strftime("%Y%m%d"),
        "end"        : end.strftime("%Y%m%d"),
        "format"     : "JSON",
    }
    try:
        r = requests.get(NASA_POWER_URL, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        props = j.get("properties", {}).get("parameter", {})
        if props:
            print(f"[NASA POWER] ✓ {lat},{lon} — {len(list(props.get('T2M',{}).keys()))} days")
            return props
    except Exception as e:
        print(f"[NASA POWER] ✗ {e}")
    return None


def solar_to_ndvi(rad: float, mu: float, sigma: float, seed_offset: float = 0.0) -> float:
    """
    Map solar irradiance (MJ/m²/day, range 0–25) to an NDVI value.

    Physics rationale:
        Healthy vegetation shows high PAR absorption → strong NDVI.
        High solar radiation during growing season correlates positively.
        We fit a sigmoid and add Gaussian noise to simulate pixel variability.
    """
    if rad <= 0:
        return float(np.clip(mu + np.random.normal(0, sigma * 0.5), 0.05, 0.95))
    norm   = np.clip(rad / 22.0, 0, 1)           # normalise to [0,1]
    sig    = 1 / (1 + np.exp(-8 * (norm - 0.45))) # sigmoid centred at 0.45
    ndvi   = mu - 0.06 + sig * 0.22 + seed_offset
    noise  = np.random.normal(0, sigma * 0.25)
    return float(np.clip(ndvi + noise, 0.05, 0.95))


def build_ndvi_series(field: Dict, days: int = 30) -> pd.DataFrame:
    """
    Build a daily NDVI time-series for a field using NASA POWER data.
    Falls back to a synthetic AR(1) process if the API is unavailable.

    Returns DataFrame with columns: date, ndvi, zscore, stress_level
    """
    crop_type = CROP_TYPE_MAP.get(field["crop"], "cereal")
    base      = CROP_BASELINES[crop_type]
    mu, sigma, tau = base["mu"], base["sigma"], base["tau"]

    nasa = fetch_nasa_power(field["lat"], field["lon"], days)

    records = []
    np.random.seed(abs(hash(field["name"])) % (2**31))  # deterministic per field

    if nasa and "ALLSKY_SFC_SW_DWN" in nasa:
        rad_data  = nasa["ALLSKY_SFC_SW_DWN"]
        t2m_data  = nasa.get("T2M", {})
        prec_data = nasa.get("PRECTOTCORR", {})
        rh_data   = nasa.get("RH2M", {})

        for key in sorted(rad_data.keys()):
            try:
                d    = datetime.strptime(key, "%Y%m%d")
                rad  = rad_data[key]
                ndvi = solar_to_ndvi(rad, mu, sigma)
                z    = (ndvi - mu) / sigma
                records.append({
                    "date"       : d,
                    "ndvi"       : round(ndvi, 4),
                    "zscore"     : round(z, 3),
                    "rad_mj"     : round(rad, 2) if rad > 0 else None,
                    "temp_c"     : t2m_data.get(key),
                    "precip_mm"  : prec_data.get(key),
                    "humidity_pct": rh_data.get(key),
                    "stress_level": _classify_stress(z, tau),
                    "source"     : "NASA POWER",
                })
            except Exception:
                pass
    else:
        # AR(1) synthetic fallback
        print(f"[NDVI] Using synthetic AR(1) baseline for {field['name']}")
        ndvi_prev = mu
        for i in range(days, -1, -1):
            d      = datetime.utcnow() - timedelta(days=i)
            phi    = 0.85                                   # AR(1) coefficient
            eps    = np.random.normal(0, sigma * 0.3)
            trend  = -0.001 * i                             # slight decline toward present
            ndvi   = phi * ndvi_prev + (1 - phi) * mu + eps + trend
            ndvi   = float(np.clip(ndvi, 0.05, 0.95))
            z      = (ndvi - mu) / sigma
            records.append({
                "date"        : d,
                "ndvi"        : round(ndvi, 4),
                "zscore"      : round(z, 3),
                "rad_mj"      : None,
                "temp_c"      : None,
                "precip_mm"   : None,
                "humidity_pct": None,
                "stress_level": _classify_stress(z, tau),
                "source"      : "Synthetic AR(1)",
            })
            ndvi_prev = ndvi

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Add 4-week rolling linear trend β₁
    if len(df) >= 4:
        df["beta1"] = _rolling_slope(df["ndvi"].values, window=28)
    else:
        df["beta1"] = 0.0

    df["mu"]    = mu
    df["sigma"] = sigma
    df["tau"]   = tau
    return df


def _classify_stress(z: float, tau: float) -> str:
    az = abs(z)
    if az <= 1.0:         return "Healthy"
    if az <= tau:         return "Mild Stress"
    if az <= 2.0:         return "Moderate Stress"
    return "Severe Stress"


def _rolling_slope(arr: np.ndarray, window: int = 28) -> np.ndarray:
    """Compute rolling linear slope β₁ using OLS."""
    slopes = np.zeros(len(arr))
    x      = np.arange(window, dtype=float)
    for i in range(len(arr)):
        if i < window - 1:
            slopes[i] = 0.0
        else:
            y         = arr[i - window + 1: i + 1].astype(float)
            slopes[i] = float(np.polyfit(x, y, 1)[0])
    return slopes


# ─────────────────────────────────────────────────────────────────────────────
# Open-Meteo API  (completely free, no key)
# Docs: https://open-meteo.com/en/docs
# ─────────────────────────────────────────────────────────────────────────────

def fetch_weather(lat: float, lon: float) -> Optional[Dict]:
    """
    Fetch current conditions + 7-day daily forecast from Open-Meteo.

    Returns dict with keys:
        current  — {temperature_2m, relative_humidity_2m, precipitation,
                    wind_speed_10m, cloud_cover}
        daily    — {time, temp_max, temp_min, precip_sum, humidity_max}
    """
    params = {
        "latitude"  : lat,
        "longitude" : lon,
        "current"   : "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,cloud_cover",
        "daily"     : "temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_max",
        "timezone"  : "auto",
        "forecast_days": 7,
    }
    try:
        r = requests.get(OPEN_METEO_URL, params=params, timeout=20)
        r.raise_for_status()
        j   = r.json()
        cur = j.get("current", {})
        day = j.get("daily", {})
        print(f"[Open-Meteo] ✓ {lat},{lon} — {cur.get('temperature_2m')}°C, "
              f"{cur.get('relative_humidity_2m')}% RH")
        return {
            "current": {
                "temperature_2m"      : cur.get("temperature_2m", 28),
                "relative_humidity_2m": cur.get("relative_humidity_2m", 70),
                "precipitation"       : cur.get("precipitation", 0),
                "wind_speed_10m"      : cur.get("wind_speed_10m", 10),
                "cloud_cover"         : cur.get("cloud_cover", 30),
            },
            "daily": {
                "time"        : day.get("time", []),
                "temp_max"    : day.get("temperature_2m_max", []),
                "temp_min"    : day.get("temperature_2m_min", []),
                "precip_sum"  : day.get("precipitation_sum", []),
                "humidity_max": day.get("relative_humidity_2m_max", []),
            },
        }
    except Exception as e:
        print(f"[Open-Meteo] ✗ {e}")
        return None
