# =============================================================================
# CropLogic — Disease Risk Model (Gradient Boosting Classifier)
# Section 4.1.5 of dissertation
# Uses scikit-learn GradientBoostingClassifier with isotonic calibration
# =============================================================================
import os, sys, joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import GBM_MODEL, SCALER_PATH, CROP_TYPE_MAP

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score
    SK_AVAILABLE = True
except ImportError:
    SK_AVAILABLE = False
    print("[GBM] scikit-learn not available")


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "ndvi",           # current field-averaged NDVI
    "ndvi_trend",     # β₁ slope over 4-week window
    "zscore",         # Z-score deviation from crop baseline
    "temperature_c",  # mean daily temperature (°C)
    "humidity_pct",   # mean daily relative humidity (%)
    "rain_7d_mm",     # 7-day cumulative rainfall (mm)
    "days_since_sow", # growth stage proxy
    # one-hot encoded crop type (5 categories)
    "is_cereal", "is_vegetable", "is_cash", "is_legume", "is_other",
]

def build_feature_vector(ndvi: float, beta1: float, zscore: float,
                          temp: float, humidity: float, rain_7d: float,
                          days_since_sow: int, crop: str) -> np.ndarray:
    """
    Build the 12-dimensional feature vector for GBM inference.
    Matches Table 4.1 in dissertation.
    """
    crop_type = CROP_TYPE_MAP.get(crop, "other")
    oh        = {t: int(crop_type == t) for t in ("cereal","vegetable","cash","legume")}
    oh["other"] = int(crop_type not in oh)

    return np.array([
        ndvi, beta1, zscore, temp, humidity, rain_7d, days_since_sow,
        oh["cereal"], oh["vegetable"], oh["cash"], oh["legume"], oh["other"],
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Training Data Generator
# Used when no labelled field dataset is available
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_dataset(n: int = 5000, random_state: int = 42) -> Tuple:
    """
    Generate synthetic field observations for initial GBM training.
    Labels are assigned using the expert rule-based logic from ndvi_analysis.py
    so the GBM learns to approximate the rules but can generalise beyond them.
    """
    rng = np.random.default_rng(random_state)

    ndvi          = rng.uniform(0.10, 0.95, n)
    beta1         = rng.uniform(-0.025, 0.015, n)
    zscore        = rng.normal(0, 1.2, n)
    temperature   = rng.uniform(18, 45, n)
    humidity      = rng.uniform(30, 98, n)
    rain_7d       = rng.exponential(20, n)
    days_since_sow= rng.integers(0, 180, n)
    crops         = rng.choice(list(CROP_TYPE_MAP.keys()), n)

    X = np.column_stack([
        ndvi, beta1, zscore, temperature, humidity, rain_7d, days_since_sow,
        *[np.array([int(CROP_TYPE_MAP.get(c,"other") == t) for c in crops])
          for t in ("cereal","vegetable","cash","legume","other")]
    ])

    # Rule-based label assignment (0=Low, 1=Medium, 2=High)
    y = np.zeros(n, dtype=int)
    for i in range(n):
        score = 0
        if ndvi[i] < 0.35:     score += 3
        elif ndvi[i] < 0.50:   score += 2
        elif ndvi[i] < 0.62:   score += 1
        if beta1[i] < -0.010:  score += 2
        elif beta1[i] < -0.004:score += 1
        if humidity[i] > 85:   score += 3
        elif humidity[i] > 75: score += 2
        elif humidity[i] > 65: score += 1
        if rain_7d[i] > 60:    score += 2
        elif rain_7d[i] > 25:  score += 1
        if temperature[i] > 38:score += 1
        # add label noise ±1 level
        noise = rng.choice([-1, 0, 0, 0, 1])
        if score <= 2:          y[i] = max(0, 0 + noise)
        elif score <= 5:        y[i] = max(0, min(2, 1 + noise))
        else:                   y[i] = max(0, min(2, 2 + noise))

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_risk_model(save_dir: str = None) -> "CalibratedClassifierCV":
    """
    Train and calibrate the Gradient Boosting risk classifier.
    Saves model + scaler to disk if save_dir is provided.
    """
    assert SK_AVAILABLE, "scikit-learn required"
    if save_dir is None:
        save_dir = os.path.dirname(GBM_MODEL)
    os.makedirs(save_dir, exist_ok=True)

    print("[GBM] Generating synthetic training dataset (n=5000)…")
    X, y = generate_synthetic_dataset(5000)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20,
                                                random_state=42, stratify=y)

    # Standardise features
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # GBM
    gbm = GradientBoostingClassifier(
        n_estimators    = 300,
        max_depth       = 4,
        learning_rate   = 0.05,
        subsample       = 0.8,
        min_samples_leaf= 10,
        random_state    = 42,
    )

    # Isotonic calibration
    cal_model = CalibratedClassifierCV(gbm, method="isotonic", cv=5)
    print("[GBM] Training calibrated GBM…")
    cal_model.fit(X_tr_s, y_tr)

    # Evaluate
    y_pred = cal_model.predict(X_te_s)
    print("[GBM] Classification report:")
    print(classification_report(y_te, y_pred,
                                  target_names=["Low","Medium","High"]))

    # Save
    joblib.dump(cal_model, os.path.join(save_dir, "risk_gbm.joblib"))
    joblib.dump(scaler,    os.path.join(save_dir, "scaler.joblib"))
    print(f"[GBM] ✓ Model saved to {save_dir}")
    return cal_model, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Inference Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class RiskModel:
    """
    Loads trained GBM + scaler from disk.
    Falls back to rule-based scoring if files are missing.
    """

    def __init__(self, model_path: str = GBM_MODEL, scaler_path: str = SCALER_PATH):
        self.model  = None
        self.scaler = None
        self.labels = ["Low", "Medium", "High"]

        if (model_path and os.path.exists(model_path) and
                scaler_path and os.path.exists(scaler_path) and SK_AVAILABLE):
            self.model  = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print("[GBM] ✓ Risk model loaded from disk")
        else:
            print("[GBM] No saved model found — using rule-based fallback")

    def predict(self, ndvi: float, beta1: float, zscore: float,
                temp: float, humidity: float, rain_7d: float,
                days_since_sow: int, crop: str) -> Dict:
        """
        Predict disease risk for a single field observation.
        Returns {"level": str, "probability": float, "proba_all": list}
        """
        fv = build_feature_vector(ndvi, beta1, zscore, temp, humidity,
                                   rain_7d, days_since_sow, crop)

        if self.model is not None and self.scaler is not None:
            fvs   = self.scaler.transform(fv.reshape(1, -1))
            proba = self.model.predict_proba(fvs)[0]
            idx   = int(np.argmax(proba))
            return {
                "level"      : self.labels[idx],
                "probability": round(float(proba[idx]), 3),
                "proba_all"  : {l: round(float(p), 3) for l, p in zip(self.labels, proba)},
            }
        else:
            # Rule-based fallback
            from modules.ndvi_analysis import compute_disease_risk
            rr = compute_disease_risk(ndvi, beta1, humidity, rain_7d, temp, crop)
            p  = rr["probability"]
            proba_map = {"Low": max(0, 1-p-0.1), "Medium": 0.1, "High": p} if p >= 0.6 else \
                        {"Low": max(0, 1-p), "Medium": p, "High": 0.0}
            return {
                "level"      : rr["level"],
                "probability": p,
                "proba_all"  : proba_map,
            }
