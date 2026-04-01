# =============================================================================
# CropLogic — Configuration
# =============================================================================
import os

# ── Free API endpoints (no key needed) ───────────────────────────────────────
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# ── OpenWeatherMap (optional — free tier, 1000 calls/day) ────────────────────
# Sign up at https://openweathermap.org/api → set as env var or paste here
OWM_API_KEY = os.getenv("OWM_API_KEY", "")  # leave "" to use Open-Meteo instead

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "models")
CNN_MODEL   = os.path.join(MODEL_DIR, "resnet50_plantvillage.h5")
GBM_MODEL   = os.path.join(MODEL_DIR, "risk_gbm.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

# ── Dataset ───────────────────────────────────────────────────────────────────
DATA_DIR       = os.path.join(os.path.dirname(__file__), "data")
PLANTVILLAGE_DIR = os.path.join(DATA_DIR, "plantvillage")

# ── CNN hyperparameters ───────────────────────────────────────────────────────
IMG_SIZE       = (224, 224)
BATCH_SIZE     = 32
PHASE1_EPOCHS  = 20
PHASE2_EPOCHS  = 30
PHASE1_LR      = 1e-3
PHASE2_LR      = 1e-5
DROPOUT_RATE   = 0.5
UNFREEZE_LAYERS = 30   # top N ResNet-50 layers to unfreeze in phase 2

# ── NDVI / Z-score thresholds ─────────────────────────────────────────────────
CROP_BASELINES = {
    # type : (mu, sigma, tau)
    "cereal"   : {"mu": 0.68, "sigma": 0.09, "tau": 1.5},
    "vegetable": {"mu": 0.72, "sigma": 0.10, "tau": 1.5},
    "cash"     : {"mu": 0.65, "sigma": 0.11, "tau": 1.8},
    "legume"   : {"mu": 0.63, "sigma": 0.08, "tau": 1.8},
}

CROP_TYPE_MAP = {
    "Rice (Paddy)": "cereal",
    "Wheat"       : "cereal",
    "Maize"       : "cereal",
    "Millets"     : "cereal",
    "Tomato"      : "vegetable",
    "Potato"      : "vegetable",
    "Pepper"      : "vegetable",
    "Sugarcane"   : "cash",
    "Cotton"      : "cash",
    "Soybean"     : "legume",
    "Groundnut"   : "legume",
}

NDVI_TREND_WINDOW = 4   # weeks for linear β₁ estimation
CLOUD_MASK_PCT    = 20  # max cloud % for Sentinel-2 image

# ── PlantVillage classes (38) ─────────────────────────────────────────────────
DISEASE_CLASSES = [
    "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy",
    "Blueberry___healthy","Cherry___Powdery_mildew","Cherry___healthy",
    "Corn___Cercospora_leaf_spot","Corn___Common_rust","Corn___Northern_Leaf_Blight","Corn___healthy",
    "Grape___Black_rot","Grape___Esca_(Black_Measles)","Grape___Leaf_blight","Grape___healthy",
    "Orange___Haunglongbing","Peach___Bacterial_spot","Peach___healthy",
    "Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy",
    "Potato___Early_blight","Potato___Late_blight","Potato___healthy",
    "Raspberry___healthy","Soybean___healthy","Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch","Strawberry___healthy",
    "Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___Late_blight",
    "Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites","Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus","Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

NUM_CLASSES = len(DISEASE_CLASSES)   # 38

# ── Demo fields (for Streamlit / notebook demo) ───────────────────────────────
DEMO_FIELDS = [
    {"id":1,"name":"Kaveri Rice Block","crop":"Rice (Paddy)","lat":10.79,"lon":77.00,"area":3.2,"sow":"2024-07-01"},
    {"id":2,"name":"South Tomato Plot","crop":"Tomato",       "lat":13.08,"lon":80.27,"area":1.5,"sow":"2024-09-15"},
    {"id":3,"name":"Krishna Wheat Farm","crop":"Wheat",       "lat":17.38,"lon":78.49,"area":4.8,"sow":"2024-11-01"},
    {"id":4,"name":"Sugarcane West",    "crop":"Sugarcane",   "lat":18.52,"lon":73.85,"area":6.1,"sow":"2024-03-01"},
    {"id":5,"name":"Cotton Delta",      "crop":"Cotton",      "lat":16.51,"lon":80.61,"area":5.5,"sow":"2024-06-15"},
    {"id":6,"name":"Maize Hills Block", "crop":"Maize",       "lat":11.00,"lon":76.96,"area":2.1,"sow":"2024-08-01"},
]
