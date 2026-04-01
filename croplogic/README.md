# 🌿 CropLogic — AI-Driven Smart Agriculture DSS

---

## Quick Start

### VS Code / Local
```bash
git clone <your-repo>
cd croplogic
pip install -r requirements.txt
streamlit run app.py
# → open http://localhost:8501
```

### Google Colab
```python
# 1. Upload croplogic.zip to Colab, then:
import zipfile
with zipfile.ZipFile('croplogic.zip') as z:
    z.extractall('.')
%cd croplogic

# 2. Install
!pip install -r requirements.txt

# 3. Open and run:  notebooks/CropLogic_Full_Pipeline.ipynb
```

### Kaggle Notebook
```python
# 1. Upload croplogic.zip as a dataset
# 2. In notebook:
import zipfile, os
with zipfile.ZipFile('/kaggle/input/croplogic/croplogic.zip') as z:
    z.extractall('/kaggle/working/')
os.chdir('/kaggle/working/croplogic')

!pip install -r requirements.txt
!streamlit run app.py --server.port 8501 &
```

---

## Project Structure

```
croplogic/
├── app.py                          # Streamlit dashboard (8 pages)
├── config.py                       # All hyperparameters & constants
├── requirements.txt
├── modules/
│   ├── data_fetcher.py             # NASA POWER + Open-Meteo APIs
│   ├── ndvi_analysis.py            # Z-score, trend β₁, risk model
│   ├── cnn_model.py                # ResNet-50 two-phase training + inference
│   ├── risk_model.py               # GBM risk classifier
│   └── recommendation_engine.py   # Rule-based recommendation logic
├── notebooks/
│   └── CropLogic_Full_Pipeline.ipynb
├── models/                         # Saved after training
│   ├── resnet50_plantvillage.h5
│   ├── risk_gbm.joblib
│   ├── scaler.joblib
│   └── class_indices.json
├── data/
│   └── plantvillage/               # Download from Kaggle (see below)
└── outputs/                        # Charts, reports
```

---

## Free Data Sources (No API Key Required)

| Source | URL | Used For |
|--------|-----|----------|
| **NASA POWER** | power.larc.nasa.gov | Solar radiation → NDVI proxy |
| **Open-Meteo** | open-meteo.com | Real-time weather + 7-day forecast |
| **PlantVillage** | kaggle.com/vipoooool | CNN disease training (54,306 images) |

---

## Download PlantVillage Dataset

```bash
# Option 1: Kaggle CLI
pip install kaggle
# Put kaggle.json in ~/.kaggle/
kaggle datasets download -d vipoooool/new-plant-diseases-dataset
unzip new-plant-diseases-dataset.zip -d data/plantvillage

# Option 2: opendatasets (prompts for Kaggle credentials)
python -c "
import opendatasets as od
od.download('https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset',
            data_dir='data/plantvillage')
"
```

Expected directory after extraction:
```
data/plantvillage/New Plant Diseases Dataset(Augmented)/
├── train/   (43,456 images, 38 class folders)
├── valid/   (10,849 images)
└── test/    (optional)
```

---

## Training the CNN

```python
# In Python / notebook:
from modules.cnn_model import train
model, history1, history2 = train(
    data_dir = 'data/plantvillage/New Plant Diseases Dataset(Augmented)',
    save_dir = 'models'
)
```

Or use the **🎓 Train CNN Model** page in the Streamlit dashboard.

**Training phases:**
| Phase | Epochs | LR | Layers |
|-------|--------|----|--------|
| 1 — Head only | 20 | 1e-3 | ResNet-50 frozen, head trained |
| 2 — Fine-tune | ≤30 (early stop, patience=5) | 1e-5 | Top 30 layers unfrozen |

Expected accuracy on PlantVillage test set: **≥93% Top-1**

---

## Mathematical Models

### NDVI (Eq. 3.1)
```
NDVI = (NIR - RED) / (NIR + RED)
```

### Z-score Stress Detection (Eq. 3.5 / 3.6)
```
Z = (NDVI_t - μ_crop) / σ_crop
|Z| > τ  →  Vegetation Stress Alert
τ = 1.5 (cereals), τ = 1.8 (legumes)
```

### Temporal Trend (Eq. 3.3 / 3.4)
```
NDVI(t_k) = β₀ + β₁·t_k + ε_k
β₁ < 0 and p < 0.05  →  Trend Alert
```

### Disease Risk (Eq. 3.7)
```
P(risk | x) = f(NDVI, ∇NDVI, crop, weather)
Low: P < 0.3  |  Medium: 0.3 ≤ P < 0.6  |  High: P ≥ 0.6
```

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| 📊 Dashboard | KPIs, field status table, NDVI trend chart |
| 🗺️ Field Map | Interactive Folium map, colour-coded by health |
| 📈 NDVI Analysis | Time-series, Z-score, trend β₁, math formulas |
| 🔬 Disease Scanner | ResNet-50 leaf image classifier (upload any leaf photo) |
| 🌤️ Weather Risk | Open-Meteo live data, 7-day forecast, disease risk matrix |
| 💡 Recommendations | Priority-sorted agronomic interventions |
| ➕ Register Field | Add new fields with live NDVI fetch |
| 🎓 Train CNN Model | Launch training + view accuracy/loss curves |

---

## Recommendation Logic (Table 4.2)

| Condition | Priority | Action |
|-----------|----------|--------|
| Severe stress + High risk | CRITICAL | Immediate irrigation + fungicide + expert |
| Moderate stress / Medium risk | HIGH | Increase irrigation, scout 48h |
| Mild stress | MEDIUM | Monitor NDVI, check soil moisture |
| Healthy + Low risk | LOW | Continue management |
| CNN disease detected | HIGH | Targeted treatment |
| Humidity > 80% | MEDIUM | Preventive fungicide |
| Temp > 38°C | HIGH | Foliar spray, irrigation |

---

*Built with TensorFlow 2.15 · scikit-learn 1.4 · Streamlit 1.32 · NASA POWER · Open-Meteo*
