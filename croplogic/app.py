# =============================================================================
# CropLogic — Streamlit Dashboard (app.py)
# Run: streamlit run app.py
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from config import DEMO_FIELDS, CROP_TYPE_MAP, CROP_BASELINES, DISEASE_CLASSES
from modules.data_fetcher import build_ndvi_series, fetch_weather
from modules.ndvi_analysis import analyse_field, compute_zscore
from modules.recommendation_engine import generate_recommendations, format_report
from modules.cnn_model import DiseaseClassifier
from modules.risk_model import RiskModel

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "CropLogic — AI Agri DSS",
    page_icon   = "🌿",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Styles
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .metric-card {
    background: #1e2329; border: 1px solid #2d333b; border-radius: 10px;
    padding: 14px 18px; margin-bottom: 8px;
  }
  .metric-label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
  .metric-value { font-size: 28px; font-weight: 700; margin: 4px 0; }
  .badge-critical { background: rgba(248,81,73,0.2); color: #f85149; border: 1px solid rgba(248,81,73,0.4); padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; }
  .badge-high     { background: rgba(219,109,40,0.2); color: #db6d28; border: 1px solid rgba(219,109,40,0.4); padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; }
  .badge-medium   { background: rgba(210,153,34,0.2); color: #d29922; border: 1px solid rgba(210,153,34,0.4); padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; }
  .badge-low      { background: rgba(63,185,80,0.2);  color: #3fb950; border: 1px solid rgba(63,185,80,0.4);  padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; }
  .rec-box { background: #1e2329; border-left: 4px solid; border-radius: 6px; padding: 14px 16px; margin-bottom: 10px; }
  .rec-critical { border-color: #f85149; }
  .rec-high     { border-color: #db6d28; }
  .rec-medium   { border-color: #d29922; }
  .rec-low      { border-color: #3fb950; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "fields" not in st.session_state:
    st.session_state.fields = DEMO_FIELDS.copy()
if "ndvi_cache" not in st.session_state:
    st.session_state.ndvi_cache = {}
if "wx_cache" not in st.session_state:
    st.session_state.wx_cache = {}

# ─────────────────────────────────────────────────────────────────────────────
# Cached data loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_ndvi(field_id, lat, lon, crop, name, days=30):
    field = next(f for f in st.session_state.fields if f["id"] == field_id)
    return build_ndvi_series(field, days)

@st.cache_data(ttl=1800, show_spinner=False)
def get_weather(lat, lon):
    return fetch_weather(lat, lon)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/plant-under-sun.png", width=60)
    st.title("CropLogic")
    st.caption("AI-Driven Smart Agriculture DSS")
    st.divider()

    page = st.radio("Navigation", [
        "📊 Dashboard",
        "🗺️ Field Map",
        "📈 NDVI Analysis",
        "🔬 Disease Scanner",
        "🌤️ Weather Risk",
        "💡 Recommendations",
        "➕ Register Field",
        "🎓 Train CNN Model",
    ])

    st.divider()
    st.caption("Data Sources")
    st.markdown("🛰 NASA POWER API (free)  \n🌤 Open-Meteo (free)  \n🤖 TensorFlow / ResNet-50  \n📊 PlantVillage (38 classes)")
    st.divider()
    st.caption(f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
STRESS_COLOR = {
    "Healthy": "#3fb950",
    "Mild Stress": "#d29922",
    "Moderate Stress": "#db6d28",
    "Severe Stress": "#f85149",
}
RISK_COLOR = {"Low": "#3fb950", "Medium": "#d29922", "High": "#f85149"}

def stress_badge(level):
    cls = {"Healthy":"low","Mild Stress":"medium","Moderate Stress":"high","Severe Stress":"critical"}.get(level,"low")
    return f'<span class="badge-{cls}">{level}</span>'

def risk_badge(level):
    cls = {"Low":"low","Medium":"medium","High":"critical"}.get(level,"low")
    return f'<span class="badge-{cls}">{level} Risk</span>'

def ndvi_color(v):
    if v >= 0.65: return "#3fb950"
    if v >= 0.45: return "#d29922"
    if v >= 0.30: return "#db6d28"
    return "#f85149"

# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE: DASHBOARD ──────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    st.header("System Dashboard")

    fields = st.session_state.fields
    summaries = []

    with st.spinner("Fetching live data from NASA POWER + Open-Meteo…"):
        for f in fields:
            df  = get_ndvi(f["id"], f["lat"], f["lon"], f["crop"], f["name"])
            wx  = get_weather(f["lat"], f["lon"])
            ana = analyse_field(f, df, wx)
            summaries.append(ana)

    # KPI row
    total   = len(fields)
    avg_ndvi= round(np.mean([s["ndvi_now"] for s in summaries]), 3)
    stressed= sum(1 for s in summaries if s["zscore"]["alert"])
    hi_risk = sum(1 for s in summaries if s["risk"]["level"] == "High")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌾 Fields Active",   total,     help="Total registered fields")
    col2.metric("🛰 Avg NDVI",        avg_ndvi,  delta="+healthy" if avg_ndvi > 0.6 else "⚠ below threshold")
    col3.metric("⚠ Stress Alerts",   stressed,  delta=f"{stressed}/{total} fields flagged")
    col4.metric("🦠 High Disease Risk", hi_risk, delta=f"{hi_risk}/{total} fields")

    st.divider()

    # Summary table
    st.subheader("Field Status Overview")
    rows = []
    for s in summaries:
        rows.append({
            "Field"       : s["field"]["name"],
            "Crop"        : s["field"]["crop"],
            "NDVI"        : s["ndvi_now"],
            "Z-Score"     : s["zscore"]["zscore"],
            "Stress"      : s["zscore"]["stress_level"],
            "Risk"        : s["risk"]["level"],
            "Temp (°C)"   : s["weather"]["temperature_c"],
            "Humidity (%)" : s["weather"]["humidity_pct"],
            "Rain 7d (mm)" : round(s["weather"]["rain_7d_mm"], 1),
        })
    df_table = pd.DataFrame(rows)
    st.dataframe(
        df_table.style
            .background_gradient(subset=["NDVI"], cmap="RdYlGn", vmin=0.1, vmax=0.9)
            .applymap(lambda v: f"color: {STRESS_COLOR.get(v,'#8b949e')}", subset=["Stress"])
            .applymap(lambda v: f"color: {RISK_COLOR.get(v,'#8b949e')}", subset=["Risk"]),
        use_container_width=True, hide_index=True,
    )

    # NDVI trend chart (first field)
    st.subheader(f"NDVI Trend — {fields[0]['name']}")
    df0  = get_ndvi(fields[0]["id"], fields[0]["lat"], fields[0]["lon"],
                    fields[0]["crop"], fields[0]["name"])
    base = CROP_BASELINES[CROP_TYPE_MAP.get(fields[0]["crop"], "cereal")]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df0["date"], y=df0["ndvi"],
        name="NDVI", line=dict(color="#3fb950", width=2),
        fill="tozeroy", fillcolor="rgba(63,185,80,0.08)"))
    fig.add_hline(y=base["mu"] + base["tau"]*base["sigma"],
                  line=dict(color="#f85149", dash="dash", width=1), annotation_text="Stress upper")
    fig.add_hline(y=base["mu"],
                  line=dict(color="rgba(255,255,255,0.3)", dash="dot", width=1), annotation_text="μ")
    fig.add_hline(y=base["mu"] - base["tau"]*base["sigma"],
                  line=dict(color="#f85149", dash="dash", width=1), annotation_text="Stress lower")
    fig.update_layout(
        height=300, template="plotly_dark", margin=dict(l=0,r=0,t=20,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", range=[0, 1]),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE: FIELD MAP ──────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🗺️ Field Map":
    st.header("Field Map")
    st.caption("Interactive map — fields colour-coded by health status")

    m = folium.Map(location=[15.0, 78.0], zoom_start=6,
                   tiles="CartoDB dark_matter")

    for f in st.session_state.fields:
        df  = get_ndvi(f["id"], f["lat"], f["lon"], f["crop"], f["name"])
        if df.empty:
            continue
        ndvi_now = float(df["ndvi"].iloc[-1])
        zr       = compute_zscore(ndvi_now, f["crop"])
        color    = {"Healthy":"green","Mild Stress":"orange",
                    "Moderate Stress":"red","Severe Stress":"darkred"}.get(zr["stress_level"],"gray")
        folium.CircleMarker(
            location=[f["lat"], f["lon"]],
            radius   = max(8, f["area"] * 2),
            color    = color, fill=True, fill_color=color, fill_opacity=0.75,
            popup    = folium.Popup(
                f"<b>{f['name']}</b><br>{f['crop']}<br>"
                f"NDVI: {ndvi_now:.3f}<br>Status: {zr['stress_level']}", max_width=200),
            tooltip  = f"{f['name']} — {zr['stress_level']}",
        ).add_to(m)

    st_folium(m, width="100%", height=500)

    st.subheader("Field Registry")
    st.dataframe(pd.DataFrame([{
        "ID": f["id"], "Name": f["name"], "Crop": f["crop"],
        "Lat": f["lat"], "Lon": f["lon"], "Area (ha)": f["area"], "Sow Date": f["sow"],
    } for f in st.session_state.fields]), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE: NDVI ANALYSIS ──────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 NDVI Analysis":
    st.header("NDVI Analysis")

    fields   = st.session_state.fields
    sel_name = st.selectbox("Select Field", [f["name"] for f in fields])
    field    = next(f for f in fields if f["name"] == sel_name)
    days     = st.slider("History (days)", 7, 90, 30)

    with st.spinner("Loading NASA POWER data…"):
        df = get_ndvi(field["id"], field["lat"], field["lon"],
                      field["crop"], field["name"], days)

    if df.empty:
        st.error("No NDVI data available for this field.")
        st.stop()

    ndvi_now = float(df["ndvi"].iloc[-1])
    zr       = compute_zscore(ndvi_now, field["crop"])
    base     = CROP_BASELINES[CROP_TYPE_MAP.get(field["crop"], "cereal")]

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NDVI Now",  f"{ndvi_now:.4f}", help="Field-averaged NDVI")
    c2.metric("Z-Score",   f"{zr['zscore']:+.3f}", help="Deviation from crop baseline")
    c3.metric("Stress",    zr["stress_level"])
    trend_val = float(df["beta1"].iloc[-1]) if "beta1" in df.columns else 0
    c4.metric("Trend β₁", f"{trend_val:+.5f}/day",
              delta="⬇ Declining" if trend_val < -0.003 else "⬆ Stable")

    st.divider()

    # Main NDVI chart with threshold bands
    fig = go.Figure()
    upper = base["mu"] + base["tau"] * base["sigma"]
    lower = base["mu"] - base["tau"] * base["sigma"]

    fig.add_traces([
        go.Scatter(x=df["date"], y=[upper]*len(df), name=f"μ+τσ (={upper:.3f})",
                   line=dict(color="rgba(248,81,73,0.6)", dash="dash", width=1)),
        go.Scatter(x=df["date"], y=[base["mu"]]*len(df), name=f"μ (={base['mu']})",
                   line=dict(color="rgba(255,255,255,0.3)", dash="dot", width=1)),
        go.Scatter(x=df["date"], y=[lower]*len(df), name=f"μ−τσ (={lower:.3f})",
                   line=dict(color="rgba(248,81,73,0.6)", dash="dash", width=1),
                   fill="tonexty", fillcolor="rgba(248,81,73,0.05)"),
        go.Scatter(x=df["date"], y=df["ndvi"], name="NDVI",
                   line=dict(color="#3fb950", width=2.5),
                   fill="tozeroy", fillcolor="rgba(63,185,80,0.1)",
                   mode="lines+markers", marker=dict(size=4)),
    ])
    fig.update_layout(
        title=f"NDVI Time Series — {field['name']}",
        height=350, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Date", yaxis_title="NDVI",
        yaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Z-score chart
    fig2 = go.Figure()
    colors = [STRESS_COLOR.get(s, "#8b949e") for s in df["stress_level"]]
    fig2.add_trace(go.Bar(x=df["date"], y=df["zscore"], name="Z-Score",
                           marker_color=colors))
    fig2.add_hline(y=base["tau"],  line=dict(color="#f85149", dash="dash"), annotation_text=f"+τ={base['tau']}")
    fig2.add_hline(y=-base["tau"], line=dict(color="#f85149", dash="dash"), annotation_text=f"-τ={base['tau']}")
    fig2.update_layout(
        title="Z-Score (Vegetation Stress Detection)",
        height=250, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Raw data table
    with st.expander("Raw NDVI Data"):
        st.dataframe(df[["date","ndvi","zscore","stress_level","beta1","source"]].round(4),
                     use_container_width=True, hide_index=True)

    # Mathematical formulas
    with st.expander("📐 Mathematical Model"):
        st.latex(r"NDVI = \frac{NIR - RED}{NIR + RED}")
        st.latex(r"Z = \frac{NDVI_t - \mu_{crop}}{\sigma_{crop}}")
        st.latex(r"|Z| > \tau \Rightarrow \text{Vegetation Stress Alert}")
        st.latex(r"NDVI(t_k) = \beta_0 + \beta_1 t_k + \varepsilon_k")
        st.info(f"Crop: **{field['crop']}** | μ={base['mu']} | σ={base['sigma']} | τ={base['tau']}")


# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE: DISEASE SCANNER ────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔬 Disease Scanner":
    st.header("Disease Scanner — CNN Classifier (ResNet-50)")
    st.caption("Upload a leaf image to identify plant disease using the PlantVillage-trained model")

    col1, col2 = st.columns([1, 1])

    with col1:
        crop_sel = st.selectbox("Crop Type", list(CROP_TYPE_MAP.keys()))
        uploaded = st.file_uploader("Upload Leaf Image", type=["jpg","jpeg","png","webp"])

        if uploaded:
            st.image(uploaded, caption="Uploaded leaf image", use_column_width=True)

        run_btn = st.button("🤖 Run Classification", type="primary", use_container_width=True)

    with col2:
        if run_btn and uploaded:
            with st.spinner("Running ResNet-50 inference…"):
                from config import CNN_MODEL, MODEL_DIR
                idx_path = os.path.join(MODEL_DIR, "class_indices.json")
                clf      = DiseaseClassifier(
                    model_path      = CNN_MODEL if os.path.exists(CNN_MODEL) else None,
                    class_index_path= idx_path  if os.path.exists(idx_path)  else None,
                )
                results = clf.predict(uploaded, top_k=5)

            top = results[0] if results else {}
            cls_name = top.get("class","Unknown").replace("___"," — ").replace("_"," ")
            conf     = top.get("confidence", 0) * 100
            healthy  = top.get("healthy", False)

            if healthy:
                st.success(f"✅ **{cls_name}** — No disease detected ({conf:.1f}%)")
            else:
                severity = "Low" if conf < 60 else "High" if conf > 80 else "Moderate"
                st.error(f"🚨 **{cls_name}** detected — {conf:.1f}% confidence")

            # Confidence bars
            st.subheader("Top-5 Predictions")
            for r in results:
                c = r["class"].replace("___"," — ").replace("_"," ")
                p = r["confidence"] * 100
                col = "#3fb950" if r["healthy"] else "#f85149"
                st.markdown(f"**{c}**")
                st.progress(r["confidence"])
                st.caption(f"{p:.1f}%")

            # Recommendation
            if not healthy and conf > 50:
                st.warning(f"⚠ **Treatment**: Apply targeted pesticide/fungicide for {cls_name}. "
                           f"Remove infected leaves. Improve ventilation and reduce humidity.")

        elif run_btn:
            st.warning("Please upload a leaf image first.")
        else:
            st.info("Upload a leaf image and click 'Run Classification' to begin.")
            st.subheader("Supported Disease Classes")
            cats = {}
            for d in DISEASE_CLASSES:
                parts = d.split("___")
                crop_k = parts[0]
                cats.setdefault(crop_k, []).append(parts[1].replace("_"," ") if len(parts)>1 else d)
            for crop_k, diseases in cats.items():
                with st.expander(f"🌿 {crop_k.replace('_',' ')} ({len(diseases)} classes)"):
                    st.write(", ".join(diseases))


# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE: WEATHER RISK ───────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🌤️ Weather Risk":
    st.header("Weather Risk Analysis")

    fields   = st.session_state.fields
    sel_name = st.selectbox("Select Field", [f["name"] for f in fields])
    field    = next(f for f in fields if f["name"] == sel_name)

    with st.spinner("Fetching Open-Meteo weather…"):
        wx = get_weather(field["lat"], field["lon"])

    if not wx:
        st.error("Could not fetch weather data. Check internet connection.")
        st.stop()

    cur   = wx["current"]
    daily = wx["daily"]

    # Current conditions
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🌡 Temperature",   f"{cur['temperature_2m']}°C")
    c2.metric("💧 Humidity",      f"{cur['relative_humidity_2m']}%")
    c3.metric("🌧 Precipitation", f"{cur['precipitation']} mm")
    c4.metric("💨 Wind Speed",    f"{cur['wind_speed_10m']} km/h")

    st.caption(f"📍 {field['name']} · {field['lat']}°N, {field['lon']}°E · Source: Open-Meteo API")
    st.divider()

    # 7-day forecast charts
    dates    = [d[:10] for d in daily.get("time", [])]
    tmax     = daily.get("temp_max", [])
    tmin     = daily.get("temp_min", [])
    precip   = daily.get("precip_sum", [])
    hum_max  = daily.get("humidity_max", [])

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=tmax, name="T max", line=dict(color="#f85149")))
        fig.add_trace(go.Scatter(x=dates, y=tmin, name="T min", line=dict(color="#58a6ff")))
        fig.update_layout(title="Temperature 7-Day Forecast (°C)", height=280,
                          template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=dates, y=precip, name="Rainfall (mm)",
                               marker_color="#58a6ff", opacity=0.8))
        fig2.update_layout(title="Rainfall Forecast (mm/day)", height=280,
                           template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    # Risk matrix
    st.subheader("Disease Risk Matrix")
    df_ndvi = get_ndvi(field["id"], field["lat"], field["lon"],
                       field["crop"], field["name"])
    ndvi_now = float(df_ndvi["ndvi"].iloc[-1])
    beta1    = float(df_ndvi["beta1"].iloc[-1]) if "beta1" in df_ndvi else 0
    rain_7d  = sum(precip or [0])
    temp_now = cur["temperature_2m"]
    hum_now  = cur["relative_humidity_2m"]

    from modules.ndvi_analysis import compute_disease_risk
    rr = compute_disease_risk(ndvi_now, beta1, hum_now, rain_7d, temp_now, field["crop"])

    overall_color = RISK_COLOR[rr["level"]]
    st.markdown(f"""
    **Overall Disease Risk: <span style='color:{overall_color};font-size:20px;'>{rr['level']}</span>
    — P={rr['probability']:.2f}**
    """, unsafe_allow_html=True)

    factor_rows = []
    for factor, score in rr["factors"].items():
        level = "High" if score > 0.18 else "Medium" if score > 0.08 else "Low"
        factor_rows.append({"Factor": factor.title(), "Score": round(score,3), "Contribution": level})
    st.dataframe(pd.DataFrame(factor_rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE: RECOMMENDATIONS ────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💡 Recommendations":
    st.header("Agronomic Recommendations")
    st.caption("Priority-sorted interventions based on NDVI, Z-score, disease risk, and weather")

    fields = st.session_state.fields
    sel_all = st.checkbox("Show all fields", value=True)

    if not sel_all:
        sel_name = st.selectbox("Select Field", [f["name"] for f in fields])
        fields   = [f for f in fields if f["name"] == sel_name]

    all_recs = []
    with st.spinner("Generating recommendations…"):
        for f in fields:
            df  = get_ndvi(f["id"], f["lat"], f["lon"], f["crop"], f["name"])
            wx  = get_weather(f["lat"], f["lon"])
            ana = analyse_field(f, df, wx)
            recs = generate_recommendations(ana)
            for r in recs:
                r["field_name"] = f["name"]
                all_recs.append(r)

    PRIORITY_ORDER = {"CRITICAL":0,"HIGH":1,"MEDIUM":2,"LOW":3}
    all_recs.sort(key=lambda r: PRIORITY_ORDER.get(r["priority"],99))

    COLOR_MAP = {"CRITICAL":"#f85149","HIGH":"#db6d28","MEDIUM":"#d29922","LOW":"#3fb950"}
    for r in all_recs:
        color = COLOR_MAP.get(r["priority"],"#8b949e")
        st.markdown(f"""
        <div class="rec-box rec-{r['priority'].lower()}">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
            <span style="font-size:22px;">{r['icon']}</span>
            <div>
              <span style="background:rgba(0,0,0,0.3);color:{color};border:1px solid {color};
                           padding:2px 8px;border-radius:20px;font-size:11px;font-weight:700;">
                {r['priority']}
              </span>
              <span style="margin-left:8px;font-size:11px;color:#8b949e;">→ {r.get('field_name','')}</span>
            </div>
          </div>
          <div style="font-size:15px;font-weight:700;color:#e6edf3;margin-bottom:4px;">{r['title']}</div>
          <div style="font-size:13px;color:#8b949e;">{r['description']}</div>
          <div style="margin-top:6px;font-size:11px;color:{color};">⏰ {r['action_window']}</div>
        </div>
        """, unsafe_allow_html=True)

    if st.button("📄 Export Recommendations as Text"):
        text = "\n".join([format_report(r["field_name"], [r]) for r in all_recs])
        st.download_button("⬇ Download Report", text, "croplogic_recommendations.txt", "text/plain")


# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE: REGISTER FIELD ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "➕ Register Field":
    st.header("Register New Field")

    with st.form("register_form"):
        c1, c2 = st.columns(2)
        name     = c1.text_input("Field Name", placeholder="e.g. North Paddy Block A")
        crop     = c2.selectbox("Crop Type", list(CROP_TYPE_MAP.keys()))
        lat      = c1.number_input("Latitude",  value=12.97, format="%.4f", step=0.001)
        lon      = c2.number_input("Longitude", value=77.59, format="%.4f", step=0.001)
        area     = c1.number_input("Area (Hectares)", value=2.0, min_value=0.1, step=0.1)
        sow_date = c2.date_input("Sowing Date")
        notes    = st.text_area("Notes (optional)", placeholder="Soil type, irrigation method…")
        submitted = st.form_submit_button("✅ Register & Fetch Baseline NDVI", type="primary")

    if submitted:
        if not name:
            st.error("Field name is required.")
        else:
            new_id = max(f["id"] for f in st.session_state.fields) + 1
            new_field = {
                "id": new_id, "name": name, "crop": crop,
                "lat": lat, "lon": lon, "area": area,
                "sow": str(sow_date), "notes": notes,
            }
            st.session_state.fields.append(new_field)
            with st.spinner(f"Fetching NDVI baseline for {name} via NASA POWER…"):
                df = build_ndvi_series(new_field, days=30)
            if not df.empty:
                st.success(f"✅ **{name}** registered! Baseline NDVI: {df['ndvi'].iloc[-1]:.4f} "
                           f"| {len(df)} days loaded from NASA POWER.")
                st.dataframe(df[["date","ndvi","zscore","stress_level"]].tail(10),
                             use_container_width=True, hide_index=True)
            else:
                st.warning("Field registered but could not fetch NDVI data — check coordinates.")


# ─────────────────────────────────────────────────────────────────────────────
# ── PAGE: TRAIN CNN ──────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🎓 Train CNN Model":
    st.header("Train ResNet-50 on PlantVillage Dataset")

    st.info("""
    **Steps to train the CNN model:**
    1. Download PlantVillage dataset from Kaggle (see instructions below)
    2. Set the data directory path
    3. Click **Start Training**
    
    The model uses a 2-phase training protocol:
    - **Phase 1** (20 epochs): Only the new classification head is trained (ResNet-50 frozen)
    - **Phase 2** (up to 30 epochs): Top 30 ResNet-50 layers are unfrozen + fine-tuned with LR=1e-5
    """)

    with st.expander("📥 How to Download PlantVillage Dataset"):
        st.code("""
# Option 1: Kaggle CLI
pip install kaggle
kaggle datasets download -d vipoooool/new-plant-diseases-dataset
unzip new-plant-diseases-dataset.zip -d data/plantvillage

# Option 2: opendatasets
pip install opendatasets
import opendatasets as od
od.download("https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")

# Option 3: Google Drive (Colab)
# Upload kaggle.json to /root/.kaggle/ then run Option 1
        """, language="bash")

    from config import PLANTVILLAGE_DIR
    data_dir = st.text_input("PlantVillage Data Directory",
                              value=os.path.join(PLANTVILLAGE_DIR, "New Plant Diseases Dataset(Augmented)"))
    st.caption("Make sure this folder exists or point to your downloaded dataset root, e.g. data/plantvillage")
    col1, col2, col3 = st.columns(3)
    phase1_ep = col1.number_input("Phase 1 Epochs", 5, 50, 20)
    phase2_ep = col2.number_input("Phase 2 Epochs", 5, 50, 30)
    batch_sz  = col3.selectbox("Batch Size", [16, 32, 64], index=1)

    if st.button("🚀 Start Training", type="primary"):
        if not os.path.isdir(data_dir):
            st.error(f"Directory not found: {data_dir}")
        else:
            try:
                import tensorflow as tf
                st.info(f"TensorFlow {tf.__version__} | GPU: {len(tf.config.list_physical_devices('GPU'))} device(s)")
                from modules.cnn_model import train
                with st.spinner("Training in progress… (check terminal for live output)"):
                    model, h1, h2 = train(data_dir)
                st.success("✅ Training complete! Model saved to `models/resnet50_plantvillage.h5`")

                # Plot training curves
                fig = make_subplots(rows=1, cols=2, subplot_titles=["Accuracy", "Loss"])
                for h, label in [(h1,"Phase 1"),(h2,"Phase 2")]:
                    fig.add_trace(go.Scatter(y=h.history["accuracy"],    name=f"{label} train acc"), row=1,col=1)
                    fig.add_trace(go.Scatter(y=h.history["val_accuracy"],name=f"{label} val acc"),   row=1,col=1)
                    fig.add_trace(go.Scatter(y=h.history["loss"],        name=f"{label} train loss"),row=1,col=2)
                    fig.add_trace(go.Scatter(y=h.history["val_loss"],    name=f"{label} val loss"),  row=1,col=2)
                fig.update_layout(height=350, template="plotly_dark",
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            except ImportError as e:
                st.error(f"TensorFlow not installed: {e}")
            except Exception as e:
                st.exception(e)

    st.divider()
    st.subheader("Train Risk Model (GBM)")
    st.caption("Trains Gradient Boosting classifier on synthetic field observations")
    if st.button("🚀 Train Risk Model"):
        try:
            from modules.risk_model import train_risk_model
            with st.spinner("Training GBM…"):
                model, scaler = train_risk_model()
            st.success("✅ GBM risk model saved to `models/risk_gbm.joblib`")
        except Exception as e:
            st.exception(e)
