# ✅ Chunk 1 of 3 — Full Main App (~362 lines)
# --- Imports, Sidebar, Yelp, Sentiment, Traffic, Geolocation ---

import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
import torch
import joblib
import os
import pandas as pd
import plotly.express as px
from streamlit_folium import st_folium
import folium
from dotenv import load_dotenv
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
import time
import logging

# --- Load Environment ---
load_dotenv()
YELP_API_KEY = os.getenv("YELP_API_KEY")
YELP_HEADERS = {"Authorization": f"Bearer {YELP_API_KEY}"}

# --- App Layout ---
st.set_page_config(page_title="Retail AI Forecast", layout="wide")
st.sidebar.title("🔧 Settings")
store_type = st.sidebar.selectbox("Store Type", ["Any", "Coffee Shop", "Boutique", "Fast Food", "Other"])
theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=1)
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = (theme == "Dark")

if st.sidebar.button("🗑️ Clear History"):
    if os.path.exists("sales_history.csv"):
        os.remove("sales_history.csv")
        st.sidebar.success("History cleared.")

# --- Theme Styling ---
def set_theme():
    dark = st.session_state.dark_mode
    style = f"""
        <style>
        body, .main, .block-container {{
            background-color: {'#111827' if dark else '#ffffff'} !important;
            color: {'#e5e7eb' if dark else '#111827'} !important;
        }}
        .stTextInput>div>div>input {{
            background-color: {'#1f2937' if dark else 'white'} !important;
            color: {'#e5e7eb' if dark else '#111827'} !important;
        }}
        .stButton>button {{
            background-color: #6366f1;
            color: white;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)
set_theme()

# --- Yelp Business Search ---
def search_yelp_business(name, location):
    url = "https://api.yelp.com/v3/businesses/search"
    params = {"term": name, "location": location, "limit": 1}
    try:
        response = requests.get(url, headers=YELP_HEADERS, params=params)
        if response.status_code == 200:
            data = response.json()
            if data["businesses"]:
                return data["businesses"][0]
    except:
        pass
    return None

# --- Sentiment Score (Yelp-based) ---
def get_yelp_sentiment_score(business):
    if not business:
        return 50.0, "❓ Unknown"
    rating = business.get("rating", 3.0)
    review_count = business.get("review_count", 0)
    score = round(min(100, max(0, (rating - 3) * 25 + review_count * 0.1)), 1)
    return score, "⭐️ Rating only"

# --- Mock Traffic Signal ---
def get_mock_placer_traffic(zip_code):
    if not zip_code:
        return 0.45
    hashval = sum(ord(c) for c in zip_code) % 10
    return round(0.3 + 0.05 * hashval, 2)

# --- Geolocation ---
def get_coords_from_store_name(name, zip_code):
    if not name:
        return []
    business = search_yelp_business(name, zip_code if zip_code else "USA")
    if business:
        coords = business.get("coordinates", {})
        lat, lon = coords.get("latitude"), coords.get("longitude")
        if lat and lon:
            return [(lat, lon, business.get("name") + ", " + business.get("location", {}).get("address1", "Yelp location"))]
    geolocator = Nominatim(user_agent="retail_ai_locator")
    def safe_geocode(query):
        for _ in range(3):
            try:
                return geolocator.geocode(query, exactly_one=True, timeout=10)
            except (GeocoderTimedOut, GeocoderUnavailable):
                time.sleep(1)
        return None
    zip_location = safe_geocode(f"{zip_code}, USA") if zip_code else None
    full_query = f"{name}, {zip_code}, USA" if zip_code else f"{name}, USA"
    name_location = safe_geocode(full_query)
    if name_location and zip_location:
        dist = np.sqrt((name_location.latitude - zip_location.latitude)**2 + (name_location.longitude - zip_location.longitude)**2)
        if dist > 0.3:
            return []
    if name_location:
        return [(name_location.latitude, name_location.longitude, name_location.address)]
    return []
# ✅ Chunk 2 of 3 — Feature Extraction, Satellite Fetch, Model Prediction

# --- Map & Location Picker ---
def show_map_with_selection(options):
    st.subheader("📍 Select Store Location")
    m = folium.Map(location=[options[0][0], options[0][1]], zoom_start=14)
    for lat, lon, label in options:
        folium.Marker(location=[lat, lon], tooltip=label).add_to(m)
    st_folium(m, height=350, width=700)
    return options[0][:2]

# --- Satellite Fetch ---
def fetch_or_upload_satellite_image(coords):
    uploaded = st.file_uploader("Upload a satellite image (optional)", type=["jpg", "jpeg", "png"])
    if uploaded:
        return Image.open(uploaded).convert("RGB")
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        st.error("Missing Google Maps API key")
        return Image.new("RGB", (512, 512), color=(180, 180, 180))
    try:
        url = f"https://maps.googleapis.com/maps/api/staticmap?center={coords[0]},{coords[1]}&zoom=18&size=600x400&maptype=satellite&key={api_key}"
        res = requests.get(url)
        res.raise_for_status()
        return Image.open(io.BytesIO(res.content)).convert("RGB")
    except Exception as e:
        st.error(f"Satellite error: {e}")
        return Image.new("RGB", (512, 512), color=(160, 160, 160))

# --- Feature Extraction ---
def extract_satellite_features(img):
    model = resnet18(pretrained=True)
    model.eval()
    model = nn.Sequential(*list(model.children())[:-1])
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = model(tensor).view(1, -1).numpy().flatten()
    return np.resize(features, 512)

# --- Build Feature Vector ---
def build_feature_vector(img, coords, store, zip_code):
    features = np.concatenate([extract_satellite_features(img), [coords[0], coords[1]]])
    business = search_yelp_business(store, zip_code)
    foot = get_mock_placer_traffic(zip_code)
    soc, source = get_yelp_sentiment_score(business)
    return features, foot, soc, source

# --- Trend Arrow ---
def get_trend_arrow(store, current):
    try:
        df = pd.read_csv("sales_history.csv")
        df = df[df["store"].str.lower() == store.lower()].sort_values("timestamp")
        if len(df) < 2:
            return ""
        prev = df.iloc[-2]["sales"]
        if current > prev: return "🔼 Higher than last week"
        if current < prev: return "🔽 Lower than last week"
        return "⏸️ Same as last week"
    except:
        return ""

# --- Fallback Logger ---
def log_fallback_usage():
    with open("fallback_log.txt", "a") as f:
        f.write(f"Fallback triggered at {pd.Timestamp.now()}\n")

# --- Load Model ---
def load_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    if os.path.exists("real_sales_data.csv"):
        try:
            df = pd.read_csv("real_sales_data.csv")
            X = df[[f"f{i}" for i in range(512)] + ["lat", "lon"]]
            y = df["sales"]
            model = GradientBoostingRegressor()
            model.fit(X, y)
            joblib.dump(model, "model.pkl")
            return model
        except:
            log_fallback_usage()
    dummy = DummyRegressor(strategy="constant", constant=np.random.randint(10000, 30000))
    dummy.fit([[0]*514], [dummy.constant])
    log_fallback_usage()
    return dummy
# ✅ Chunk 3 of 3 — Final logic: prediction save, dashboard, recs

import plotly.express as px

# --- Save Predictions ---
def save_prediction(store, coords, pred, foot, soc):
    df = pd.DataFrame([[store, coords[0], coords[1], pred, foot, soc, pd.Timestamp.now()]],
                      columns=["store", "lat", "lon", "sales", "foot", "social", "timestamp"])
    if os.path.exists("sales_history.csv"):
        try:
            old = pd.read_csv("sales_history.csv", on_bad_lines='skip')
            df = pd.concat([old, df], ignore_index=True)
        except:
            st.warning("Corrupted history file. Overwriting.")
    df.to_csv("sales_history.csv", index=False)

# --- Trend Dashboard ---
def show_trend_dashboard(store):
    if not os.path.exists("sales_history.csv"):
        return
    try:
        df = pd.read_csv("sales_history.csv", on_bad_lines='skip')
    except:
        return
    df = df[df["store"].str.lower() == store.lower()]
    if df.empty:
        return
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    st.subheader("📈 Sales Over Time")
    st.plotly_chart(px.line(df, x="timestamp", y="sales", title="Weekly Sales", markers=True))

    st.subheader("📊 Traffic & Social Sentiment")
    df_long = df.melt(id_vars="timestamp", value_vars=["foot", "social"], var_name="Metric", value_name="Score")
    st.plotly_chart(px.line(df_long, x="timestamp", y="Score", color="Metric", markers=True))

# --- Recommendations ---
def generate_recommendations(store, foot, soc, sales):
    r = []
    if foot < 0.4:
        r.append("🚶 Low foot traffic: Partner locally and distribute coupons nearby.")
    elif foot < 0.6:
        r.append("📣 Average traffic: Push flash sales during peaks.")
    else:
        r.append("🏃 High traffic: Upsell combos and reduce wait time.")

    if soc < 40:
        r.append("📉 Weak online sentiment: Ask for reviews and optimize Google Business profile.")
    elif soc < 70:
        r.append("📱 Moderate buzz: Encourage shares with referral codes.")
    else:
        r.append("🔥 High sentiment: Launch loyalty program or event marketing.")

    if sales > 30000:
        r.append("📈 Strong sales: Consider expansion or menu innovation.")
    elif sales < 10000:
        r.append("⚖️ Weak sales: Try time-based deals or collab promos.")
    return r

# --- Final Prediction Action ---
if store and coords:
    if st.button("📊 Predict & Analyze"):
        try:
            features, foot, soc = build_feature_vector(image, coords, store, zip_code)
            model = load_model()
            pred = max(model.predict([features])[0], 0)
            trend = get_trend_arrow(store, pred)

            if hasattr(model, "estimators_"):
                preds = np.array([est.predict([features])[0] for est in model.estimators_])
                std = np.std(preds)
                ci_low = max(pred - 1.96 * std, 0)
                ci_up = pred + 1.96 * std
                st.markdown(f"## 💰 Predicted Weekly Sales: **${pred:,.2f}** {trend}")
                st.caption(f"95% Confidence Interval: ${ci_low:,.2f} - ${ci_up:,.2f}")
            else:
                st.markdown(f"## 💰 Predicted Weekly Sales: **${pred:,.2f}** {trend}")

            save_prediction(store, coords, pred, foot, soc)
            show_trend_dashboard(store)
            st.subheader("📦 Strategy Recommendations")
            for tip in generate_recommendations(store, foot, soc, pred):
                st.markdown(f"- {tip}")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
