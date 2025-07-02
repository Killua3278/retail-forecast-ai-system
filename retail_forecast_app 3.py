# --- Imports and Setup ---
from sklearn.datasets import make_regression
import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
import torch
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import folium
from dotenv import load_dotenv
import torch.nn as nn
from geopy.geocoders import Nominatim

# Load environment variables from .env if available
load_dotenv()

# --- Sidebar Config ---
st.set_page_config(page_title="Retail Forecast AI", layout="wide")

st.sidebar.title("⚙️ Settings")
store_type = st.sidebar.selectbox("Store Type", ["Any", "Coffee Shop", "Boutique", "Fast Food", "Other"])
theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = (theme == "Dark")

if st.sidebar.button("Clear Sales History"):
    if os.path.exists("sales_history.csv"):
        os.remove("sales_history.csv")
        st.sidebar.success("Sales history cleared!")

# --- Theme Styling ---
def set_theme():
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
        body, .main, .block-container, .sidebar .sidebar-content {
            background-color: #121212 !important;
            color: #e0e0e0 !important;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        </style>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        body, .main, .block-container, .sidebar .sidebar-content {
            background-color: white !important;
            color: black !important;
        }
        .stButton>button {
            background-color: #f0f0f0;
            color: black;
        }
        </style>""", unsafe_allow_html=True)
set_theme()

# --- Imports for vision ---
try:
    from torchvision import transforms
    from torchvision.models import resnet18
except ModuleNotFoundError as e:
    st.error("Please install 'torchvision' in requirements.txt")
    raise e

# --- 1. Upload or Fetch Satellite Image ---
def fetch_or_upload_satellite_image(coords):
    uploaded_file = st.file_uploader("Upload satellite image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        return Image.open(uploaded_file).convert("RGB")
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        st.warning("No valid Google Maps API key found. Showing placeholder.")
        return Image.new("RGB", (512, 512), color=(150, 150, 150))
    try:
        url = f"https://maps.googleapis.com/maps/api/staticmap?center={coords[0]},{coords[1]}&zoom=17&size=600x400&maptype=satellite&key={api_key}"
        r = requests.get(url)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        st.error(f"Satellite fetch error: {e}")
        return Image.new("RGB", (512, 512), color=(200, 200, 200))

# --- 2. Extract Vision Features ---
def extract_satellite_features(image):
    model = resnet18(pretrained=True)
    model.eval()
    model = nn.Sequential(*list(model.children())[:-1])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(tensor).view(1, -1)
    return features.numpy().flatten()

# --- Mocked APIs ---
def get_safegraph_score(lat, lon):
    return np.random.uniform(0.3, 0.85)

def fetch_social_sentiment(lat, lon):
    return np.random.randint(30, 100)

# --- Feature Construction ---
def build_feature_vector(image, coords):
    vision = extract_satellite_features(image)
    foot = get_safegraph_score(*coords)
    social = fetch_social_sentiment(*coords)
    return np.concatenate([vision, [foot, social]]), foot, social

# --- Model ---
def load_model():
    from sklearn.ensemble import GradientBoostingRegressor
    if os.path.exists("model.pkl"):
        model = joblib.load("model.pkl")
        if hasattr(model, 'n_features_in_') and model.n_features_in_ != 514:
            raise ValueError("Model feature mismatch.")
        return model
    X, y = make_regression(n_samples=100, n_features=514, noise=0.1)
    model = GradientBoostingRegressor().fit(X, y)
    joblib.dump(model, "model.pkl")
    return model

# --- Logging ---
sales_log = "sales_history.csv"
def save_prediction(store, coords, pred, foot, soc):
    row = pd.DataFrame([[store, coords[0], coords[1], store_type, pred, foot, soc]],
                       columns=["store", "lat", "lon", "type", "sales", "foot", "social"])
    if os.path.exists(sales_log):
        past = pd.read_csv(sales_log)
        new = pd.concat([past, row], ignore_index=True)
    else:
        new = row
    new.to_csv(sales_log, index=False)

# --- Graph + Competitor Insight ---
def plot_trends(store):
    if not os.path.exists(sales_log):
        return st.info("No data to display yet.")
    df = pd.read_csv(sales_log)
    df = df[df["store"].astype(str).str.lower() == store.lower()]
    if df.empty:
        return st.info("No data found for that store name.")
    df["timestamp"] = pd.date_range(end=pd.Timestamp.today(), periods=len(df))
    st.line_chart(df.set_index("timestamp")["sales"], use_container_width=True)
    st.area_chart(df.set_index("timestamp")[["foot", "social"]], use_container_width=True)
    
    all_df = pd.read_csv(sales_log)
    group_avg = all_df.groupby("type")["sales"].mean().reset_index()
    st.bar_chart(group_avg.set_index("type"))

# --- Geolocation from Store Name ---
def get_coords_from_store_name(name):
    geolocator = Nominatim(user_agent="retail_app")
    try:
        loc = geolocator.geocode(name)
        if loc:
            return (loc.latitude, loc.longitude)
    except:
        pass
    return None

# --- Map UI ---
def get_coords_from_map():
    st.subheader("📍 Select Your Store's Location")
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
    m.add_child(folium.LatLngPopup())
    out = st_folium(m, height=300, width=700)
    if out.get("last_clicked"):
        coords = (out["last_clicked"]["lat"], out["last_clicked"]["lng"])
        st.success(f"Location selected: {coords}")
        return coords
    return None

# --- Main App ---
st.title("📊 Retail Forecast & Insight Engine")
store = st.text_input("🏬 Enter Store Name")
coords = get_coords_from_store_name(store) if store else None
if not coords:
    coords = get_coords_from_map()

if coords:
    image = fetch_or_upload_satellite_image(coords)
    st.image(image, caption="Satellite View", use_container_width=True)

    if st.button("🔮 Predict Weekly Sales"):
        try:
            features, foot, soc = build_feature_vector(image, coords)
            model = load_model()
            pred = model.predict([features])[0]

            st.markdown(f"## 💰 Estimated Weekly Sales: **${pred:,.2f}**")
            save_prediction(store, coords, pred, foot, soc)
            plot_trends(store)

            st.subheader("📌 Strategic Recommendations")
            recs = []
            if foot < 0.4:
                recs.append("👣 Low foot traffic — improve signage, run geo-targeted mobile ads.")
            elif foot > 0.75:
                recs.append("💥 High foot traffic — use sidewalk promos and flash deals.")

            if soc < 35:
                recs.append("📱 Low social buzz — host a giveaway or collab with local influencers.")
            elif soc > 80:
                recs.append("🔥 High social buzz — push limited drops or referral bonuses now!")

            if "taco" in store.lower():
                recs.append("🌮 Best seller tip: stock $5 boxes, promote late-night combo deals.")
            elif store_type == "Coffee Shop":
                recs.append("☕ Offer loyalty cards & post seasonal drinks on IG reels.")
            elif store_type == "Boutique":
                recs.append("🛍️ Feature a 'look of the week' window display to convert foot traffic.")
            elif store_type == "Fast Food":
                recs.append("🍟 Test combo upgrades and upsell high-margin items like beverages.")

            for r in recs:
                st.markdown(f"- {r}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

