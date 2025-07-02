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
import plotly.express as px
from streamlit_folium import st_folium
import folium
from dotenv import load_dotenv
import torch.nn as nn
from geopy.geocoders import Nominatim

load_dotenv()

# --- Page Setup ---
st.set_page_config(page_title="Retail AI Platform", layout="wide")

# --- Sidebar ---
st.sidebar.title("🛠️ Settings")
store_type = st.sidebar.selectbox("Store Type", ["Any", "Coffee Shop", "Boutique", "Fast Food", "Other"])
theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = (theme == "Dark")

if st.sidebar.button("🧹 Clear Sales History"):
    if os.path.exists("sales_history.csv"):
        os.remove("sales_history.csv")
        st.sidebar.success("History cleared.")

# --- Theme ---
def set_theme():
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
        body, .main, .block-container, .sidebar .sidebar-content {
            background-color: #111827 !important;
            color: #e5e7eb !important;
        }
        .stButton>button {
            background-color: #6366f1;
            color: white;
        }
        </style>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stButton>button {
            background-color: #e5e7eb;
            color: #111827;
        }
        </style>""", unsafe_allow_html=True)
set_theme()

# --- Vision ---
try:
    from torchvision import transforms
    from torchvision.models import resnet18
except:
    st.error("Missing torchvision. Add it to requirements.txt")

def fetch_or_upload_satellite_image(coords):
    uploaded = st.file_uploader("Upload custom satellite image", type=["jpg", "jpeg", "png"])
    if uploaded:
        return Image.open(uploaded).convert("RGB")
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return Image.new("RGB", (512, 512), color=(200, 200, 200))
    try:
        url = f"https://maps.googleapis.com/maps/api/staticmap?center={coords[0]},{coords[1]}&zoom=17&size=600x400&maptype=satellite&key={api_key}"
        res = requests.get(url)
        res.raise_for_status()
        return Image.open(io.BytesIO(res.content)).convert("RGB")
    except:
        return Image.new("RGB", (512, 512), color=(160, 160, 160))

def extract_satellite_features(img):
    model = resnet18(pretrained=True)
    model.eval()
    model = nn.Sequential(*list(model.children())[:-1])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        return model(tensor).view(1, -1).numpy().flatten()

def get_safegraph_score(lat, lon):
    return np.random.uniform(0.4, 0.85)

def fetch_social_sentiment(lat, lon):
    return np.random.randint(35, 100)

def build_feature_vector(img, coords):
    return np.concatenate([
        extract_satellite_features(img),
        [get_safegraph_score(*coords), fetch_social_sentiment(*coords)]
    ]), get_safegraph_score(*coords), fetch_social_sentiment(*coords)

def load_model():
    from sklearn.ensemble import GradientBoostingRegressor
    if os.path.exists("model.pkl"):
        model = joblib.load("model.pkl")
        if getattr(model, 'n_features_in_', 0) != 514:
            raise ValueError("Model mismatch with features")
        return model
    X, y = make_regression(n_samples=100, n_features=514, noise=0.1)
    y = np.abs(y)
    model = GradientBoostingRegressor().fit(X, y)
    joblib.dump(model, "model.pkl")
    return model

def get_coords_from_store_name(name):
    geolocator = Nominatim(user_agent="retail_ai")
    try:
        matches = geolocator.geocode(name, exactly_one=False, limit=3, addressdetails=True)
        return [(m.latitude, m.longitude, m.address) for m in matches] if matches else []
    except:
        return []

def show_map_with_selection(options):
    st.subheader("📍 Select Your Store Location")
    m = folium.Map(location=[options[0][0], options[0][1]], zoom_start=14)
    for lat, lon, label in options:
        folium.Marker(location=[lat, lon], tooltip=label).add_to(m)
    result = st_folium(m, height=350, width=700)
    return options[0][:2] if options else None

def save_prediction(store, coords, pred, foot, soc):
    df = pd.DataFrame([[store, coords[0], coords[1], store_type, pred, foot, soc]],
                      columns=["store", "lat", "lon", "type", "sales", "foot", "social"])
    if os.path.exists("sales_history.csv"):
        old = pd.read_csv("sales_history.csv")
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv("sales_history.csv", index=False)

def plot_insights(store):
    if not os.path.exists("sales_history.csv"):
        return st.info("No data yet.")
    df = pd.read_csv("sales_history.csv")
    df = df[df["store"].astype(str).str.lower() == store.lower()]
    if df.empty:
        return st.warning("No data found.")
    df["timestamp"] = pd.date_range(end=pd.Timestamp.today(), periods=len(df))
    st.plotly_chart(px.line(df, x="timestamp", y="sales", title="Sales Over Time"))
    st.plotly_chart(px.area(df, x="timestamp", y=["foot", "social"], title="Foot Traffic & Social Buzz"))
    total = pd.read_csv("sales_history.csv")
    avg_type = total.groupby("type")["sales"].mean().reset_index()
    st.plotly_chart(px.bar(avg_type, x="type", y="sales", title="Avg Sales by Store Type"))
    st.plotly_chart(px.pie(df, names="type", values="sales", title="Store Type Distribution"))

# --- App ---
st.title("📈 Retail AI: Forecast, Benchmarking & Strategy")
store = st.text_input("🏪 Store Name (e.g. Taco Bell Robbinsville)")
coords = None

if store:
    candidates = get_coords_from_store_name(store)
    if candidates:
        st.success("Pick your exact store from below:")
        coords = show_map_with_selection(candidates)
    else:
        st.warning("Could not geolocate. Please pick on map below.")

if not coords:
    coords = show_map_with_selection([(40.7128, -74.0060, "New York (default)")])

if coords:
    image = fetch_or_upload_satellite_image(coords)
    st.image(image, caption="🛰️ Satellite View", use_container_width=True)

    if st.button("📊 Predict & Analyze"):
        try:
            features, foot, soc = build_feature_vector(image, coords)
            model = load_model()
            pred = max(model.predict([features])[0], 0)
            st.markdown(f"## 💰 Predicted Weekly Sales: **${pred:,.2f}**")
            save_prediction(store, coords, pred, foot, soc)
            plot_insights(store)

            st.subheader("📦 Actionable Strategy & Inventory Advice")
            recs = []
            if foot < 0.4:
                recs.append("🛑 Low traffic: Offer in-store exclusive deals + signage visibility upgrades.")
            elif foot > 0.7:
                recs.append("🚦 High traffic: Consider fast checkout stations or bundle pricing.")
            if soc < 35:
                recs.append("📉 Low buzz: Start TikTok or Reels challenges tagged locally.")
            elif soc > 70:
                recs.append("🔥 Trending: Leverage influencer promo codes for new customers.")

            if store_type == "Fast Food" or "taco" in store.lower():
                recs.append("🌮 Taco Insight: Boost inventory for beef/chicken combo SKUs Fri-Sun.")
                recs.append("📊 Popular hours: 11:30am–1pm & 6pm–8pm — plan staff accordingly.")
                recs.append("📦 Consider pre-packing popular $5 box items to reduce order time.")

            if not recs:
                recs.append("🧠 Tip: Add loyalty punch card + QR-based feedback system.")

            for r in recs:
                st.markdown(f"- {r}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

