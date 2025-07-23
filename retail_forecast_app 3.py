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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor

load_dotenv()

# --- Page Setup ---
st.set_page_config(page_title="Retail AI Platform", layout="wide")

# --- Sidebar ---
st.sidebar.title("🛠️ Settings")
store_type = st.sidebar.selectbox("Store Type", ["Any", "Coffee Shop", "Boutique", "Fast Food", "Other"])
theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = (theme == "Dark")

if st.sidebar.button("🩹 Clear Sales History"):
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
        features = model(tensor).view(1, -1).numpy().flatten()
    features = np.resize(features, 512)  # Ensure exactly 512
    return features

def get_safegraph_score(lat, lon):
    return np.random.uniform(0.4, 0.85)

def fetch_social_sentiment(lat, lon):
    return np.random.randint(35, 100)

def build_feature_vector(img, coords):
    foot_score = get_safegraph_score(*coords)
    social_score = fetch_social_sentiment(*coords)
    satellite_features = extract_satellite_features(img)
    features = np.concatenate([satellite_features, [coords[0], coords[1]]])
    return features, foot_score, social_score

def get_coords_from_store_name(name):
    try:
        geolocator = Nominatim(user_agent="retail_ai_locator")
        location = geolocator.geocode(name)
        if location:
            return [(location.latitude, location.longitude, location.address)]
        return []
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
    df = pd.DataFrame([[store, coords[0], coords[1], store_type, pred, foot, soc, pd.Timestamp.now()]],
                      columns=["store", "lat", "lon", "type", "sales", "foot", "social", "timestamp"])
    if os.path.exists("sales_history.csv"):
        try:
            old = pd.read_csv("sales_history.csv", on_bad_lines='skip')
            df = pd.concat([old, df], ignore_index=True)
        except:
            st.warning("Corrupted history file. Overwriting.")
    df.to_csv("sales_history.csv", index=False)

def plot_insights(store):
    if not os.path.exists("sales_history.csv"):
        return st.info("No data yet.")
    try:
        df = pd.read_csv("sales_history.csv", on_bad_lines='skip')
    except:
        return st.warning("Could not read history file.")
    df = df[df["store"].astype(str).str.lower() == store.lower()]
    if df.empty:
        return st.warning("No data found.")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    st.plotly_chart(px.line(df, x="timestamp", y="sales", title="Sales Over Time"))
    st.plotly_chart(px.area(df, x="timestamp", y=["foot", "social"], title="Foot Traffic & Social Buzz"))
    total = pd.read_csv("sales_history.csv", on_bad_lines='skip')
    avg_type = total.groupby("type")["sales"].mean().reset_index()
    st.plotly_chart(px.bar(avg_type, x="type", y="sales", title="Avg Sales by Store Type"))
    st.plotly_chart(px.pie(df, names="type", values="sales", title="Store Type Distribution"))

def generate_recommendations(store, store_type, foot, soc, sales):
    recs = []
    if foot < 0.4:
        recs.append("🔻 Foot traffic is low. Consider placing geofenced mobile ads or joining local delivery platforms.")
    elif foot < 0.6:
        recs.append("🚶‍♂️ Average foot traffic. Set up sidewalk signage or window displays to increase walk-ins.")
    else:
        recs.append("🚦 High foot traffic detected. Promote limited-time bundles and impulse buys.")
    if soc < 40:
        recs.append("📉 Low social media activity. Post behind-the-scenes videos, reviews, and tag your location on Instagram.")
    elif soc < 70:
        recs.append("📱 Moderate buzz. Use hashtag campaigns and stories to boost daily engagement.")
    else:
        recs.append("📢 High buzz! Launch influencer deals or flash discounts to ride the momentum.")
    if "taco" in store.lower() or "bell" in store.lower() or store_type == "Fast Food":
        recs.extend([
            "🌮 *Fast Food Strategy*: Stock popular items like Cravings Boxes and combo meals between 12–2pm & 6–8pm.",
            "📦 Use pre-prepared ingredients during peak hours to cut wait times.",
            "📊 Test digital order kiosks or loyalty app promotions."
        ])
    if sales > 30000:
        recs.append("📈 Sales are strong! Evaluate expanding inventory or testing higher-margin products.")
    elif sales < 10000:
        recs.append("🔍 Underperforming sales. Benchmark competitors in the area using foot traffic + buzz to identify gaps.")
    recs.append("🧠 Pro Tip: Use customer purchase history (even manually tracked) to promote repeat buying patterns.")
    return recs

# --- Model loading and fallback ---
def load_fallback_model():
    dummy = DummyRegressor(strategy="mean")
    dummy.fit([[0]*514], [10000])
    return dummy

def load_real_data_model():
    real_data_path = "real_sales_data.csv"
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    if not os.path.exists(real_data_path):
        st.warning("Real dataset not found. Using fallback regression model.")
        return load_fallback_model()
    try:
        df = pd.read_csv(real_data_path, on_bad_lines='skip')
    except:
        st.error("Error reading real_sales_data.csv")
        return load_fallback_model()
    expected_features = [f"f{i}" for i in range(512)] + ["lat", "lon"]
    if not all(col in df.columns for col in expected_features + ["sales"]):
        st.error("real_sales_data.csv must have 512 features (f0 to f511), lat, lon, and sales columns")
        return load_fallback_model()
    X = df[expected_features]
    y = df["sales"]
    model = GradientBoostingRegressor()
    model.fit(X, y)
    joblib.dump(model, "model.pkl")
    st.success("Real dataset-based model trained and loaded ✅")
    return model

# ACTUAL LOAD MODEL INSTANCE
load_model = load_real_data_model()

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
    st.image(image, caption="🧪 Satellite View", use_container_width=True)

    if st.button("📊 Predict & Analyze"):
        try:
            features, foot, soc = build_feature_vector(image, coords)
            pred = max(load_model.predict([features])[0], 0)
            st.markdown(f"## 💰 Predicted Weekly Sales: **${pred:,.2f}**")
            save_prediction(store, coords, pred, foot, soc)
            plot_insights(store)
            st.subheader("📦 Actionable Strategy & Inventory Advice")
            recs = generate_recommendations(store, store_type, foot, soc, pred)
            for r in recs:
                st.markdown(f"- {r}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")



