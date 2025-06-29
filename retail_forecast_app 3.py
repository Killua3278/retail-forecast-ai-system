# retail_forecast_app.py
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

# --- Sidebar Settings ---
st.sidebar.title("⚙️ Settings")
store_type = st.sidebar.selectbox("Store Type", ["Any", "Coffee Shop", "Boutique", "Fast Food", "Other"])
theme = st.sidebar.radio("Theme", ["Light", "Dark"])

# Theme Styling
def apply_theme(theme):
    if theme == "Dark":
        st.markdown(
            """
            <style>
            .main { background-color: #121212; color: #e0e0e0; }
            .stButton>button { background-color: #333; color: white; }
            .sidebar .sidebar-content { background-color: #1e1e1e; color: #ddd; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .main, .sidebar .sidebar-content { background-color: white; color: black; }
            </style>
            """,
            unsafe_allow_html=True,
        )

apply_theme(theme)

# Clear Sales History
if st.sidebar.button("Clear Sales History"):
    if os.path.exists("sales_history.csv"):
        os.remove("sales_history.csv")
        st.sidebar.success("Sales history cleared!")

# Torchvision Model Import
try:
    from torchvision import transforms
    from torchvision.models import resnet18
except:
    st.error("torchvision is missing. Add it to requirements.txt")
    raise

# Satellite Image Upload/Fallback
def fetch_satellite_image():
    uploaded_file = st.file_uploader("Upload a Satellite Image", type=["jpg", "png"])
    if uploaded_file:
        return Image.open(uploaded_file).convert("RGB")
    else:
        fallback_url = "https://eoimages.gsfc.nasa.gov/images/imagerecords/79000/79915/world.topo.bathy.200412.3x5400x2700.png"
        response = requests.get(fallback_url)
        return Image.open(io.BytesIO(response.content)).convert("RGB")

# Feature Extraction
@st.cache_resource
def load_cnn_model():
    model = resnet18(pretrained=True)
    model.eval()
    return model

def extract_satellite_features(image):
    model = load_cnn_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(tensor)
    return features.numpy().flatten()

# Social Sentiment
def fetch_social_sentiment(lat, lon):
    try:
        import snscrape.modules.twitter as sntwitter
        from datetime import date, timedelta

        today = date.today()
        since = today - timedelta(days=7)
        query = f"near:{lat},{lon} within:1km since:{since}"
        tweets = list(sntwitter.TwitterSearchScraper(query).get_items())
        return min(len(tweets), 100)
    except:
        return np.random.randint(0, 100)

# Mock Foot Traffic
def get_mock_foot_traffic_score():
    return np.random.uniform(0, 1)

# Build Feature Vector
def build_feature_vector(image, coords):
    sat = extract_satellite_features(image)
    foot = get_mock_foot_traffic_score()
    social = fetch_social_sentiment(*coords)
    return np.concatenate([sat, [foot, social]]), foot, social

# Load or Create Model
def load_model():
    from sklearn.ensemble import GradientBoostingRegressor
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    else:
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=500, n_features=513, noise=0.1)
        model = GradientBoostingRegressor().fit(X, y)
        joblib.dump(model, "model.pkl")
        return model

# Save Prediction
sales_log = "sales_history.csv"
def save_prediction(store, coords, pred):
    df = pd.DataFrame([[store, coords[0], coords[1], store_type, pred]], columns=["store", "lat", "lon", "type", "sales"])
    if os.path.exists(sales_log):
        df = pd.concat([pd.read_csv(sales_log), df], ignore_index=True)
    df.to_csv(sales_log, index=False)

# Plot Trends
def plot_trends(store):
    if os.path.exists(sales_log):
        df = pd.read_csv(sales_log)
        df = df[df.store.str.lower() == store.lower()]
        if not df.empty:
            st.subheader("📊 Sales Trend")
            df["timestamp"] = pd.date_range(end=pd.Timestamp.today(), periods=len(df))
            st.line_chart(df.set_index("timestamp")["sales"])

# UI
st.title("🛍️ Retail Sales Forecaster")

m = folium.Map(location=[40.7128, -74.0060], zoom_start=12)
marker = folium.Marker(location=[40.7128, -74.0060], draggable=True)
marker.add_to(m)
output = st_folium(m, height=400)

latlon = output.get("last_clicked") or output.get("center")

store = st.text_input("🏪 Store Name")
if st.button("📈 Predict Sales"):
    if not store or not latlon:
        st.warning("Please enter a store name and choose a location on the map.")
    else:
        coords = (latlon["lat"], latlon["lng"])
        image = fetch_satellite_image()
        st.image(image, caption="🛰️ Satellite View", use_container_width=True)

        features, foot, social = build_feature_vector(image, coords)
        model = load_model()
        prediction = model.predict([features])[0]

        st.success(f"Predicted Weekly Sales: ${prediction:,.2f}")
        save_prediction(store, coords, prediction)
        plot_trends(store)

        st.subheader("🧠 Smart Suggestions")
        if foot < 0.3:
            st.warning("🚧 Low foot traffic. Consider signage improvements.")
        if social < 20:
            st.info("📱 Low buzz. Try a geo-tagged social promo.")
        if foot > 0.7 and social > 60:
            st.success("🎯 Great momentum. Run a flash sale!")



