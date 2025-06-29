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

# Sidebar config and theme toggle
st.sidebar.title("⚙️ Settings")

store_type = st.sidebar.selectbox("Store Type", ["Any", "Coffee Shop", "Boutique", "Fast Food", "Other"])
theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = (theme == "Dark")

def set_theme():
    if st.session_state.dark_mode:
        # Inject dark mode CSS overrides
        st.markdown(
            """
            <style>
            .main {
                background-color: #121212;
                color: #e0e0e0;
            }
            .stButton>button {
                background-color: #333;
                color: white;
            }
            .sidebar .sidebar-content {
                background-color: #1e1e1e;
                color: #ddd;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .main, .sidebar .sidebar-content {
                background-color: white;
                color: black;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

set_theme()

if st.sidebar.button("Clear Sales History"):
    if os.path.exists("sales_history.csv"):
        os.remove("sales_history.csv")
        st.sidebar.success("Sales history cleared!")

# Ensure torchvision is available before importing
try:
    from torchvision import transforms
    from torchvision.models import resnet18
except ModuleNotFoundError as e:
    st.error("❌ Required module 'torchvision' not found. Please add it to requirements.txt:")
    st.code("torchvision")
    raise e

# 1. Fetch satellite image — STATIC placeholder to avoid NASA API error
def fetch_satellite_image(coords):
    # Placeholder satellite image from NASA (public domain)
    url = "https://eoimages.gsfc.nasa.gov/images/imagerecords/79000/79915/world.topo.bathy.200412.3x5400x2700.png"
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return image
    except Exception as e:
        st.error(f"Failed to fetch satellite image: {e}")
        # Return blank white image if fail
        return Image.new("RGB", (512, 512), color=(255, 255, 255))

# 2. Extract features using pretrained ResNet18
def extract_satellite_features(image):
    model = resnet18(pretrained=True)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(tensor)
    return features.numpy().flatten()

# 3. Placeholder foot traffic data (simulate)
def get_mock_foot_traffic_score(location_name):
    return np.random.uniform(0, 1)

# 4. Social sentiment using snscrape (Twitter/X)
def fetch_social_sentiment(lat, lon):
    try:
        import snscrape.modules.twitter as sntwitter
        from datetime import date, timedelta

        today = date.today()
        since = today - timedelta(days=7)
        query = f"near:{lat},{lon} within:1km since:{since}"
        tweets = list(sntwitter.TwitterSearchScraper(query).get_items())
        return min(len(tweets), 100)  # Limit to 100
    except:
        return np.random.randint(0, 100)

# 5. Build feature vector
def build_feature_vector(image, location_name, coords):
    sat_features = extract_satellite_features(image)
    foot_traffic = get_mock_foot_traffic_score(location_name)
    social = fetch_social_sentiment(*coords)
    return np.concatenate([sat_features, [foot_traffic, social]])

# 6. Load or generate model.pkl automatically
def load_model():
    from sklearn.ensemble import GradientBoostingRegressor
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    else:
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=500, n_features=513, noise=0.2, random_state=42)
        model = GradientBoostingRegressor(random_state=42).fit(X, y)
        joblib.dump(model, "model.pkl")
        return model

# 7. Save sales prediction history
sales_log = "sales_history.csv"
def save_prediction(store, coords, store_type_local, pred):
    df = pd.DataFrame([[store, coords[0], coords[1], store_type_local, pred]], columns=["store", "lat", "lon", "type", "sales"])
    if os.path.exists(sales_log):
        old = pd.read_csv(sales_log)
        new = pd.concat([old, df], ignore_index=True)
    else:
        new = df
    new.to_csv(sales_log, index=False)

def plot_trends(store):
    if not os.path.exists(sales_log):
        st.info("No historical data yet.")
        return
    df = pd.read_csv(sales_log)
    df = df[df.store.str.lower() == store.lower()]
    if df.empty:
        st.info("No historical data for this store.")
        return
    st.subheader(f"📊 Sales Trend for {store}")
    # Use index as pseudo time
    df["timestamp"] = pd.date_range(end=pd.Timestamp.today(), periods=len(df))
    df = df.sort_values("timestamp")
    plt.figure(figsize=(10, 4))
    plt.plot(df["timestamp"], df["sales"], marker='o', linestyle='-')
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.grid(True)
    st.pyplot(plt)

# 8. Main UI
st.title("🛍️ Retail Store Weekly Sales Forecast")

store = st.text_input("🏪 Enter store name")
location = st.text_input("📍 Enter coordinates (lat, lon)")

if st.button("Predict Weekly Sales"):
    if not store or not location:
        st.warning("Please enter both store name and coordinates.")
    else:
        try:
            coords = tuple(map(float, location.split(",")))
            image = fetch_satellite_image(coords)
            st.image(image, caption=f"🛰️ Satellite View of {store}", use_column_width=True)

            features = build_feature_vector(image, store, coords)
            model = load_model()
            prediction = model.predict([features])[0]

            st.markdown(f"### 📈 Predicted Weekly Sales: **${prediction:,.2f}**")

            save_prediction(store, coords, store_type, prediction)
            plot_trends(store)

            # Recommendations
            foot_traffic = features[-2]
            social = features[-1]
            st.subheader("🧠 Smart Recommendations")
            if foot_traffic < 0.3:
                st.warning("🚧 Low visibility: Consider adding signage or window displays.")
            if social < 20:
                st.info("📱 Minimal social buzz: Try a geo-tagged giveaway on Instagram or TikTok.")
            if foot_traffic > 0.7 and social > 60:
                st.success("🎯 High attention zone: Ideal time to upsell or promote bundles!")

        except Exception as e:
            st.error(f"Error: {e}")


