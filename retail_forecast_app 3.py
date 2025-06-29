# --- Imports and Setup ---
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

# --- Sidebar Config ---
st.sidebar.title("⚙️ Settings")
store_type = st.sidebar.selectbox("Store Type", ["Any", "Coffee Shop", "Boutique", "Fast Food", "Other"])
theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=0)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = (theme == "Dark")

def set_theme():
    if st.session_state.dark_mode:
        st.markdown("""
            <style>
            .main { background-color: #121212; color: #e0e0e0; }
            .stButton>button { background-color: #333; color: white; }
            .sidebar .sidebar-content { background-color: #1e1e1e; color: #ddd; }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .main, .sidebar .sidebar-content { background-color: white; color: black; }
            </style>
        """, unsafe_allow_html=True)
set_theme()

if st.sidebar.button("Clear Sales History"):
    if os.path.exists("sales_history.csv"):
        os.remove("sales_history.csv")
        st.sidebar.success("Sales history cleared!")

# --- Imports for vision ---
try:
    from torchvision import transforms
    from torchvision.models import resnet18
except ModuleNotFoundError as e:
    st.error("Required module 'torchvision' not found. Please add it to requirements.txt:")
    st.code("torchvision")
    raise e

# --- 1. Upload or Fetch Satellite Image ---
def fetch_or_upload_satellite_image(coords):
    uploaded_file = st.file_uploader("Or upload a satellite image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        return Image.open(uploaded_file).convert("RGB")

    # Default fallback NASA placeholder
    url = "https://eoimages.gsfc.nasa.gov/images/imagerecords/79000/79915/world.topo.bathy.200412.3x5400x2700.png"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except:
        st.error("Failed to load fallback image.")
        return Image.new("RGB", (512, 512), color=(255, 255, 255))

# --- 2. Extract Vision Features ---
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

# --- 3. SafeGraph Foot Traffic API (mocked unless key provided) ---
def get_safegraph_score(lat, lon):
    safegraph_key = os.getenv("SAFEGRAPH_API_KEY")
    if not safegraph_key:
        return np.random.uniform(0, 1)
    # Placeholder for actual SafeGraph call
    # Here you would call your endpoint with lat/lon and extract foot traffic info
    return np.random.uniform(0, 1)

# --- 4. Twitter v2 Social Sentiment ---
def fetch_social_sentiment_v2(lat, lon):
    twitter_bearer_token = os.getenv("TWITTER_BEARER")
    if not twitter_bearer_token:
        return np.random.randint(0, 100)
    headers = {"Authorization": f"Bearer {twitter_bearer_token}"}
    query = f"point_radius:[{lon} {lat} 1km] -is:retweet lang:en"
    url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results=50"
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        return np.random.randint(0, 100)
    tweets = resp.json().get("data", [])
    return len(tweets)

# --- 5. Build Feature Vector ---
def build_feature_vector(image, coords):
    sat_features = extract_satellite_features(image)
    foot_traffic = get_safegraph_score(*coords)
    social = fetch_social_sentiment_v2(*coords)
    return np.concatenate([sat_features, [foot_traffic, social]]), foot_traffic, social

# --- 6. Load or Train Model ---
def load_model():
    from sklearn.ensemble import GradientBoostingRegressor
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    else:
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=500, n_features=513, noise=0.2)
        model = GradientBoostingRegressor().fit(X, y)
        joblib.dump(model, "model.pkl")
        return model

# --- 7. Save Results ---
sales_log = "sales_history.csv"
def save_prediction(store, coords, pred, foot, soc):
    df = pd.DataFrame([[store, coords[0], coords[1], store_type, pred, foot, soc]],
                      columns=["store", "lat", "lon", "type", "sales", "foot", "social"])
    if os.path.exists(sales_log):
        old = pd.read_csv(sales_log)
        new = pd.concat([old, df], ignore_index=True)
    else:
        new = df
    new.to_csv(sales_log, index=False)

# --- 8. Graph Trends ---
def plot_trends(store):
    if not os.path.exists(sales_log):
        st.info("No historical data yet.")
        return
    df = pd.read_csv(sales_log)
    df = df[df.store.str.lower() == store.lower()]
    if df.empty:
        st.info("No data for this store.")
        return
    df["timestamp"] = pd.date_range(end=pd.Timestamp.today(), periods=len(df))
    df = df.sort_values("timestamp")
    plt.figure(figsize=(10, 4))
    plt.plot(df["timestamp"], df["sales"], label="Sales")
    plt.plot(df["timestamp"], df["foot"] * 1000, label="Foot Traffic (scaled)")
    plt.plot(df["timestamp"], df["social"] * 10, label="Social Buzz (scaled)")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# --- 9. Map Selection ---
def get_coords_from_map():
    st.subheader("🌍 Select Store Location on Map")
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
    loc_marker = folium.LatLngPopup()
    m.add_child(loc_marker)
    output = st_folium(m, height=300, width=700)
    if output.get("last_clicked"):
        coords = (output["last_clicked"]["lat"], output["last_clicked"]["lng"])
        st.success(f"Selected Location: {coords}")
        return coords
    return None

# --- Main Interface ---
st.title("🏣 Retail Store Forecast Platform")
store = st.text_input("🏥 Store Name")
coords = get_coords_from_map()

if coords:
    image = fetch_or_upload_satellite_image(coords)
    st.image(image, caption="🛰️ Satellite View", use_container_width=True)

    if st.button("Predict Weekly Sales"):
        features, foot, soc = build_feature_vector(image, coords)
        model = load_model()
        pred = model.predict([features])[0]

        st.markdown(f"### 📊 Predicted Sales: **${pred:,.2f}**")
        save_prediction(store, coords, pred, foot, soc)
        plot_trends(store)

        st.subheader("🧐 Recommendations")
        if foot < 0.3:
            st.warning("🚧 Low foot traffic: improve signage or placement.")
        if soc < 15:
            st.info("📱 Run a local Instagram giveaway or post.")
        if foot > 0.7 and soc > 60:
            st.success("🎉 High attention area: Upsell with bundles!")




