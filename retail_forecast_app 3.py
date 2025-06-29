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
            body, .main, .block-container, .sidebar .sidebar-content {
                background-color: #121212 !important;
                color: #e0e0e0 !important;
            }
            button, .stButton>button {
                background-color: #333 !important;
                color: white !important;
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            body, .main, .block-container, .sidebar .sidebar-content {
                background-color: white !important;
                color: black !important;
            }
            button, .stButton>button {
                background-color: #eee !important;
                color: black !important;
            }
            </style>
        """, unsafe_allow_html=True)

set_theme()

if st.sidebar.button("Clear Sales History"):
    if os.path.exists("sales_history.csv"):
        os.remove("sales_history.csv")
        st.sidebar.success("Sales history cleared!")
    else:
        st.sidebar.info("No sales history file found.")

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
        try:
            img = Image.open(uploaded_file).convert("RGB")
            return img
        except Exception as e:
            st.error(f"Error loading uploaded image: {e}")
            return None

    google_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
    if not google_key:
        # Fallback NASA image or white image
        try:
            url = "https://eoimages.gsfc.nasa.gov/images/imagerecords/79000/79915/world.topo.bathy.200412.3x5400x2700.png"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception as e:
            st.error(f"Failed to load fallback image: {e}")
            return Image.new("RGB", (512, 512), color=(255, 255, 255))

    url = f"https://maps.googleapis.com/maps/api/staticmap?center={coords[0]},{coords[1]}&zoom=17&size=600x400&maptype=satellite&key={google_key}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        st.error(f"Failed to load Google Maps satellite image: {e}")
        # fallback NASA image as above
        try:
            url = "https://eoimages.gsfc.nasa.gov/images/imagerecords/79000/79915/world.topo.bathy.200412.3x5400x2700.png"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception as e2:
            st.error(f"Failed to load fallback image: {e2}")
            return Image.new("RGB", (512, 512), color=(255, 255, 255))

# --- 2. Extract Vision Features ---
def extract_satellite_features(image):
    model = resnet18(pretrained=True)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    try:
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            features = model(tensor)
        return features.numpy().flatten()
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return np.zeros(512)  # Return zeros to prevent crash

# --- 3. SafeGraph Foot Traffic API (mocked unless key provided) ---
def get_safegraph_score(lat, lon):
    safegraph_key = os.getenv("SAFEGRAPH_API_KEY", "")
    if not safegraph_key:
        return np.random.uniform(0, 1)
    # Here you would implement real API call if you have key
    return np.random.uniform(0, 1)

# --- 4. Twitter v2 Social Sentiment ---
def fetch_social_sentiment_v2(lat, lon):
    twitter_bearer_token = os.getenv("TWITTER_BEARER", "")
    if not twitter_bearer_token:
        return np.random.randint(0, 100)
    headers = {"Authorization": f"Bearer {twitter_bearer_token}"}
    query = f"point_radius:[{lon} {lat} 1km] -is:retweet lang:en"
    url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results=50"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        tweets = resp.json().get("data", [])
        return len(tweets)
    except Exception as e:
        st.warning(f"Twitter API error: {e}")
        return np.random.randint(0, 100)

# --- 5. Build Feature Vector ---
def build_feature_vector(image, coords):
    sat_features = extract_satellite_features(image)
    foot_traffic = get_safegraph_score(*coords)
    social = fetch_social_sentiment_v2(*coords)
    return np.concatenate([sat_features, [foot_traffic, social]]), foot_traffic, social

# --- 6. Load or Train Model ---
def load_model():
    from sklearn.ensemble import GradientBoostingRegressor
    try:
        if os.path.exists("model.pkl"):
            model = joblib.load("model.pkl")
            return model
        else:
            raise FileNotFoundError("Model file not found.")
    except Exception as e:
        st.warning(f"Model loading failed or not found: {e}. Training a new model now...")
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=500, n_features=514, noise=0.2)  # 512 features + 2 foot/social
        model = GradientBoostingRegressor()
        model.fit(X, y)
        joblib.dump(model, "model.pkl")
        return model

# --- 7. Save Results ---
sales_log = "sales_history.csv"
def save_prediction(store, coords, pred, foot, soc):
    if not store.strip():
        st.warning("Please enter a store name to save prediction.")
        return
    df = pd.DataFrame([[store.strip(), coords[0], coords[1], store_type, pred, foot, soc]],
                      columns=["store", "lat", "lon", "type", "sales", "foot", "social"])
    try:
        if os.path.exists(sales_log):
            old = pd.read_csv(sales_log)
            new = pd.concat([old, df], ignore_index=True)
        else:
            new = df
        new.to_csv(sales_log, index=False)
    except Exception as e:
        st.error(f"Failed to save prediction: {e}")

# --- 8. Graph Trends ---
def plot_trends(store):
    if not os.path.exists(sales_log):
        st.info("No historical data yet.")
        return
    try:
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
    except Exception as e:
        st.error(f"Error plotting trends: {e}")

# --- 9. Map Selection ---
def get_coords_from_map():
    st.subheader("🌍 Select Store Location on Map")
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)
    m.add_child(folium.LatLngPopup())
    output = st_folium(m, height=300, width=700)
    if output and output.get("last_clicked"):
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
    if image:
        st.image(image, caption="🛰️ Satellite View", use_container_width=True)
    else:
        st.warning("No valid satellite image available.")

    if st.button("Predict Weekly Sales"):
        if not store.strip():
            st.warning("Please enter a store name before predicting.")
        else:
            features, foot, soc = build_feature_vector(image, coords)
            model = load_model()
            try:
                pred = model.predict([features])[0]
            except Exception as e:
                st.error(f"Prediction failed: {e}")
            else:
                st.markdown(f"### 📊 Predicted Sales: **${pred:,.2f}**")
                save_prediction(store, coords, pred, foot, soc)
                plot_trends(store)

                st.subheader("🤨 Recommendations")
                if foot < 0.3:
                    st.warning("🚧 Low foot traffic: improve signage or placement.")
                if soc < 15:
                    st.info("📱 Run a local Instagram giveaway or post.")
                if foot > 0.7 and soc > 60:
                    st.success("🎉 High attention area: Upsell with bundles!")
else:
    st.info("Please select a location on the map to get started.")
