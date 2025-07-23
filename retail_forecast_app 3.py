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
import plotly.express as px
from streamlit_folium import st_folium
import folium
from dotenv import load_dotenv
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from geopy.geocoders import Nominatim
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor

load_dotenv()

st.set_page_config(page_title="Retail AI Platform", layout="wide")

# --- Sidebar ---
st.sidebar.title("🔧 Settings")
store_type = st.sidebar.selectbox("Store Type", ["Any", "Coffee Shop", "Boutique", "Fast Food", "Other"])
theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=1)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = (theme == "Dark")

if st.sidebar.button("🩹 Clear Sales History"):
    if os.path.exists("sales_history.csv"):
        os.remove("sales_history.csv")
        st.sidebar.success("History cleared.")

# --- Theme Setup ---
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

# --- Utilities ---
def fetch_or_upload_satellite_image(coords):
    uploaded = st.file_uploader("Upload custom satellite image", type=["jpg", "jpeg", "png"])
    if uploaded:
        return Image.open(uploaded).convert("RGB")
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        st.error("Missing Google Maps API key")
        return Image.new("RGB", (512, 512), color=(200, 200, 200))
    try:
        url = f"https://maps.googleapis.com/maps/api/staticmap?center={coords[0]},{coords[1]}&zoom=18&size=600x400&maptype=satellite&key={api_key}"
        res = requests.get(url)
        res.raise_for_status()
        return Image.open(io.BytesIO(res.content)).convert("RGB")
    except Exception as e:
        st.error(f"Satellite image fetch error: {e}")
        return Image.new("RGB", (512, 512), color=(160, 160, 160))

def extract_satellite_features(img):
    model = resnet18(pretrained=True)
    model.eval()
    model = nn.Sequential(*list(model.children())[:-1])
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = model(tensor).view(1, -1).numpy().flatten()
    return np.resize(features, 512)

def get_safegraph_score(lat, lon):
    return np.random.uniform(0.4, 0.85)

def fetch_social_sentiment(lat, lon):
    return np.random.randint(35, 100)

def build_feature_vector(img, coords):
    return np.concatenate([
        extract_satellite_features(img), [coords[0], coords[1]]
    ]), get_safegraph_score(*coords), fetch_social_sentiment(*coords)

def get_coords_from_store_name(name, zip_code):
    try:
        query = f"{name}, {zip_code}" if zip_code else name
        geolocator = Nominatim(user_agent="retail_ai_locator")
        location = geolocator.geocode(query)
        if location:
            return [(location.latitude, location.longitude, location.address)]
    except:
        pass
    return []

def show_map_with_selection(options):
    st.subheader("📍 Select Your Store Location")
    m = folium.Map(location=[options[0][0], options[0][1]], zoom_start=14)
    for lat, lon, label in options:
        folium.Marker(location=[lat, lon], tooltip=label).add_to(m)
    st_folium(m, height=350, width=700)
    return options[0][:2]

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
        recs.append("🔻 Low foot traffic. Consider mobile ads or joining delivery platforms.")
    elif foot < 0.6:
        recs.append("🚶‍♂️ Average traffic. Use signage or in-store events.")
    else:
        recs.append("🚦 High traffic. Push time-limited combos or flash sales.")
    if soc < 40:
        recs.append("📉 Weak social presence. Post reels, behind-the-scenes, promos.")
    elif soc < 70:
        recs.append("📱 Moderate engagement. Add stories and location tags.")
    else:
        recs.append("📢 Great buzz. Offer referral or influencer rewards.")
    if store_type == "Fast Food" or "taco" in store.lower():
        recs.append("🌮 Optimize peak lunch and dinner rush with ready-to-go inventory.")
    if sales > 30000:
        recs.append("📈 High sales. Test new premium items or expand inventory.")
    elif sales < 10000:
        recs.append("🔍 Low performance. Benchmark neighbors, improve window appeal.")
    recs.append("🧠 Tip: Track customer favorites, even on paper, to personalize deals.")
    return recs

def load_fallback_model():
    dummy = DummyRegressor(strategy="mean")
    dummy.fit([[0]*514], [10000])
    return dummy

def load_real_data_model():
    real_data_path = "real_sales_data.csv"
    expected_cols = [f"f{i}" for i in range(512)] + ["lat", "lon", "sales"]
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    if not os.path.exists(real_data_path):
        st.warning("real_sales_data.csv not found. Using fallback model.")
        return load_fallback_model()
    try:
        df = pd.read_csv(real_data_path)
        if not all(col in df.columns for col in expected_cols):
            st.error("real_sales_data.csv missing required columns")
            return load_fallback_model()
        X = df[[f"f{i}" for i in range(512)] + ["lat", "lon"]]
        y = df["sales"]
        model = GradientBoostingRegressor()
        model.fit(X, y)
        joblib.dump(model, "model.pkl")
        return model
    except Exception as e:
        st.error(f"Model load error: {e}")
        return load_fallback_model()

# --- App Execution ---
st.title("📈 Retail AI: Forecast & Strategy")
store = st.text_input("🏪 Store Name (e.g. Dave's Hot Chicken)")
zip_code = st.text_input("📍 ZIP Code (optional)")
coords = None

if store:
    candidates = get_coords_from_store_name(store, zip_code)
    if candidates:
        coords = show_map_with_selection(candidates)
    else:
        st.warning("Location not found. Defaulting to New York.")

if not coords:
    coords = show_map_with_selection([(40.7128, -74.0060, "New York (default)")])

if coords:
    image = fetch_or_upload_satellite_image(coords)
    st.image(image, caption="🧪 Satellite View", use_container_width=True)

    if st.button("📊 Predict & Analyze"):
        try:
            features, foot, soc = build_feature_vector(image, coords)
            model = load_real_data_model()
            pred = max(model.predict([features])[0], 0)
            st.markdown(f"## 💰 Predicted Weekly Sales: **${pred:,.2f}**")
            save_prediction(store, coords, pred, foot, soc)
            plot_insights(store)
            st.subheader("📦 Strategy Recommendations")
            for r in generate_recommendations(store, store_type, foot, soc, pred):
                st.markdown(f"- {r}")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
