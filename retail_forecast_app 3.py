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

# Ensure torchvision is available before importing
try:
    from torchvision import transforms
    from torchvision.models import resnet18
except ModuleNotFoundError as e:
    st.error("❌ Required module 'torchvision' not found. Please add it to requirements.txt:")
    st.code("torchvision")
    raise e

# 1. Fetch satellite image (NASA API for demo)
def fetch_satellite_image(coords):
    lat, lon = coords
    nasa_api_key = "DEMO_KEY"  # Replace with your NASA API key
    metadata_url = (
        f"https://api.nasa.gov/planetary/earth/assets?lon={lon}&lat={lat}&dim=0.1&api_key={nasa_api_key}"
    )
    meta_response = requests.get(metadata_url)
    if meta_response.status_code != 200:
        st.error("Failed to fetch satellite metadata.")
        print("Metadata response:", meta_response.text)
        raise Exception("Satellite metadata error")
    image_url = meta_response.json().get("url")
    if not image_url:
        raise Exception("No image URL in response")
    image_response = requests.get(image_url)
    if image_response.status_code != 200:
        raise Exception("Failed to download image")
    image = Image.open(io.BytesIO(image_response.content)).convert("RGB")
    return image

# 2. Extract features from satellite image using pretrained ResNet18
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

# 3. Placeholder for foot traffic data
def get_mock_foot_traffic_score(location_name):
    return np.random.uniform(0, 1)

# 4. Live social media sentiment (simulated Twitter search count)
def fetch_social_sentiment(lat, lon):
    try:
        import snscrape.modules.twitter as sntwitter
        from datetime import date, timedelta

        today = date.today()
        since = today - timedelta(days=7)
        query = f"near:{lat},{lon} within:1km since:{since}"
        tweets = list(sntwitter.TwitterSearchScraper(query).get_items())
        return min(len(tweets), 100)  # Limit to 100 for scale
    except:
        return np.random.randint(0, 100)

# 5. Build feature vector
def build_feature_vector(image, location_name, coords):
    sat_features = extract_satellite_features(image)
    foot_traffic = get_mock_foot_traffic_score(location_name)
    social = fetch_social_sentiment(*coords)
    return np.concatenate([sat_features, [foot_traffic, social]])

# 6. Load or generate synthetic model.pkl
def load_model():
    from sklearn.ensemble import GradientBoostingRegressor
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    else:
        # Generate dummy model
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=500, n_features=513, noise=0.2)
        model = GradientBoostingRegressor().fit(X, y)
        joblib.dump(model, "model.pkl")
        return model

# 7. Store sales history
sales_log = "sales_history.csv"
def save_prediction(store, coords, pred):
    df = pd.DataFrame([[store, coords[0], coords[1], pred]], columns=["store", "lat", "lon", "sales"])
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
    df = df[df.store == store]
    if df.empty:
        st.info("No historical data for this store.")
        return
    st.subheader("📊 Sales Trend for " + store)
    df["timestamp"] = pd.date_range(end=pd.Timestamp.today(), periods=len(df))
    st.line_chart(df.set_index("timestamp")["sales"])

# 8. Streamlit interface
st.title("Retail Store Weekly Sales Forecast")
store = st.text_input("Enter store name")
location = st.text_input("Enter coordinates (lat, lon)")

if st.button("Predict Weekly Sales"):
    try:
        coords = tuple(map(float, location.split(",")))
        image = fetch_satellite_image(coords)
        st.image(image, caption="Satellite View")
        features = build_feature_vector(image, store, coords)
        model = load_model()
        prediction = model.predict([features])[0]

        st.success(f"📈 Predicted Weekly Sales: ${prediction:,.2f}")

        save_prediction(store, coords, prediction)
        plot_trends(store)

        # Recommend actions
        foot_traffic = features[-2]
        social = features[-1]
        if foot_traffic < 0.3:
            st.warning("⚠️ Consider improving signage or exterior displays to increase foot traffic.")
        if social < 20:
            st.info("💡 Try running a local Instagram promo or hashtag contest to build buzz.")

    except Exception as e:
        st.error(f"Error: {e}")


