# ✅ Fully Integrated Retail AI App with Yelp, ZIP, Satellite, and Strategy Intelligence + Improved Visualizations + Advanced Recommendations

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
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
import time
from bs4 import BeautifulSoup
from math import radians, sin, cos, asin, sqrt  # <-- added
import random  # <-- added

load_dotenv()

st.set_page_config(page_title="Retail AI Platform", layout="wide")

# --- API Keys ---
YELP_API_KEY = os.getenv("YELP_API_KEY")
YELP_HEADERS = {"Authorization": f"Bearer {YELP_API_KEY}"} if YELP_API_KEY else {}

# --- Sidebar ---
st.sidebar.title("🔧 Settings")
store_type = st.sidebar.selectbox("Store Type", ["Any", "Coffee Shop", "Boutique", "Fast Food", "Other"])
theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=1)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = (theme == "Dark")

if st.sidebar.button("🥩 Clear Sales History"):
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

# --- Geo/units helpers (NEW) ---
STATE_ABBR = {
    "alabama":"AL","alaska":"AK","arizona":"AZ","arkansas":"AR","california":"CA","colorado":"CO","connecticut":"CT",
    "delaware":"DE","florida":"FL","georgia":"GA","hawaii":"HI","idaho":"ID","illinois":"IL","indiana":"IN","iowa":"IA",
    "kansas":"KS","kentucky":"KY","louisiana":"LA","maine":"ME","maryland":"MD","massachusetts":"MA","michigan":"MI",
    "minnesota":"MN","mississippi":"MS","missouri":"MO","montana":"MT","nebraska":"NE","nevada":"NV","new hampshire":"NH",
    "new jersey":"NJ","new mexico":"NM","new york":"NY","north carolina":"NC","north dakota":"ND","ohio":"OH","oklahoma":"OK",
    "oregon":"OR","pennsylvania":"PA","rhode island":"RI","south carolina":"SC","south dakota":"SD","tennessee":"TN",
    "texas":"TX","utah":"UT","vermont":"VT","virginia":"VA","washington":"WA","west virginia":"WV","wisconsin":"WI","wyoming":"WY",
    "dc":"DC","district of columbia":"DC"
}

def norm_state(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    k = s.lower()
    return STATE_ABBR.get(k, s.upper() if len(s) == 2 else s)

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    φ1, φ2 = radians(lat1), radians(lat2)
    dφ, dλ = radians(lat2-lat1), radians(lon2-lon1)
    a = sin(dφ/2)**2 + cos(φ1)*cos(φ2)*sin(dλ/2)**2
    return 2*R*asin(sqrt(a))

# --- Yelp & Traffic ---
def search_yelp_business(name, location):
    """Keep location simple: ZIP or 'town, ST'. Gracefully handle missing key/rate limits."""
    if not YELP_API_KEY:
        return None
    url = "https://api.yelp.com/v3/businesses/search"
    params = {"term": name, "location": location, "limit": 1}
    try:
        r = requests.get(url, headers=YELP_HEADERS, params=params, timeout=10)
        if r.status_code == 200:
            js = r.json()
            if js.get("businesses"):
                return js["businesses"][0]
    except requests.RequestException:
        pass
    return None

def get_yelp_sentiment_score(business):
    if not business:
        return 50.0
    rating = business.get("rating", 3.0)
    review_count = business.get("review_count", 0)
    return round(min(100, max(0, (rating - 3) * 25 + review_count * 0.1)), 1)

def get_yelp_insights(store, location):
    business = search_yelp_business(store, location)
    if not business:
        return None
    yelp_data = {
        "name": business.get("name", "N/A"),
        "rating": business.get("rating", "N/A"),
        "review_count": business.get("review_count", "N/A"),
        "categories": ", ".join([cat["title"] for cat in business.get("categories", [])]),
        "location": business.get("location", {}).get("address1", "N/A"),
        "phone": business.get("phone", "N/A"),
        "yelp_url": business.get("url", "N/A")
    }
    return yelp_data

def get_mock_placer_traffic(zip_code):
    if not zip_code:
        return 0.45
    hashval = sum(ord(c) for c in zip_code) % 10
    return round(0.3 + 0.05 * hashval, 2)

# --- Geolocation (REPLACED) ---
def get_coords_from_store_name(name, zip_code, town, state, radius_m=25000):
    """Return a list of candidate (lat, lon, label). Accept results within `radius_m` of ZIP centroid if available."""
    if not (name and zip_code and town and state):
        return []

    st_norm = norm_state(state)
    geolocator = Nominatim(user_agent="retail_ai_locator")

    def safe_geocode(query):
        for _ in range(3):
            try:
                return geolocator.geocode(query, exactly_one=True, timeout=10)
            except (GeocoderTimedOut, GeocoderUnavailable):
                time.sleep(1)
        return None

    # 1) ZIP centroid (anchor)
    zip_loc = safe_geocode(f"{zip_code}, USA")
    zip_anchor = (zip_loc.latitude, zip_loc.longitude) if zip_loc else None

    # 2) Try Yelp near ZIP (preferred) else "town, ST"
    yelp_loc_str = zip_code if zip_anchor else f"{town}, {st_norm}"
    business = search_yelp_business(name, yelp_loc_str)
    if business:
        c = business.get("coordinates") or {}
        lat, lon = c.get("latitude"), c.get("longitude")
        if lat and lon:
            lbl = f"{business.get('name','Business')}, {business.get('location',{}).get('address1','Yelp')}"
            if not zip_anchor or haversine_m(lat, lon, *zip_anchor) <= radius_m:
                return [(lat, lon, lbl)]

    # 3) Geocode combinations (name constrained to area)
    q1 = f"{name}, {town}, {st_norm}, {zip_code}, USA"
    q2 = f"{name}, {zip_code}, USA"
    q3 = f"{name}, {town}, {st_norm}, USA"
    for q in (q1, q2, q3):
        loc = safe_geocode(q)
        if loc:
            if not zip_anchor or haversine_m(loc.latitude, loc.longitude, *zip_anchor) <= radius_m:
                return [(loc.latitude, loc.longitude, loc.address)]

    # 4) Fall back to anchors so user can still proceed
    if zip_anchor:
        return [(zip_anchor[0], zip_anchor[1], f"{zip_code} centroid")]
    town_loc = safe_geocode(f"{town}, {st_norm}, USA")
    if town_loc:
        return [(town_loc.latitude, town_loc.longitude, town_loc.address)]

    return []

def show_map_with_selection(options):
    st.subheader("📍 Select Your Store Location")
    m = folium.Map(location=[options[0][0], options[0][1]], zoom_start=14)
    for lat, lon, label in options:
        folium.Marker(location=[lat, lon], tooltip=label).add_to(m)
    st_folium(m, height=350, width=700)
    return options[0][:2]

# --- Satellite & Feature Engineering ---
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

def build_feature_vector(img, coords, store, zip_code):
    features = np.concatenate([extract_satellite_features(img), [coords[0], coords[1]]])
    # Keep Yelp lookup simple; location as ZIP is fine
    business = search_yelp_business(store, zip_code)
    foot = get_mock_placer_traffic(zip_code)
    soc = get_yelp_sentiment_score(business)

    # Store features in session state for consistency
    st.session_state['features'] = features
    st.session_state['foot'] = foot
    st.session_state['soc'] = soc

    return features, foot, soc

# --- Model & Prediction ---
def load_fallback_model():
    dummy = DummyRegressor(strategy="constant", constant=np.random.randint(10000, 30000))
    dummy.fit([[0]*514], [dummy.constant])
    return dummy

def load_real_data_model():
    path = "real_sales_data.csv"
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    if not os.path.exists(path):
        return load_fallback_model()
    try:
        df = pd.read_csv(path)
        X = df[[f"f{i}" for i in range(512)] + ["lat", "lon"]]
        y = df["sales"]
        model = GradientBoostingRegressor()
        model.fit(X, y)
        joblib.dump(model, "model.pkl")
        return model
    except:
        return load_fallback_model()

# --- Save & Visualize ---
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
    if 'sales_history.csv' not in os.listdir():
        return st.info("No data yet.")
    try:
        df = pd.read_csv("sales_history.csv", on_bad_lines='skip')
    except:
        return st.warning("Could not read history file.")

    df = df[df["store"].str.lower() == store.lower()]
    if df.empty:
        return st.warning("No data found.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    st.subheader("📈 Sales Over Time")
    fig_sales = px.line(df, x="timestamp", y="sales", title="Weekly Sales Forecast", markers=True)
    fig_sales.update_traces(line=dict(width=2))
    st.plotly_chart(fig_sales)

    st.subheader("👣 Foot Traffic vs. 📱 Online Buzz")
    df_long = df.melt(id_vars=["timestamp"], value_vars=["foot", "social"], var_name="metric", value_name="score")
    fig_buzz = px.line(df_long, x="timestamp", y="score", color="metric", markers=True)
    fig_buzz.update_traces(line=dict(width=2))
    st.plotly_chart(fig_buzz)

    avg_type = df.groupby("type")["sales"].mean().reset_index()
    st.subheader("🏷️ Average Sales by Store Type")
    fig_type = px.bar(avg_type, x="type", y="sales", color="type", title="Avg Weekly Sales per Store Type")
    st.plotly_chart(fig_type)

# --- Strategy Engine (REPLACED for variety) ---
def generate_recommendations(store, store_type, foot, soc, sales, *, n=6, seed=None):
    rng = random.Random(seed or f"{store}-{int(time.time()//3600)}")  # rotates hourly

    # Foot-traffic levers
    low_foot = [
        "Partner with nearby gyms/schools for cross-promos (e.g., show ID for 10% off).",
        "Run a map-pin ad targeting a 2-mile radius during commute hours.",
        "Place sidewalk signage with a time-boxed offer (e.g., 3–5pm happy hour).",
        "Host a micro-event (trivia, open mic, kids’ craft table) to create a reason to visit.",
    ]
    mid_foot = [
        "Introduce off-peak bundles (e.g., drink + snack at a small discount).",
        "Test 2-week A/B on hours: open 30 minutes earlier or later and compare ticket counts.",
        "Promote limited-time items on in-store screens or table tents.",
        "Offer 'bring-a-friend' coupons that trigger only during lull hours.",
    ]
    high_foot = [
        "Speed wins: pre-queue ordering signs and a clear express lane.",
        "Upsell at POS with one-tap add-ons (sauces, sides, bakery).",
        "Loyalty ladder: free item at 5 visits, premium at 10 to raise repeat rate.",
        "Place impulse items within 3 feet of POS to lift average ticket.",
    ]

    # Social/buzz levers
    weak_buzz = [
        "Ask for reviews via QR on receipts; thank-you coupon unlocks after posting.",
        "Refresh Google/Yelp photos with bright, current shots; add alt text & hours.",
        "Run a 'review of the week' board to normalize/celebrate feedback.",
    ]
    mid_buzz = [
        "Post 2 short-form videos/week highlighting staff picks and behind-the-scenes.",
        "UGC drive: 'share your combo' hashtag; pick a weekly winner for a free item.",
    ]
    high_buzz = [
        "Trial a micro-influencer (1–5k local followers) with a store-only code.",
        "Launch a 'VIP hour' for followers with a secret menu item.",
    ]

    # Store-type flavor
    flavor = {
        "Coffee Shop": [
            "Introduce a rotating single-origin and stamp cards for trying all 3.",
            "Offer mobile pre-order for morning rush; pickup shelf clearly labeled.",
        ],
        "Fast Food": [
            "Bundle main+side+drink at a slight discount; promote on window clings.",
            "Add a $1-$2 add-on menu to lift attachment rate.",
        ],
        "Boutique": [
            "Create a try-on wall/LookBook; staff do 15-sec styling reels.",
            "Host a 'bring an item, get styled' night with RSVPs.",
        ],
        "Any": [],
        "Other": []
    }

    base = []
    if sales < 10000:
        base += [
            "Tighten menu/SKU count to best-sellers for faster service and simpler ops.",
            "Flyer drop to closest apartments/offices with a first-visit code.",
        ]
    elif sales > 30000:
        base += [
            "Scope second-site feasibility: same ZIP heatmap, competitor spacing, staffing.",
            "Spin up an online ordering page with curbside pickup slots.",
        ]

    if foot < 0.4:
        pool = low_foot
    elif foot < 0.6:
        pool = mid_foot
    else:
        pool = high_foot

    if soc < 40:
        pool_buzz = weak_buzz
    elif soc < 70:
        pool_buzz = mid_buzz
    else:
        pool_buzz = high_buzz

    recs = set()
    recs.update(rng.sample(pool, k=min(2, len(pool))))
    recs.update(rng.sample(pool_buzz, k=min(2, len(pool_buzz))))
    fpool = flavor.get(store_type, []) or flavor["Other"]
    if fpool:
        recs.update(rng.sample(fpool, k=min(1, len(fpool))))
    if base:
        recs.update(rng.sample(base, k=min(1, len(base))))

    all_pools = list({*low_foot, *mid_foot, *high_foot, *weak_buzz, *mid_buzz, *high_buzz, *sum(flavor.values(), []), *base})
    while len(recs) < n and len(recs) < len(all_pools):
        recs.add(rng.choice(all_pools))

    pretty = []
    for item in recs:
        if any(k in item.lower() for k in ["review", "google", "yelp"]):
            pretty.append(("📉 " if soc < 40 else "📱 ") + item)
        elif any(k in item.lower() for k in ["bundle", "add-on", "upsell"]):
            pretty.append("🧺 " + item)
        elif any(k in item.lower() for k in ["loyal", "stamp"]):
            pretty.append("🎟️ " + item)
        elif any(k in item.lower() for k in ["influencer", "hashtag", "reels"]):
            pretty.append("📣 " + item)
        else:
            pretty.append("💡 " + item)
    return list(pretty)[:n]

# --- Main App Execution ---
st.title("📊 Retail AI: Forecast & Strategy")
store = st.text_input("🏪 Store Name (e.g. Dave's Hot Chicken)")
zip_code = st.text_input("📍 ZIP Code (required)")
town = st.text_input("🏙️ Town")
state = st.text_input("🏞️ State")
coords = None

if store and zip_code and town and state:
    candidates = get_coords_from_store_name(store, zip_code, town, state)
    if candidates:
        coords = show_map_with_selection(candidates)
    else:
        st.warning("Location not found or outside ZIP radius.")

if not coords:
    coords = show_map_with_selection([(40.7128, -74.0060, "New York (default)")])

if coords:
    image = fetch_or_upload_satellite_image(coords)
    st.image(image, caption="🧪 Satellite View", use_container_width=True)

    # Get Yelp Insights (keep location simple: ZIP is fine)
    yelp_insights = get_yelp_insights(store, zip_code)
    if yelp_insights:
        st.subheader("📖 Yelp Insights")
        st.write(f"**Store Name**: {yelp_insights['name']}")
        st.write(f"**Rating**: {yelp_insights['rating']}⭐")
        st.write(f"**Reviews**: {yelp_insights['review_count']} reviews")
        st.write(f"**Categories**: {yelp_insights['categories']}")
        st.write(f"**Location**: {yelp_insights['location']}")
        st.write(f"**Phone**: {yelp_insights['phone']}")
        st.write(f"[Yelp Link]({yelp_insights['yelp_url']})")

    if st.button("📊 Predict & Analyze"):
        try:
            features, foot, soc = build_feature_vector(image, coords, store, zip_code)
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
