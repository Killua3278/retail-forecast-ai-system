# ✅ Retail AI App — Yelp + ZIP + Satellite + FRED Macros + Strategy Intelligence

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
from math import radians, sin, cos, asin, sqrt
import random
from datetime import datetime, timedelta

load_dotenv()
st.set_page_config(page_title="Retail AI Platform", layout="wide")

# --- Secrets helpers ---
def _try_secrets(path_list):
    """Try multiple locations inside st.secrets (root, sections)"""
    cur = st.secrets if hasattr(st, "secrets") else {}
    for key in path_list:
        try:
            cur = cur[key]
        except Exception:
            return None
    return cur

def get_secret(name):
    """
    Resolve secret from:
      1) st.secrets[name]
      2) st.secrets['api'][name]
      3) st.secrets['general'][name]
      4) os.environ[name]
    """
    val = None
    val = _try_secrets([name]) or _try_secrets(["api", name]) or _try_secrets(["general", name])
    if not val:
        val = os.getenv(name)
    return val

def _mask(s):
    return s[:4] + "…" + s[-4:] if s and len(s) > 8 else "(unset)"

# --- API Keys (from Secrets/Env) ---
YELP_API_KEY = get_secret("YELP_API_KEY")
GOOGLE_MAPS_API_KEY = get_secret("GOOGLE_MAPS_API_KEY")
FRED_API_KEY = get_secret("FRED_API_KEY")

# --- Sidebar ---
st.sidebar.title("🔧 Settings")
store_type = st.sidebar.selectbox("Store Type", ["Any", "Coffee Shop", "Boutique", "Fast Food", "Other"])
theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=1)

with st.sidebar.expander("🔌 Integrations"):
    st.caption(f"Google Maps: {_mask(GOOGLE_MAPS_API_KEY) if GOOGLE_MAPS_API_KEY else '(missing)'}")
    st.caption(f"Yelp: {_mask(YELP_API_KEY) if YELP_API_KEY else '(missing)'}")
    st.caption(f"FRED: {_mask(FRED_API_KEY) if FRED_API_KEY else '(missing)'}")

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = (theme == "Dark")

if st.sidebar.button("🥩 Clear Sales History"):
    if os.path.exists("sales_history.csv"):
        os.remove("sales_history.csv")
        st.sidebar.success("History cleared.")

# Build headers from active key
YELP_HEADERS = {"Authorization": f"Bearer {YELP_API_KEY}"} if YELP_API_KEY else {}

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

# --- Geo/units helpers ---
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
    if not s: return ""
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
        elif r.status_code in (401, 403):
            st.warning("Yelp API unauthorized/rate-limited. Verify key and quotas.")
        else:
            st.info(f"Yelp API returned {r.status_code}.")
    except requests.RequestException:
        pass
    return None

def get_yelp_sentiment_score(business):
    if not business:
        return 50.0
    rating = business.get("rating", 3.0) or 3.0
    review_count = business.get("review_count", 0) or 0
    # rating centered at 3.0; mild lift from review volume
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
        "location": ", ".join(filter(None, [
            business.get("location", {}).get("address1", ""),
            business.get("location", {}).get("city", ""),
            business.get("location", {}).get("state", ""),
            business.get("location", {}).get("zip_code", "")
        ])) or "N/A",
        "phone": business.get("phone", "N/A"),
        "yelp_url": business.get("url", "N/A"),
        "coordinates": business.get("coordinates", {})
    }
    return yelp_data

def get_mock_placer_traffic(zip_code):
    if not zip_code:
        return 0.45
    hashval = sum(ord(c) for c in zip_code) % 10
    return round(0.3 + 0.05 * hashval, 2)

# --- FRED Macros ---
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

def fred_series(series_id, api_key=FRED_API_KEY, months=24):
    """Fetch last N months of a FRED series, return pandas Series of floats indexed by date."""
    if not api_key:
        return pd.Series(dtype=float)
    start_date = (datetime.today() - timedelta(days=31*months)).strftime("%Y-%m-%d")
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date
    }
    try:
        r = requests.get(FRED_BASE, params=params, timeout=12)
        r.raise_for_status()
        data = r.json().get("observations", [])
        dates = []
        vals = []
        for o in data:
            v = o.get("value")
            if v is None or v in ("", "."):
                continue
            try:
                vals.append(float(v))
                dates.append(pd.to_datetime(o["date"]))
            except:
                continue
        if not vals:
            return pd.Series(dtype=float)
        s = pd.Series(vals, index=pd.to_datetime(dates)).sort_index()
        return s
    except Exception:
        return pd.Series(dtype=float)

def fred_features():
    """
    Pull macro features:
      - RRSFS: Real Retail & Food Services Sales (index; higher is good)
      - UMCSENT: Consumer Sentiment (higher is good)
      - UNRATE: Unemployment Rate (lower is good)
    Return dict with latest, YoY pct change, and z-scores for each.
    """
    series_map = {
        "RRSFS": "RRSFS",   # Real Retail & Food Services Sales
        "UMCSENT": "UMCSENT",  # Consumer Sentiment
        "UNRATE": "UNRATE"  # Unemployment rate
    }
    out = {}
    for key, sid in series_map.items():
        s = fred_series(sid)
        if s.empty:
            out[key] = {"latest": None, "yoy": None, "z": 0.0}
            continue
        latest = s.iloc[-1]
        # YoY change: compare to value ~12 months earlier if available
        prev_idx = s.index.searchsorted(s.index[-1] - pd.DateOffset(years=1))
        yoy = None
        if 0 <= prev_idx < len(s):
            prev = s.iloc[prev_idx]
            if prev != 0:
                yoy = (latest - prev) / abs(prev)
        # z-score using rolling window for robustness
        roll = s.rolling(12, min_periods=6)
        mean = roll.mean().iloc[-1]
        std = roll.std(ddof=0).iloc[-1] or 1.0
        z = float((latest - mean) / std) if std else 0.0
        out[key] = {"latest": float(latest), "yoy": float(yoy) if yoy is not None else None, "z": z}
    return out

# --- Geolocation ---
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
    ybiz = search_yelp_business(name, yelp_loc_str)
    if ybiz:
        c = ybiz.get("coordinates") or {}
        lat, lon = c.get("latitude"), c.get("longitude")
        if lat and lon:
            lbl = f"{ybiz.get('name','Business')}, {ybiz.get('location',{}).get('address1','Yelp')}"
            if not zip_anchor or haversine_m(lat, lon, *zip_anchor) <= radius_m:
                return [(lat, lon, lbl)]

    # 3) Geocode combinations (name constrained to area)
    for q in (f"{name}, {town}, {st_norm}, {zip_code}, USA",
              f"{name}, {zip_code}, USA",
              f"{name}, {town}, {st_norm}, USA"):
        loc = safe_geocode(q)
        if loc:
            if not zip_anchor or haversine_m(loc.latitude, loc.longitude, *zip_anchor) <= radius_m:
                return [(loc.latitude, loc.longitude, loc.address)]

    # 4) Fall back to anchors so user can proceed
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
    api_key = GOOGLE_MAPS_API_KEY
    if not api_key:
        st.error("Missing Google Maps API key")
        return Image.new("RGB", (512, 512), color=(200, 200, 200))
    try:
        url = f"https://maps.googleapis.com/maps/api/staticmap?center={coords[0]},{coords[1]}&zoom=18&size=600x400&maptype=satellite&key={api_key}"
        res = requests.get(url, timeout=12)
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

# --- Feature builder (Yelp + Foot + FRED) ---
def build_feature_vector(img, coords, store, zip_code, town=None, state=None, fred=None):
    sat = extract_satellite_features(img)
    latlon = np.array([coords[0], coords[1]], dtype=float)

    # Yelp (localized by ZIP or "town, ST")
    yelp_loc = zip_code or (f"{town}, {norm_state(state)}" if (town and state) else "")
    ybiz = search_yelp_business(store, yelp_loc)
    rating = (ybiz or {}).get("rating", 3.0) or 3.0
    reviews = (ybiz or {}).get("review_count", 0) or 0
    yelp_sent = get_yelp_sentiment_score(ybiz)

    # Foot traffic proxy
    foot = get_mock_placer_traffic(zip_code)

    # FRED macro features
    fred = fred or fred_features()
    rrsfs_z = fred.get("RRSFS", {}).get("z", 0.0) or 0.0
    umcsent_z = fred.get("UMCSENT", {}).get("z", 0.0) or 0.0
    unrate_z = fred.get("UNRATE", {}).get("z", 0.0) or 0.0

    aux = np.array([
        rating, reviews, yelp_sent, foot,
        rrsfs_z, umcsent_z, unrate_z
    ], dtype=float)

    # Full feature vector for model (keeps old shape first 514; we append aux safely)
    # If a pre-trained model expects 514 dims, we can still pass only 514; keep AUX for heuristic.
    model_feats = np.concatenate([sat, latlon], axis=0)   # len 514
    aux_feats = aux                                      # len 7

    st.session_state['aux_feats'] = {
        "rating": rating, "reviews": reviews, "yelp_sent": yelp_sent, "foot": foot,
        "rrsfs_z": rrsfs_z, "umcsent_z": umcsent_z, "unrate_z": unrate_z
    }
    return model_feats, aux_feats

# --- Model & Prediction ---
def load_fallback_model():
    dummy = DummyRegressor(strategy="constant", constant=np.random.randint(12000, 24000))
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
    except Exception:
        return load_fallback_model()

def hybrid_prediction(model, model_feats, aux_feats):
    """
    Try model.predict with 514-dim features. If it fails or looks off, blend with a heuristic that
    uses Yelp + Foot + FRED macros.
    """
    pred = None
    try:
        pred = float(model.predict([model_feats])[0])
    except Exception:
        pred = None

    # Heuristic baseline (15k) scaled by signals
    rating, reviews, yelp_sent, foot, rrsfs_z, umcsent_z, unrate_z = aux_feats
    baseline = 15000.0

    # Yelp multipliers
    rat_mult = 1.0 + (rating - 4.0) * 0.07          # +/- ~7% per star around 4.0
    rev_mult = 1.0 + min(reviews, 1000) / 10000.0   # up to +10% from heavy review volume
    buzz_mult = 1.0 + (yelp_sent - 50) / 500.0      # +/-10% around 50

    # Foot traffic proxy
    foot_mult = 0.8 + foot  # 0.8–1.3 rough range (0.3→1.1, 0.8→1.6) clamped later

    # Macro multipliers from z-scores
    macro_mult = (1.0 + 0.05 * rrsfs_z) * (1.0 + 0.04 * umcsent_z) * (1.0 - 0.04 * unrate_z)

    heuristic = baseline * rat_mult * rev_mult * buzz_mult * foot_mult * macro_mult
    heuristic = float(np.clip(heuristic, 3000, 75000))

    if pred is None or np.isnan(pred) or pred < 2000 or pred > 100000:
        return heuristic
    # Blend model with heuristic for stability
    return 0.65 * pred + 0.35 * heuristic

# --- Save & Visualize ---
def save_prediction(store, coords, pred, aux, timestamp=None):
    ts = timestamp or pd.Timestamp.now()
    df = pd.DataFrame([[
        store, coords[0], coords[1], store_type, pred, aux["foot"], aux["yelp_sent"],
        aux["rating"], aux["reviews"], aux["rrsfs_z"], aux["umcsent_z"], aux["unrate_z"], ts
    ]], columns=["store","lat","lon","type","sales","foot","social","rating","reviews",
                 "rrsfs_z","umcsent_z","unrate_z","timestamp"])
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

    dff = df[df["store"].str.lower() == store.lower()].copy()
    if dff.empty:
        return st.warning("No data found for this store yet.")
    dff["timestamp"] = pd.to_datetime(dff["timestamp"])

    st.subheader("📈 Sales Over Time")
    fig_sales = px.line(dff, x="timestamp", y="sales", title="Weekly Sales Forecast", markers=True)
    fig_sales.update_traces(line=dict(width=2))
    st.plotly_chart(fig_sales, use_container_width=True)

    st.subheader("👣 Foot Traffic vs. 📱 Online Buzz")
    df_long = dff.melt(id_vars=["timestamp"], value_vars=["foot", "social"], var_name="metric", value_name="score")
    fig_buzz = px.line(df_long, x="timestamp", y="score", color="metric", markers=True)
    fig_buzz.update_traces(line=dict(width=2))
    st.plotly_chart(fig_buzz, use_container_width=True)

    st.subheader("🏷️ Average Sales by Store Type")
    avg_type = df.groupby("type")["sales"].mean().reset_index()
    fig_type = px.bar(avg_type, x="type", y="sales", color="type", title="Avg Weekly Sales per Store Type")
    st.plotly_chart(fig_type, use_container_width=True)

# --- Strategy Engine (macro-aware, varied) ---
def generate_recommendations(store, store_type, foot, soc, sales, fred, *, n=6, seed=None):
    rng = random.Random(seed or f"{store}-{int(time.time()//3600)}")
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
        "Any": [], "Other": []
    }

    # Macro-aware nudges
    macros = []
    if fred.get("UMCSENT", {}).get("z", 0) < -0.5:
        macros += ["Lean into value messaging in ads; emphasize bundles and loyalty to offset weak sentiment."]
    if fred.get("UNRATE", {}).get("z", 0) > 0.5:
        macros += ["Promote budget-friendly options and limited-time deals to protect traffic during soft labor markets."]
    if fred.get("RRSFS", {}).get("z", 0) > 0.5:
        macros += ["Experiment with premium add-ons while retail demand is above trend."]

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

    pool = low_foot if foot < 0.4 else mid_foot if foot < 0.6 else high_foot
    pool_buzz = weak_buzz if soc < 40 else mid_buzz if soc < 70 else high_buzz

    recs = set()
    recs.update(rng.sample(pool, k=min(2, len(pool))))
    recs.update(rng.sample(pool_buzz, k=min(2, len(pool_buzz))))
    fpool = flavor.get(store_type, []) or flavor["Other"]
    if fpool: recs.update(rng.sample(fpool, k=min(1, len(fpool))))
    if base:  recs.update(rng.sample(base,  k=min(1, len(base))))
    if macros: recs.update(rng.sample(macros, k=min(1, len(macros))))

    all_pools = list({*low_foot, *mid_foot, *high_foot, *weak_buzz, *mid_buzz, *high_buzz, *sum(flavor.values(), []), *base, *macros})
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
    # Satellite
    image = fetch_or_upload_satellite_image(coords)
    st.image(image, caption="🛰️ Satellite View", use_container_width=True)

    # Yelp Insights
    yelp_location_hint = zip_code or f"{town}, {norm_state(state)}"
    yelp_insights = get_yelp_insights(store, yelp_location_hint)
    if yelp_insights:
        st.subheader("📖 Yelp Insights")
        st.write(f"**Store Name**: {yelp_insights['name']}")
        st.write(f"**Rating**: {yelp_insights['rating']}⭐")
        st.write(f"**Reviews**: {yelp_insights['review_count']} reviews")
        st.write(f"**Categories**: {yelp_insights['categories']}")
        st.write(f"**Location**: {yelp_insights['location']}")
        st.write(f"**Phone**: {yelp_insights['phone']}")
        st.write(f"[Yelp Link]({yelp_insights['yelp_url']})")
    elif not YELP_API_KEY:
        st.info("Add a Yelp API key in Streamlit Secrets to enable Yelp Insights.")

    # FRED Macro Panel
    macros = fred_features() if FRED_API_KEY else {}
    st.subheader("🏦 Macro Snapshot (FRED)")
    if macros:
        col1, col2, col3 = st.columns(3)
        def fmt(v, pct=False):
            if v is None: return "—"
            return f"{v*100:.1f}%" if pct else f"{v:.2f}"
        with col1:
            st.markdown("**RRSFS (Real Retail & Food Services)**")
            st.write("z-score:", f"{macros['RRSFS']['z']:.2f}" if macros['RRSFS']['z'] is not None else "—")
            st.write("YoY:", fmt(macros['RRSFS']['yoy'], pct=True))
        with col2:
            st.markdown("**UMCSENT (Consumer Sentiment)**")
            st.write("z-score:", f"{macros['UMCSENT']['z']:.2f}" if macros['UMCSENT']['z'] is not None else "—")
            st.write("YoY:", fmt(macros['UMCSENT']['yoy'], pct=True))
        with col3:
            st.markdown("**UNRATE (Unemployment)**")
            st.write("z-score:", f"{macros['UNRATE']['z']:.2f}" if macros['UNRATE']['z'] is not None else "—")
            st.write("YoY:", fmt(macros['UNRATE']['yoy'], pct=True))
    else:
        st.info("Add a FRED_API_KEY in Streamlit Secrets to enable macro-aware modeling.")

    # Predict & Analyze
    if st.button("📊 Predict & Analyze"):
        try:
            model_feats, aux_feats = build_feature_vector(
                image, coords, store, zip_code, town=town, state=state, fred=macros
            )
            model = load_real_data_model()
            pred = hybrid_prediction(model, model_feats, aux_feats)
            st.markdown(f"## 💰 Predicted Weekly Sales: **${pred:,.2f}**")

            # Save & Plots
            save_prediction(store, coords, pred, st.session_state['aux_feats'])
            plot_insights(store)

            # Strategy
            st.subheader("📦 Strategy Recommendations")
            for r in generate_recommendations(store, store_type,
                                              st.session_state['aux_feats']['foot'],
                                              st.session_state['aux_feats']['yelp_sent'],
                                              pred, macros):
                st.markdown(f"- {r}")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

