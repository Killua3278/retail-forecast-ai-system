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
from math import radians, sin, cos, asin, sqrt
import random
from datetime import datetime, timedelta

load_dotenv()
st.set_page_config(page_title="Retail AI Platform", layout="wide")

# --- Secrets helpers ---
def _try_secrets(path_list):
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
YELP_HEADERS = {"Authorization": f"Bearer {YELP_API_KEY}"} if YELP_API_KEY else {}

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

# --- Yelp API helpers (details + reviews) ---
def _yelp_get(path, params=None):
    if not YELP_API_KEY:
        return None
    try:
        r = requests.get(f"https://api.yelp.com/v3{path}", headers=YELP_HEADERS, params=params or {}, timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except requests.RequestException:
        return None

def get_yelp_business_details(business_id: str):
    if not business_id:
        return None
    return _yelp_get(f"/businesses/{business_id}")

def get_yelp_top_reviews(business_id: str, locale="en_US"):
    js = _yelp_get(f"/businesses/{business_id}/reviews", params={"locale": locale})
    return (js or {}).get("reviews", [])[:3]

def search_yelp_business(name, location=None, coords=None):
    """
    Prefer coordinate search (more accurate, fewer 400s).
    Fallback to location string (ZIP or 'town, ST') if coords not available.
    """
    if not YELP_API_KEY or not name:
        return None
    url = "https://api.yelp.com/v3/businesses/search"
    params = {"term": name, "limit": 1, "sort_by": "best_match"}
    if coords:
        params["latitude"], params["longitude"] = coords[0], coords[1]
        params["radius"] = 25000  # 25km max
    else:
        if not location:
            return None
        params["location"] = location
    try:
        r = requests.get(url, headers=YELP_HEADERS, params=params, timeout=10)
        if r.status_code == 200:
            js = r.json()
            if js.get("businesses"):
                return js["businesses"][0]
        elif r.status_code in (401, 403):
            st.warning("Yelp API unauthorized or rate-limited. Check key/quota.")
        elif r.status_code == 400:
            st.info("Yelp API returned 400 (validation). Will try fallback path.")
        else:
            st.info(f"Yelp API returned {r.status_code}.")
    except requests.RequestException:
        pass
    return None

def get_yelp_sentiment_score(business_or_details):
    if not business_or_details:
        return 50.0
    rating = business_or_details.get("rating", 3.0) or 3.0
    review_count = business_or_details.get("review_count", 0) or 0
    return round(min(100, max(0, (rating - 3) * 25 + review_count * 0.1)), 1)

def get_yelp_insights(store, location, coords=None):
    # Find the best match first (coords-first, fallback to location)
    biz = search_yelp_business(store, location=location, coords=coords) or \
          search_yelp_business(store, location=location, coords=None)
    if not biz:
        return None

    # Merge basic hit with full details
    details = get_yelp_business_details(biz.get("id")) or {}
    location_obj = details.get("location") or biz.get("location") or {}
    categories = details.get("categories") or biz.get("categories") or []
    coords_obj = details.get("coordinates") or biz.get("coordinates") or {}

    data = {
        "id": biz.get("id"),
        "name": details.get("name") or biz.get("name"),
        "rating": details.get("rating", biz.get("rating")),
        "review_count": details.get("review_count", biz.get("review_count")),
        "price": details.get("price"),  # "$", "$$", "$$$", "$$$$"
        "is_closed": details.get("is_closed", biz.get("is_closed")),
        "transactions": details.get("transactions", []),
        "photos": details.get("photos", []),
        "categories": ", ".join([c.get("title","") for c in categories if c.get("title")]) or "N/A",
        "location": ", ".join(filter(None, [
            location_obj.get("address1",""), location_obj.get("city",""),
            location_obj.get("state",""), location_obj.get("zip_code","")
        ])) or "N/A",
        "display_address": " • ".join(location_obj.get("display_address", [])) or None,
        "phone": details.get("display_phone") or details.get("phone") or biz.get("display_phone") or biz.get("phone"),
        "yelp_url": details.get("url", biz.get("url")),
        "coordinates": coords_obj,
        "hours": details.get("hours", []),
    }
    data["top_reviews"] = get_yelp_top_reviews(biz.get("id")) or []
    return data

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
        dates, vals = [], []
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
      - RRSFS: Real Retail & Food Services Sales (higher is good)
      - UMCSENT: Consumer Sentiment (higher is good)
      - UNRATE: Unemployment Rate (lower is good)
    """
    series_map = {"RRSFS": "RRSFS", "UMCSENT": "UMCSENT", "UNRATE": "UNRATE"}
    out = {}
    for key, sid in series_map.items():
        s = fred_series(sid)
        if s.empty:
            out[key] = {"latest": None, "yoy": None, "z": 0.0}
            continue
        latest = s.iloc[-1]
        prev_idx = s.index.searchsorted(s.index[-1] - pd.DateOffset(years=1))
        yoy = None
        if 0 <= prev_idx < len(s):
            prev = s.iloc[prev_idx]
            if prev != 0:
                yoy = (latest - prev) / abs(prev)
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
    ybiz = search_yelp_business(name, location=yelp_loc_str, coords=None)
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

def show_map_with_selection(options, *, show_radius_m=400):
    st.subheader("📍 Select Your Store Location")
    # Center map and add a CLEAR red pin + optional radius circle
    m = folium.Map(location=[options[0][0], options[0][1]], zoom_start=14, control_scale=True)
    for lat, lon, label in options:
        folium.Marker(
            location=[lat, lon],
            tooltip=label,
            popup=label,
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)
        if show_radius_m:
            folium.Circle(
                radius=show_radius_m,
                location=[lat, lon],
                color="#FF4136",
                fill=True,
                fill_opacity=0.08
            ).add_to(m)
    st_folium(m, height=380, width=None)
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

    # Yelp (coords-first inside get_yelp_insights)
    yelp_loc = zip_code or (f"{town}, {norm_state(state)}" if (town and state) else "")
    ydetails = get_yelp_insights(store, yelp_loc, coords=coords)
    rating = (ydetails or {}).get("rating", 3.0) or 3.0
    reviews = (ydetails or {}).get("review_count", 0) or 0
    yelp_sent = get_yelp_sentiment_score(ydetails)

    # Foot traffic proxy
    foot = get_mock_placer_traffic(zip_code)

    # FRED macro features
    fred = fred or fred_features()
    rrsfs_z = fred.get("RRSFS", {}).get("z", 0.0) or 0.0
    umcsent_z = fred.get("UMCSENT", {}).get("z", 0.0) or 0.0
    unrate_z = fred.get("UNRATE", {}).get("z", 0.0) or 0.0

    aux = np.array([rating, reviews, yelp_sent, foot, rrsfs_z, umcsent_z, unrate_z], dtype=float)
    model_feats = np.concatenate([sat, latlon], axis=0)   # 514 dims

    st.session_state['aux_feats'] = {
        "rating": rating, "reviews": reviews, "yelp_sent": yelp_sent, "foot": foot,
        "rrsfs_z": rrsfs_z, "umcsent_z": umcsent_z, "unrate_z": unrate_z
    }
    return model_feats, aux

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
    Try model.predict with 514-dim features; blend with heuristic using Yelp + Foot + FRED.
    """
    pred = None
    try:
        pred = float(model.predict([model_feats])[0])
    except Exception:
        pred = None

    rating, reviews, yelp_sent, foot, rrsfs_z, umcsent_z, unrate_z = aux_feats
    baseline = 15000.0

    rat_mult  = 1.0 + (rating - 4.0) * 0.07
    rev_mult  = 1.0 + min(reviews, 1000) / 10000.0
    buzz_mult = 1.0 + (yelp_sent - 50) / 500.0
    foot_mult = 0.8 + foot
    macro_mult = (1.0 + 0.05 * rrsfs_z) * (1.0 + 0.04 * umcsent_z) * (1.0 - 0.04 * unrate_z)

    heuristic = float(np.clip(baseline * rat_mult * rev_mult * buzz_mult * foot_mult * macro_mult, 3000, 75000))

    if pred is None or np.isnan(pred) or pred < 2000 or pred > 100000:
        return heuristic
    return 0.65 * pred + 0.35 * heuristic

# --- Save & Visualize ---
def save_prediction(store, coords, pred, aux, timestamp=None):
    ts = timestamp or pd.Timestamp.now()
    df = pd.DataFrame([[store, coords[0], coords[1], store_type, pred, aux["foot"], aux["yelp_sent"],
                        aux["rating"], aux["reviews"], aux["rrsfs_z"], aux["umcsent_z"], aux["unrate_z"], ts]],
                      columns=["store","lat","lon","type","sales","foot","social","rating","reviews",
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
        macros += ["Lean into value messaging; emphasize bundles and loyalty to offset weak sentiment."]
    if fred.get("UNRATE", {}).get("z", 0) > 0.5:
        macros += ["Promote budget-friendly options and limited-time deals to protect traffic during softer labor markets."]
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
        coords = show_map_with_selection(candidates)  # red pin + circle
    else:
        st.warning("Location not found or outside ZIP radius.")

if not coords:
    coords = show_map_with_selection([(40.7128, -74.0060, "New York (default)")])

if coords:
    # Satellite
    image = fetch_or_upload_satellite_image(coords)
    st.image(image, caption="🛰️ Satellite View", use_container_width=True)

    # Yelp Insights (coords-first to avoid Yelp 400s)
    yelp_location_hint = zip_code or f"{town}, {norm_state(state)}"
    yelp = get_yelp_insights(store, yelp_location_hint, coords=coords)

    def _stars(rating):
        try:
            r = float(rating)
        except:
            return "—"
        full = "★" * int(r)
        half = "½" if r - int(r) >= 0.5 else ""
        empty = "☆" * max(0, 5 - int(r) - (1 if half else 0))
        return full + half + empty

    if yelp:
        st.subheader("📖 Yelp Insights")
        left, right = st.columns([2,1])
        with left:
            title = f"**{yelp['name']}**"
            if yelp.get("price"):
                title += f" · {yelp['price']}"
            if yelp.get("is_closed") is True:
                title += " · _Closed_"
            st.markdown(title)

            st.write(f"**Rating**: {yelp['rating']} {_stars(yelp['rating'])}  ·  **Reviews**: {yelp['review_count']}")
            st.write(f"**Categories**: {yelp['categories']}")
            st.write(f"**Address**: {yelp.get('display_address') or yelp['location']}")
            if yelp.get("transactions"):
                st.write("**Transactions**: " + ", ".join(yelp["transactions"]))
            st.write(f"**Phone**: {yelp['phone'] or 'N/A'}")
            st.write(f"[Open on Yelp]({yelp['yelp_url']})")

            # Hours (today)
            if yelp.get("hours"):
                todays = yelp["hours"][0].get("open", [])
                if todays:
                    import datetime as _dt
                    dow = _dt.datetime.today().weekday()  # 0=Mon
                    todays_spans = [o for o in todays if o.get("day") == dow]
                    st.markdown("**Hours (today)**")
                    if todays_spans:
                        for span in todays_spans:
                            s = span.get("start","")
                            e = span.get("end","")
                            if len(s) >= 4 and len(e) >= 4:
                                st.write(f"{s[:2]}:{s[2:]}–{e[:2]}:{e[2:]}")
                            else:
                                st.write("—")
                    else:
                        st.write("—")

            # Up to 3 recent review snippets
            if yelp.get("top_reviews"):
                st.markdown("**Recent Reviews**")
                for r in yelp["top_reviews"]:
                    author = (r.get("user") or {}).get("name", "Anonymous")
                    rr = r.get("rating", "?")
                    txt = r.get("text","").strip()
                    st.write(f"• _{author}_ — {rr}⭐: {txt}")

        with right:
            photos = yelp.get("photos") or []
            if photos:
                for p in photos[:3]:
                    if p:
                        st.image(p, use_container_width=True)
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

            save_prediction(store, coords, pred, st.session_state['aux_feats'])
            plot_insights(store)

            st.subheader("📦 Strategy Recommendations")
            for r in generate_recommendations(store, store_type,
                                              st.session_state['aux_feats']['foot'],
                                              st.session_state['aux_feats']['yelp_sent'],
                                              pred, macros):
                st.markdown(f"- {r}")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
