# ✅ Retail AI App — Yelp + ZIP + Satellite + FRED Macros + Strategy Intelligence (Geo + Yelp fixed)

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

# =========================
# Yelp API (matches + details; resilient to 400s)
# =========================
def _yelp_get(path, params=None):
    if not YELP_API_KEY:
        return None
    try:
        r = requests.get(f"https://api.yelp.com/v3{path}", headers=YELP_HEADERS, params=params or {}, timeout=12)
        if r.status_code == 200:
            return r.json()
        return None
    except requests.RequestException:
        return None

def _yelp_matches(name, city=None, state=None, zip_code=None, address1=None):
    """Use Yelp's fuzzy matcher to avoid search validation errors."""
    p = {"name": name, "country": "US", "match_threshold": "default"}
    if city: p["city"] = city
    if state: p["state"] = norm_state(state)
    if zip_code: p["zip_code"] = zip_code
    if address1: p["address1"] = address1
    js = _yelp_get("/businesses/matches", params=p)
    if not js: return []
    return js.get("businesses", [])

def _yelp_details(business_id):
    return _yelp_get(f"/businesses/{business_id}") or {}

def _yelp_reviews(business_id, locale="en_US"):
    js = _yelp_get(f"/businesses/{business_id}/reviews", params={"locale": locale}) or {}
    return js.get("reviews", [])[:3]

def _yelp_search(term, location=None, coords=None, limit=3):
    """Coordinate-first search; clean fallbacks."""
    if not YELP_API_KEY or not term:
        return []
    url = "https://api.yelp.com/v3/businesses/search"
    params = {"term": term, "limit": limit, "sort_by": "best_match"}
    if coords:
        params["latitude"], params["longitude"] = float(coords[0]), float(coords[1])
        params["radius"] = 40000  # Yelp allows up to 40km
    elif location:
        params["location"] = location
    else:
        return []
    try:
        r = requests.get(url, headers=YELP_HEADERS, params=params, timeout=12)
        if r.status_code == 200:
            return (r.json() or {}).get("businesses", [])
        # silently fall back on validation errors
        return []
    except requests.RequestException:
        return []

def find_yelp_business(name, zip_code=None, town=None, state=None, coords_hint=None):
    """
    Robust resolver:
      1) matches endpoint (name + city/state/zip) → details
      2) search with coords (if available) → top hit → details
      3) search with location string (zip or 'town, ST') → details
    """
    # 1) Matches
    matches = _yelp_matches(name, city=town, state=state, zip_code=zip_code)
    if matches:
        bid = matches[0].get("id")
        if bid:
            det = _yelp_details(bid)
            det["top_reviews"] = _yelp_reviews(bid)
            return det

    # 2) Search by coords
    if coords_hint:
        hits = _yelp_search(name, coords=coords_hint, limit=1)
        if hits:
            bid = hits[0].get("id")
            det = _yelp_details(bid)
            det["top_reviews"] = _yelp_reviews(bid)
            return det

    # 3) Search by location string
    loc_str = zip_code or (f"{town}, {norm_state(state)}" if (town and state) else None)
    if loc_str:
        hits = _yelp_search(name, location=loc_str, limit=1)
        if hits:
            bid = hits[0].get("id")
            det = _yelp_details(bid)
            det["top_reviews"] = _yelp_reviews(bid)
            return det

    return None

def yelp_sentiment_score(biz):
    if not biz: return 50.0
    rating = biz.get("rating", 3.0) or 3.0
    review_count = biz.get("review_count", 0) or 0
    return round(min(100, max(0, (rating - 3) * 25 + review_count * 0.1)), 1)

# --- FRED Macros ---
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

def fred_series(series_id, api_key=FRED_API_KEY, months=24):
    if not api_key:
        return pd.Series(dtype=float)
    start_date = (datetime.today() - timedelta(days=31*months)).strftime("%Y-%m-%d")
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json", "observation_start": start_date}
    try:
        r = requests.get(FRED_BASE, params=params, timeout=12)
        r.raise_for_status()
        data = r.json().get("observations", [])
        dates, vals = [], []
        for o in data:
            v = o.get("value")
            if v in (None, "", "."):
                continue
            try:
                vals.append(float(v))
                dates.append(pd.to_datetime(o["date"]))
            except:
                continue
        if not vals: return pd.Series(dtype=float)
        return pd.Series(vals, index=pd.to_datetime(dates)).sort_index()
    except Exception:
        return pd.Series(dtype=float)

def fred_features():
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
            if prev != 0: yoy = (latest - prev) / abs(prev)
        roll = s.rolling(12, min_periods=6)
        mean, std = roll.mean().iloc[-1], roll.std(ddof=0).iloc[-1] or 1.0
        z = float((latest - mean) / std) if std else 0.0
        out[key] = {"latest": float(latest), "yoy": float(yoy) if yoy is not None else None, "z": z}
    return out

# --- Geocoding helpers ---
def _safe_geocode(query):
    geolocator = Nominatim(user_agent="retail_ai_locator")
    for _ in range(3):
        try:
            return geolocator.geocode(query, exactly_one=True, timeout=10)
        except (GeocoderTimedOut, GeocoderUnavailable):
            time.sleep(1)
    return None

def _zip_centroid(zip_code):
    z = _safe_geocode(f"{zip_code}, USA")
    return (z.latitude, z.longitude) if z else None

def resolve_store_location(store, zip_code, town, state):
    """
    Primary: Yelp details (accurate lat/lon).
    Fallbacks: geocode store within town/state; ZIP centroid.
    Returns (candidates_list, yelp_details)
    """
    candidates = []
    yelp = find_yelp_business(store, zip_code=zip_code, town=town, state=state)
    if yelp:
        c = (yelp.get("coordinates") or {})
        lat, lon = c.get("latitude"), c.get("longitude")
        if lat and lon:
            label = f"Yelp: {yelp.get('name','')} — {', '.join(yelp.get('location',{}).get('display_address',[]) or [])}"
            candidates.append((lat, lon, label))

    # Geocode the store name within the area as a secondary hint
    st_norm = norm_state(state)
    for q in (f"{store}, {town}, {st_norm}, {zip_code}, USA",
              f"{store}, {town}, {st_norm}, USA",
              f"{store}, {zip_code}, USA"):
        loc = _safe_geocode(q)
        if loc:
            label = f"Geocode: {loc.address}"
            ll = (loc.latitude, loc.longitude, label)
            if ll not in candidates:
                candidates.append(ll)
            break

    # Always include ZIP centroid as a "just in case" option
    zc = _zip_centroid(zip_code) if zip_code else None
    if zc:
        candidates.append((zc[0], zc[1], f"{zip_code} centroid"))

    # Ensure at least one candidate (NYC as final fallback)
    if not candidates:
        candidates = [(40.7128, -74.0060, "New York (default)")]

    return candidates, yelp

def show_map_with_selection(options, *, show_radius_m=400, key="storemap"):
    st.subheader("📍 Select Your Store Location")
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

    state = st_folium(m, height=380, width=None, key=key)
    chosen = options[0][:2]
    if state and state.get("last_clicked"):
        chosen = (state["last_clicked"]["lat"], state["last_clicked"]["lng"])
        st.caption("Tip: map click captured — using your clicked point for all downstream steps.")
    return chosen

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
def build_feature_vector(img, coords, yelp_details, zip_code, fred=None):
    sat = extract_satellite_features(img)
    latlon = np.array([coords[0], coords[1]], dtype=float)

    rating = (yelp_details or {}).get("rating", 3.0) or 3.0
    reviews = (yelp_details or {}).get("review_count", 0) or 0
    yelp_sent = yelp_sentiment_score(yelp_details)

    # Foot traffic proxy
    def _foot_from_zip(z):
        if not z: return 0.45
        h = sum(ord(c) for c in str(z)) % 10
        return round(0.3 + 0.05 * h, 2)
    foot = _foot_from_zip(zip_code)

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

# --- Strategy Engine (clear & distinct) ---
def generate_recommendations(store, store_type, foot, soc, sales, fred, *, n=6, seed=None):
    rng = random.Random(seed or f"{store}-{int(time.time()//3600)}")

    # Footfall lanes
    low_foot = [
        "OFF-PEAK BUNDLE: 3–5pm combo at -12% (window cling + counter tent).",
        "LOCAL PARTNERS: cross-promo with nearby gym/school (show ID = 10% off).",
        "SIDEWALK SIGN: daily 'today only' offer to create urgency.",
    ]
    mid_foot = [
        "HAPPY-HOUR UPGRADE: add small side for +$1.50 during 2 quiet hours.",
        "SCHEDULE TEST: open 30 min earlier/later for 2 weeks → compare tickets.",
        "REFER-A-FRIEND: print QR cards redeemable only in lull hours.",
    ]
    high_foot = [
        "EXPRESS LANE + PRE-QUEUE SIGN: cut perceived wait, raise throughput.",
        "POS UPSELL: one-tap add-ons (‘+sauce +$0.75’) visible at checkout.",
        "LOYALTY LADDER: 5 visits = free item; 10 = premium reward.",
    ]

    # Buzz lanes
    weak_buzz = [
        "REVIEW QR: on receipts + takeout bags; reward after posting.",
        "LISTING REFRESH: new photos, accurate hours, pin top items.",
    ]
    mid_buzz = [
        "2 REELS/WEEK: staff picks + behind-the-scenes; link to QR offer.",
        "UGC PROMPT: 'share your combo' hashtag; weekly winner.",
    ]
    high_buzz = [
        "MICRO-INFLUENCER (1–5k local): unique code for store-only item.",
        "VIP HOUR: followers get a secret menu item for 1 hour weekly.",
    ]

    flavor = {
        "Coffee Shop": [
            "ROTATING SINGLE-ORIGIN: stamp card for trying 3 varieties.",
            "MOBILE PRE-ORDER: labeled pickup shelf for morning rush.",
        ],
        "Fast Food": [
            "VALUE BUNDLE: main+side+drink at slight discount; push on window clings.",
            "LOW-PRICE ADD-ONS: $1–$2 items to lift attachment rate.",
        ],
        "Boutique": [
            "LOOKBOOK WALL: quick styling reels; QR to featured items.",
            "RSVP EVENT: 'bring an item, get styled' evening.",
        ],
        "Any": [], "Other": []
    }

    macros = []
    if fred.get("UMCSENT", {}).get("z", 0) < -0.5:
        macros += ["LEAN VALUE MESSAGING: emphasize bundles/loyalty while sentiment is weak."]
    if fred.get("UNRATE", {}).get("z", 0) > 0.5:
        macros += ["BUDGET-FRIENDLY MENU: spotlight affordable sets while labor market softens."]
    if fred.get("RRSFS", {}).get("z", 0) > 0.5:
        macros += ["TEST PREMIUM ADD-ONS: above-trend retail demand supports upsells."]

    base = []
    if sales < 10000:
        base += ["TRIM MENU TO BEST-SELLERS for faster service + fewer stockouts.",
                 "FLYER DROP to nearby apartments/offices with first-visit code."]
    elif sales > 30000:
        base += ["SECOND-SITE SCOPING: heatmap + competitor spacing + staffing plan.",
                 "ONLINE ORDERING with curbside slots."]

    pool = low_foot if foot < 0.4 else mid_foot if foot < 0.6 else high_foot
    pool_buzz = weak_buzz if soc < 40 else mid_buzz if soc < 70 else high_buzz

    recs = set()
    recs.update(rng.sample(pool, k=min(2, len(pool))))
    recs.update(rng.sample(pool_buzz, k=min(2, len(pool_buzz))))
    fpool = flavor.get(store_type, []) or flavor["Other"]
    if fpool: recs.update(rng.sample(fpool, k=min(1, len(fpool))))
    if base:  recs.update(rng.sample(base,  k=min(1, len(base))))
    if macros: recs.update(rng.sample(macros, k=min(1, len(macros))))

    icons = []
    for item in recs:
        if "review" in item.lower(): icons.append("⭐ " + item)
        elif any(k in item.lower() for k in ["bundle","add-on","upsell"]): icons.append("🧺 " + item)
        elif any(k in item.lower() for k in ["loyal","stamp"]): icons.append("🎟️ " + item)
        elif any(k in item.lower() for k in ["influencer","reels","vip"]): icons.append("📣 " + item)
        else: icons.append("💡 " + item)
    return icons[:n]

# --- Main App Execution ---
st.title("📊 Retail AI: Forecast & Strategy")
store = st.text_input("🏪 Store Name (e.g. Dave's Hot Chicken)")
zip_code = st.text_input("📍 ZIP Code (required)")
town = st.text_input("🏙️ Town")
state = st.text_input("🏞️ State")
coords = None
yelp_details = None

if store and zip_code and town and state:
    # resolve candidates (Yelp-first) and show interactive map (click to refine)
    candidates, yelp_details = resolve_store_location(store, zip_code, town, state)
    coords = show_map_with_selection(candidates, key="map1")
else:
    # If not enough inputs, still show a map so user understands the UI
    coords = show_map_with_selection([(40.7128, -74.0060, "New York (default)")], key="map_default")

# --- Satellite + Yelp panel ---
if coords:
    image = fetch_or_upload_satellite_image(coords)
    st.image(image, caption="🛰️ Satellite View", use_container_width=True)

    # Ensure Yelp details (if not already resolved)
    if not yelp_details and (store and (zip_code or (town and state))):
        yelp_details = find_yelp_business(store, zip_code=zip_code, town=town, state=state, coords_hint=coords)

    # Yelp UI
    def _stars(rating):
        try: r = float(rating)
        except: return "—"
        full = "★" * int(r)
        half = "½" if r - int(r) >= 0.5 else ""
        empty = "☆" * max(0, 5 - int(r) - (1 if half else 0))
        return full + half + empty

    if yelp_details:
        st.subheader("📖 Yelp Insights")
        left, right = st.columns([2,1])
        with left:
            title = f"**{yelp_details.get('name','')}**"
            if yelp_details.get("price"): title += f" · {yelp_details['price']}"
            if yelp_details.get("is_closed") is True: title += " · _Closed_"
            st.markdown(title)
            st.write(f"**Rating**: {yelp_details.get('rating','—')} {_stars(yelp_details.get('rating'))}  ·  "
                     f"**Reviews**: {yelp_details.get('review_count','—')}")
            cats = ", ".join([c.get("title","") for c in yelp_details.get("categories",[])]) \
                   if isinstance(yelp_details.get("categories"), list) else yelp_details.get("categories","")
            st.write(f"**Categories**: {cats or '—'}")
            address = " • ".join((yelp_details.get("location",{}) or {}).get("display_address", []))
            if not address:
                address = ", ".join(filter(None, [
                    (yelp_details.get("location",{}) or {}).get("address1",""),
                    (yelp_details.get("location",{}) or {}).get("city",""),
                    (yelp_details.get("location",{}) or {}).get("state",""),
                    (yelp_details.get("location",{}) or {}).get("zip_code",""),
                ]))
            st.write(f"**Address**: {address or '—'}")
            tx = yelp_details.get("transactions") or []
            if tx: st.write("**Transactions**: " + ", ".join(tx))
            st.write(f"**Phone**: {yelp_details.get('display_phone') or yelp_details.get('phone') or '—'}")
            if yelp_details.get("url"): st.write(f"[Open on Yelp]({yelp_details['url']})")

            # Hours (today)
            if yelp_details.get("hours"):
                todays = yelp_details["hours"][0].get("open", [])
                if todays:
                    import datetime as _dt
                    dow = _dt.datetime.today().weekday()
                    todays_spans = [o for o in todays if o.get("day") == dow]
                    st.markdown("**Hours (today)**")
                    if todays_spans:
                        for span in todays_spans:
                            s, e = span.get("start",""), span.get("end","")
                            st.write(f"{s[:2]}:{s[2:]}–{e[:2]}:{e[2:]}" if len(s)>=4 and len(e)>=4 else "—")
                    else:
                        st.write("—")

            # Reviews
            reviews = yelp_details.get("top_reviews") or []
            if reviews:
                st.markdown("**Recent Reviews**")
                for r in reviews:
                    author = (r.get("user") or {}).get("name", "Anonymous")
                    rr = r.get("rating", "?")
                    txt = (r.get("text","") or "").strip()
                    st.write(f"• _{author}_ — {rr}⭐: {txt}")

        with right:
            photos = yelp_details.get("photos") or []
            for p in photos[:3]:
                if p: st.image(p, use_container_width=True)
    elif not YELP_API_KEY:
        st.info("Add a Yelp API key in Streamlit Secrets to enable Yelp Insights.")

    # --- Macro Panel (FRED) ---
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

    # --- Predict & Analyze ---
    if st.button("📊 Predict & Analyze"):
        try:
            model_feats, aux_feats = build_feature_vector(
                image, coords, yelp_details, zip_code, fred=macros
            )
            model = load_real_data_model()
            pred = hybrid_prediction(model, model_feats, aux_feats)
            st.markdown(f"## 💰 Predicted Weekly Sales: **${pred:,.2f}**")

            save_prediction(store, coords, pred, st.session_state['aux_feats'])
            plot_insights(store)

            st.subheader("📦 Strategy Recommendations")
            for r in generate_recommendations(
                store, store_type,
                st.session_state['aux_feats']['foot'],
                st.session_state['aux_feats']['yelp_sent'],
                pred, macros
            ):
                st.markdown(f"- {r}")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
