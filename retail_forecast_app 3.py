# ✅ Retail AI App — Robbinsville‑optimized: Yelp + ZIP + Satellite (Google → Esri fallback) + FRED + Strategy

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
try:
    from torchvision import transforms
    from torchvision.models import resnet18, ResNet18_Weights
    _TV_AVAILABLE = True
except Exception:
    _TV_AVAILABLE = False
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
import time
from math import radians, sin, cos, asin, sqrt, log, tan, pi
import random
from datetime import datetime, timedelta

load_dotenv()
st.set_page_config(page_title="Retail AI Platform", layout="wide")

# =========================
# Helpers: Secrets & Logging
# =========================

def _try_secrets(path_list):
    cur = st.secrets if hasattr(st, "secrets") else {}
    for key in path_list:
        try:
            cur = cur[key]
        except Exception:
            return None
    return cur

def get_secret(name):
    val = _try_secrets([name]) or _try_secrets(["api", name]) or _try_secrets(["general", name])
    if not val:
        val = os.getenv(name)
    return val

def _mask(s):
    return s[:4] + "…" + s[-4:] if s and len(s) > 8 else "(unset)"

# --- API Keys ---
YELP_API_KEY = get_secret("YELP_API_KEY")
GOOGLE_MAPS_API_KEY = get_secret("GOOGLE_MAPS_API_KEY")
FRED_API_KEY = get_secret("FRED_API_KEY")
YELP_HEADERS = {"Authorization": f"Bearer {YELP_API_KEY}"} if YELP_API_KEY else {}

# --- Robbinsville priors ---
DEFAULT_TOWN = "Robbinsville"
DEFAULT_STATE = "NJ"
DEFAULT_ZIP = "08691"
ROBBINSVILLE_CENTER = (40.2147, -74.6129)

# --- Sidebar ---
st.sidebar.title("🔧 Settings")
store_type = st.sidebar.selectbox("Store Type", ["Any", "Coffee Shop", "Boutique", "Fast Food", "Other"], index=3)
theme = st.sidebar.radio("Theme", ["Light", "Dark"], index=1)
with st.sidebar.expander("🔌 Integrations"):
    st.caption(f"Google Maps: {_mask(GOOGLE_MAPS_API_KEY) if GOOGLE_MAPS_API_KEY else '(missing)'}")
    st.caption(f"Yelp: {_mask(YELP_API_KEY) if YELP_API_KEY else '(missing)'}")
    st.caption(f"FRED: {_mask(FRED_API_KEY) if FRED_API_KEY else '(missing)'}")
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = (theme == "Dark")
if st.sidebar.button("🥩 Clear Sales History"):
    if os.path.exists("sales_history.csv"): os.remove("sales_history.csv"); st.sidebar.success("History cleared.")

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
        .stButton>button {{ background-color: #6366f1; color: white; }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)
set_theme()

# --- Geo helpers ---
STATE_ABBR = {"new jersey":"NJ","nj":"NJ"}

def norm_state(s: str) -> str:
    if not s: return ""
    k = s.strip().lower()
    return STATE_ABBR.get(k, s.upper() if len(s)==2 else s)

# =========================
# Geocoding (Robbinsville‑biased)
# =========================
@st.cache_data(show_spinner=False)
def geocode_google(query: str):
    if not GOOGLE_MAPS_API_KEY: return None
    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        r = requests.get(url, params={"address": query, "key": GOOGLE_MAPS_API_KEY, "region": "us"}, timeout=12)
        js = r.json()
        if js.get("status") == "OK" and js.get("results"):
            loc = js["results"][0]["geometry"]["location"]
            return float(loc["lat"]), float(loc["lng"]), js["results"][0].get("formatted_address", query)
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def geocode_nominatim(query: str):
    geolocator = Nominatim(user_agent="retail_ai_locator")
    for _ in range(2):
        try:
            hit = geolocator.geocode(query + ", Robbinsville, NJ, USA", exactly_one=True, timeout=10)
            if hit:
                return float(hit.latitude), float(hit.longitude), hit.address
        except (GeocoderTimedOut, GeocoderUnavailable):
            time.sleep(1)
        except Exception:
            break
    return None

ZIP_FALLBACKS = {DEFAULT_ZIP: (ROBBINSVILLE_CENTER[0], ROBBINSVILLE_CENTER[1], f"{DEFAULT_ZIP} centroid")}

@st.cache_data(show_spinner=False)
def zip_centroid(zip_code: str):
    if not zip_code: return None
    z = geocode_google(f"{zip_code}, USA")
    if z: return z
    z = geocode_nominatim(f"{zip_code}, USA")
    if z: return z
    if zip_code in ZIP_FALLBACKS: return ZIP_FALLBACKS[zip_code]
    return None

# =========================
# Yelp API (Robbinsville‑biased search)
# =========================
@st.cache_data(show_spinner=False)
def _yelp_get(path, params=None):
    if not YELP_API_KEY: return None
    try:
        r = requests.get(f"https://api.yelp.com/v3{path}", headers=YELP_HEADERS, params=params or {}, timeout=12)
        if r.status_code == 200: return r.json()
        return None
    except requests.RequestException:
        return None

@st.cache_data(show_spinner=False)
def _yelp_search(term, location=None, coords=None, limit=3):
    if not YELP_API_KEY or not term: return []
    url = "https://api.yelp.com/v3/businesses/search"
    params = {"term": term, "limit": limit, "sort_by": "best_match"}
    if coords:
        params.update({"latitude": float(coords[0]), "longitude": float(coords[1]), "radius": 10000})
    else:
        params["location"] = location or "Robbinsville, NJ"
    try:
        r = requests.get(url, headers=YELP_HEADERS, params=params, timeout=12)
        if r.status_code == 200: return (r.json() or {}).get("businesses", [])
        return []
    except requests.RequestException:
        return []

@st.cache_data(show_spinner=False)
def _yelp_matches(name, city=None, state=None, zip_code=None):
    p = {"name": name, "country": "US", "match_threshold": "default", "city": city or DEFAULT_TOWN, "state": norm_state(state or DEFAULT_STATE), "zip_code": zip_code or DEFAULT_ZIP}
    js = _yelp_get("/businesses/matches", params=p)
    if not js: return []
    return js.get("businesses", [])

@st.cache_data(show_spinner=False)
def _yelp_details(business_id):
    return _yelp_get(f"/businesses/{business_id}") or {}

@st.cache_data(show_spinner=False)
def _yelp_reviews(business_id):
    js = _yelp_get(f"/businesses/{business_id}/reviews", params={"locale": "en_US"}) or {}
    return js.get("reviews", [])[:3]

@st.cache_data(show_spinner=False)
def find_yelp_business(name, zip_code=None, town=None, state=None, coords_hint=None):
    matches = _yelp_matches(name, city=town or DEFAULT_TOWN, state=state or DEFAULT_STATE, zip_code=zip_code or DEFAULT_ZIP)
    if matches:
        bid = matches[0].get("id")
        if bid:
            det = _yelp_details(bid); det["top_reviews"] = _yelp_reviews(bid); return det
    if coords_hint:
        hits = _yelp_search(name, coords=coords_hint, limit=1)
        if hits:
            bid = hits[0].get("id"); det = _yelp_details(bid); det["top_reviews"] = _yelp_reviews(bid); return det
    hits = _yelp_search(name, location=f"{DEFAULT_TOWN}, {DEFAULT_STATE}", limit=1)
    if hits:
        bid = hits[0].get("id"); det = _yelp_details(bid); det["top_reviews"] = _yelp_reviews(bid); return det
    return None

def yelp_sentiment_score(biz):
    if not biz: return 50.0
    rating = biz.get("rating", 3.0) or 3.0
    review_count = biz.get("review_count", 0) or 0
    return round(min(100, max(0, (rating - 3) * 25 + review_count * 0.1)), 1)

# --- FRED Macros ---
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

@st.cache_data(show_spinner=False)
def fred_series(series_id, api_key=FRED_API_KEY, months=24):
    if not api_key: return pd.Series(dtype=float)
    start_date = (datetime.today() - timedelta(days=31*months)).strftime("%Y-%m-%d")
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json", "observation_start": start_date}
    try:
        r = requests.get(FRED_BASE, params=params, timeout=12); r.raise_for_status()
        data = r.json().get("observations", [])
        dates, vals = [], []
        for o in data:
            v = o.get("value"); if v in (None, "", "."): continue
            try: vals.append(float(v)); dates.append(pd.to_datetime(o["date"]))
            except: continue
        if not vals: return pd.Series(dtype=float)
        return pd.Series(vals, index=pd.to_datetime(dates)).sort_index()
    except Exception:
        return pd.Series(dtype=float)

@st.cache_data(show_spinner=False)
def fred_features():
    series_map = {"RRSFS": "RRSFS", "UMCSENT": "UMCSENT", "UNRATE": "UNRATE"}
    out = {}
    for key, sid in series_map.items():
        s = fred_series(sid)
        if s.empty: out[key] = {"latest": None, "yoy": None, "z": 0.0}; continue
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

# =========================
# Location Resolver (Robbinsville‑biased)
# =========================
@st.cache_data(show_spinner=False)
def resolve_store_location(store, zip_code, town, state):
    town = town or DEFAULT_TOWN
    state = norm_state(state or DEFAULT_STATE)
    zip_code = zip_code or DEFAULT_ZIP
    candidates = []
    yelp = find_yelp_business(store, zip_code=zip_code, town=town, state=state)
    if yelp and (yelp.get("coordinates") or {}):
        c = yelp.get("coordinates"); lat, lon = c.get("latitude"), c.get("longitude")
        if lat and lon:
            label = f"Yelp: {yelp.get('name','')} — {', '.join((yelp.get('location',{}) or {}).get('display_address',[]) or [])}"
            candidates.append((float(lat), float(lon), label))
    # Geocode within Robbinsville bounds
    q_variants = [
        f"{store}, {town}, {state}, {zip_code}, USA",
        f"{store}, {town}, {state}, USA",
        f"{store}, {zip_code}, USA",
    ]
    for q in q_variants:
        g = geocode_google(q) or geocode_nominatim(q)
        if g:
            lat, lon, addr = g; cand = (lat, lon, f"Geocode: {addr}")
            if cand not in candidates: candidates.append(cand); break
    zc = zip_centroid(zip_code)
    if zc:
        lat, lon, lab = zc; candidates.append((lat, lon, lab))
    if not candidates:
        candidates = [(ROBBINSVILLE_CENTER[0], ROBBINSVILLE_CENTER[1], "Robbinsville default")]
    return candidates, yelp

# =========================
# Map UI
# =========================

def show_map_with_selection(options, *, show_radius_m=400, key="storemap"):
    st.subheader("📍 Select Your Store Location")
    m = folium.Map(location=[options[0][0], options[0][1]], zoom_start=14, control_scale=True)
    for lat, lon, label in options:
        folium.Marker(location=[lat, lon], tooltip=label, popup=label, icon=folium.Icon(color="red", icon="info-sign")).add_to(m)
        if show_radius_m:
            folium.Circle(radius=show_radius_m, location=[lat, lon], color="#FF4136", fill=True, fill_opacity=0.08).add_to(m)
    state = st_folium(m, height=380, width=None, key=key)
    chosen = options[0][:2]
    if state and state.get("last_clicked"):
        chosen = (state["last_clicked"]["lat"], state["last_clicked"]["lng"])
        st.caption("Tip: map click captured — using your clicked point for all downstream steps.")
    return chosen

# =========================
# Satellite (Google → Esri fallback to avoid 403) & Features
# =========================

# Web Mercator helpers for Esri export
R = 6378137.0

def _wmerc_from_latlon(lat, lon):
    x = R * radians(lon)
    y = R * log(tan(pi/4 + radians(lat)/2))
    return x, y

def _bbox_around(lat, lon, zoom=18, px_w=640, px_h=400):
    # meters per pixel at latitude
    mpp = 156543.03392 * cos(radians(lat)) / (2 ** zoom)
    half_w = px_w * mpp / 2
    half_h = px_h * mpp / 2
    cx, cy = _wmerc_from_latlon(lat, lon)
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h

@st.cache_data(show_spinner=False)
def _google_static_sat(coords, size=(640,400), zoom=18):
    if not GOOGLE_MAPS_API_KEY: return None
    try:
        url = (
            f"https://maps.googleapis.com/maps/api/staticmap?center={coords[0]},{coords[1]}"
            f"&zoom={zoom}&size={size[0]}x{size[1]}&maptype=satellite&scale=2&key={GOOGLE_MAPS_API_KEY}"
        )
        res = requests.get(url, timeout=12)
        if res.status_code == 200:
            return Image.open(io.BytesIO(res.content)).convert("RGB")
        # Common 403 causes: HTTP referrer restrictions on key. Fall back.
        return None
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def _esri_world_imagery(coords, size=(640,400), zoom=18):
    # Build bbox in EPSG:3857 and call Esri World_Imagery export API (no API key required).
    lat, lon = float(coords[0]), float(coords[1])
    minx, miny, maxx, maxy = _bbox_around(lat, lon, zoom=zoom, px_w=size[0], px_h=size[1])
    url = (
        "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
        f"?bbox={minx},{miny},{maxx},{maxy}&bboxSR=102100&imageSR=102100&size={size[0]},{size[1]}&format=png32&f=image"
    )
    r = requests.get(url, timeout=12)
    if r.status_code == 200:
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    return None

def fetch_or_upload_satellite_image(coords):
    uploaded = st.file_uploader("Upload custom satellite image", type=["jpg","jpeg","png"], key="up_sat")
    if uploaded: return Image.open(uploaded).convert("RGB")
    # Try Google first (if key works); otherwise fall back to Esri imagery to avoid 403
    img = _google_static_sat(coords, size=(640,400), zoom=18)
    if img is None:
        img = _esri_world_imagery(coords, size=(640,400), zoom=18)
    if img is None:
        st.warning("Satellite provider blocked the request; showing placeholder.")
        img = Image.new("RGB", (512, 512), color=(180, 180, 180))
    return img

# Feature extraction

def extract_satellite_features(img):
    if not _TV_AVAILABLE:
        rng = np.random.RandomState(42); return rng.normal(0, 1, 512)
    try:
        try: model = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception: model = resnet18(pretrained=True)
        model.eval(); model = nn.Sequential(*list(model.children())[:-1])
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            features = model(tensor).view(1, -1).numpy().flatten()
        return np.resize(features, 512)
    except Exception:
        rng = np.random.RandomState(123); return rng.normal(0, 1, 512)

# --- Feature builder ---

def yelp_sentiment_score(biz):
    if not biz: return 50.0
    rating = biz.get("rating", 3.0) or 3.0
    review_count = biz.get("review_count", 0) or 0
    return round(min(100, max(0, (rating - 3) * 25 + review_count * 0.1)), 1)


def build_feature_vector(img, coords, yelp_details, zip_code, fred=None):
    sat = extract_satellite_features(img)
    latlon = np.array([coords[0], coords[1]], dtype=float)
    rating = (yelp_details or {}).get("rating", 3.0) or 3.0
    reviews = (yelp_details or {}).get("review_count", 0) or 0
    yelp_sent = yelp_sentiment_score(yelp_details)
    def _foot_from_zip(z):
        if not z: return 0.45
        h = sum(ord(c) for c in str(z)) % 10
        return round(0.3 + 0.05 * h, 2)
    foot = _foot_from_zip(zip_code)
    fred = fred or fred_features()
    rrsfs_z = fred.get("RRSFS", {}).get("z", 0.0) or 0.0
    umcsent_z = fred.get("UMCSENT", {}).get("z", 0.0) or 0.0
    unrate_z = fred.get("UNRATE", {}).get("z", 0.0) or 0.0
    aux = np.array([rating, reviews, yelp_sent, foot, rrsfs_z, umcsent_z, unrate_z], dtype=float)
    model_feats = np.concatenate([sat, latlon], axis=0)
    st.session_state['aux_feats'] = {
        "rating": rating, "reviews": reviews, "yelp_sent": yelp_sent, "foot": foot,
        "rrsfs_z": rrsfs_z, "umcsent_z": umcsent_z, "unrate_z": unrate_z
    }
    return model_feats, aux

# --- Model & Prediction ---

def load_fallback_model():
    dummy = DummyRegressor(strategy="constant", constant=float(np.random.randint(12000, 24000)))
    dummy.fit([[0]*514], [dummy.constant]); return dummy

@st.cache_resource(show_spinner=False)
def load_real_data_model():
    path = "real_sales_data.csv"
    if os.path.exists("model.pkl"):
        try: return joblib.load("model.pkl")
        except Exception: pass
    if not os.path.exists(path): return load_fallback_model()
    try:
        df = pd.read_csv(path)
        X = df[[f"f{i}" for i in range(512)] + ["lat", "lon"]]; y = df["sales"]
        model = GradientBoostingRegressor(); model.fit(X, y); joblib.dump(model, "model.pkl"); return model
    except Exception:
        return load_fallback_model()


def hybrid_prediction(model, model_feats, aux_feats):
    pred = None
    try: pred = float(model.predict([model_feats])[0])
    except Exception: pred = None
    rating, reviews, yelp_sent, foot, rrsfs_z, umcsent_z, unrate_z = aux_feats
    baseline = 15000.0
    rat_mult  = 1.0 + (rating - 4.0) * 0.07
    rev_mult  = 1.0 + min(reviews, 1000) / 10000.0
    buzz_mult = 1.0 + (yelp_sent - 50) / 500.0
    foot_mult = 0.8 + foot
    macro_mult = (1.0 + 0.05 * rrsfs_z) * (1.0 + 0.04 * umcsent_z) * (1.0 - 0.04 * unrate_z)
    heuristic = float(np.clip(baseline * rat_mult * rev_mult * buzz_mult * foot_mult * macro_mult, 3000, 75000))
    if pred is None or np.isnan(pred) or pred < 2000 or pred > 100000: return heuristic
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
            old = pd.read_csv("sales_history.csv", on_bad_lines='skip'); df = pd.concat([old, df], ignore_index=True)
        except: st.warning("Corrupted history file. Overwriting.")
    df.to_csv("sales_history.csv", index=False)


def plot_insights(store):
    if 'sales_history.csv' not in os.listdir(): return st.info("No data yet.")
    try: df = pd.read_csv("sales_history.csv", on_bad_lines='skip')
    except: return st.warning("Could not read history file.")
    dff = df[df["store"].str.lower() == store.lower()].copy()
    if dff.empty: return st.warning("No data found for this store yet.")
    dff["timestamp"] = pd.to_datetime(dff["timestamp"])
    st.subheader("📈 Sales Over Time")
    fig_sales = px.line(dff, x="timestamp", y="sales", title="Weekly Sales Forecast", markers=True); fig_sales.update_traces(line=dict(width=2))
    st.plotly_chart(fig_sales, use_container_width=True)
    st.subheader("👣 Foot Traffic vs. 📱 Online Buzz")
    df_long = dff.melt(id_vars=["timestamp"], value_vars=["foot", "social"], var_name="metric", value_name="score")
    fig_buzz = px.line(df_long, x="timestamp", y="score", color="metric", markers=True); fig_buzz.update_traces(line=dict(width=2))
    st.plotly_chart(fig_buzz, use_container_width=True)
    st.subheader("🏷️ Average Sales by Store Type")
    avg_type = df.groupby("type")["sales"].mean().reset_index(); fig_type = px.bar(avg_type, x="type", y="sales", color="type", title="Avg Weekly Sales per Store Type")
    st.plotly_chart(fig_type, use_container_width=True)

# --- Strategy Engine (Robbinsville dayparts + KPI targets) ---

def generate_recommendations(store, store_type, foot, soc, sales, fred, *, n=8, seed=None):
    rng = random.Random(seed or f"{store}-{int(time.time()//3600)}")
    # Daypart priors for suburban Robbinsville (school/work commute):
    # Quiet: 2–4pm; Peaks: 7–9am (coffee), 12–1:30pm (lunch), 6–8pm (dinner)
    low_foot = [
        "LULL HACK (2–4pm): combo at -10% + receipt QR to join SMS list (goal: +15% tickets in lull).",
        "SCHOOL PARTNER: RHS/elementary flyer w/ code; 10% to PTA (goal: 50 redemptions/mo).",
        "SIDEWALK A-FRAME: 'Today only' rotating offer; track photo-based redemptions.",
    ]
    mid_foot = [
        "HAPPY-HOUR UPGRADE: +$1.50 side on any entree during 2 quiet hours (target: +8% ATV).",
        "SHIFT TEST: open +30 min earlier on Sat/Sun; compare 2-week cohorts.",
        "REFER-A-FRIEND: QR cards; referrer gets free add-on after 3 referrals (CAC <$3).",
    ]
    high_foot = [
        "EXPRESS PICKUP SHELF + PRE-ORDER SIGNAGE (peak dinner): aim -25% wait time.",
        "POS UPSELL BUTTONS: extra sauce +$0.75 / premium side +$2; measure attach rate.",
        "LOYALTY LADDER: 5 visits=free side; 10=signature item (target: 30% enrollment).",
    ]
    weak_buzz = [
        "REVIEW DRIVE: QR on bags; random weekly winner. Aim: +0.2★ over 90 days.",
        "LISTING REFRESH: 6 new photos, accurate hours, pin top items on Yelp/Maps.",
    ]
    mid_buzz = [
        "2 LOCAL REELS/WEEK: staff pick + behind‑the‑scenes; CTA to pre‑order link.",
        "UGC PROMPT: 'Show your combo' hashtag; weekly gift card ($15) giveaway.",
    ]
    high_buzz = [
        "MICRO‑INFLUENCER (Hamilton/Robbinsville 1–5k): unique code for secret item.",
        "VIP HOUR (Thu 7–8pm): followers only secret menu; track code RSVPs.",
    ]
    flavor = {
        "Coffee Shop": ["COMMUTER PRE‑ORDER: ready at :10 and :40; sign at parking exit."],
        "Fast Food": ["FAMILY BUNDLE (Sun–Tue): 4‑person set priced at 1.2× COGS (target: +12% ATV)."],
        "Boutique": ["NEIGHBORHOOD LOOKBOOK wall + QR to featured pieces; host 'bring an item' styling night."],
        "Any": [], "Other": []
    }
    macros = []
    if fred.get("UMCSENT", {}).get("z", 0) < -0.5:
        macros += ["VALUE MESSAGING: bundle + loyalty emphasis while sentiment is soft."]
    if fred.get("UNRATE", {}).get("z", 0) > 0.5:
        macros += ["BUDGET SETS: spotlight <$10 items; keep price fence to lull hours."]
    if fred.get("RRSFS", {}).get("z", 0) > 0.5:
        macros += ["TEST PREMIUM ADD‑ONS: limited sauces/upsizes while retail is strong."]
    base = []
    if sales < 10000:
        base += ["TRIM MENU to best‑sellers; simplify line to 12 SKUs max (goal: +15% throughput)."]
    elif sales > 30000:
        base += ["SECOND‑SITE SCOPING (Rt‑130 corridor): heatmap + competitor spacing."]
    pool = low_foot if foot < 0.4 else mid_foot if foot < 0.6 else high_foot
    pool_buzz = weak_buzz if soc < 40 else mid_buzz if soc < 70 else high_buzz
    recs = set(); recs.update(rng.sample(pool, k=min(3, len(pool)))); recs.update(rng.sample(pool_buzz, k=min(2, len(pool_buzz))))
    fpool = flavor.get(store_type, []) or flavor["Other"]
    if fpool: recs.update(rng.sample(fpool, k=min(1, len(fpool))))
    if base:  recs.update(rng.sample(base,  k=min(1, len(base))))
    if macros: recs.update(rng.sample(macros, k=min(1, len(macros))))
    icons = []
    for item in recs:
        if "review" in item.lower(): icons.append("⭐ " + item)
        elif any(k in item.lower() for k in ["bundle","add-on","upsell"]): icons.append("🧺 " + item)
        elif any(k in item.lower() for k in ["loyal","stamp","VIP".lower()]): icons.append("🎟️ " + item)
        elif any(k in item.lower() for k in ["reels","influencer","ugc"]): icons.append("📣 " + item)
        else: icons.append("💡 " + item)
    return icons[:n]

# --- Main App Execution ---
st.title("📊 Retail AI: Forecast & Strategy — Robbinsville")
store = st.text_input("🏪 Store Name (e.g. House of Wings)", key="store_name")
zip_code = st.text_input("📍 ZIP Code", value=DEFAULT_ZIP, key="zip")
town = st.text_input("🏙️ Town", value=DEFAULT_TOWN, key="town")
state = st.text_input("🏞️ State", value=DEFAULT_STATE, key="state")
coords = None; yelp_details = None

# Always render a map, biased to Robbinsville
if store:
    candidates, yelp_details = resolve_store_location(store, zip_code, town, state)
    coords = show_map_with_selection(candidates, key="map1")
else:
    zc = zip_centroid(zip_code) or (ROBBINSVILLE_CENTER[0], ROBBINSVILLE_CENTER[1], "Robbinsville default")
    lat, lon, lab = zc if isinstance(zc, tuple) else (ROBBINSVILLE_CENTER[0], ROBBINSVILLE_CENTER[1], "Robbinsville default")
    coords = show_map_with_selection([(lat, lon, lab)], key="map_default_zip")

# --- Satellite + Yelp panel ---
if coords:
    image = fetch_or_upload_satellite_image(coords)
    st.image(image, caption="🛰️ Satellite View", use_container_width=True)
    if not yelp_details and store:
        yelp_details = find_yelp_business(store, zip_code=zip_code, town=town, state=state, coords_hint=coords)

    def _stars(rating):
        try: r = float(rating)
        except: return "—"
        full = "★" * int(r); half = "½" if r - int(r) >= 0.5 else ""; empty = "☆" * max(0, 5 - int(r) - (1 if half else 0))
        return full + half + empty

    if yelp_details:
        st.subheader("📖 Yelp Insights")
        left, right = st.columns([2,1])
        with left:
            title = f"**{yelp_details.get('name','')}**"
            if yelp_details.get("price"): title += f" · {yelp_details['price']}"
            if yelp_details.get("is_closed") is True: title += " · _Closed_"
            st.markdown(title)
            st.write(f"**Rating**: {yelp_details.get('rating','—')} {_stars(yelp_details.get('rating'))}  ·  **Reviews**: {yelp_details.get('review_count','—')}")
            cats = ", ".join([c.get("title","") for c in yelp_details.get("categories",[])]) if isinstance(yelp_details.get("categories"), list) else yelp_details.get("categories","")
            st.write(f"**Categories**: {cats or '—'}")
            address = " • ".join((yelp_details.get("location",{}) or {}).get("display_address", []))
            st.write(f"**Address**: {address or '—'}")
            tx = yelp_details.get("transactions") or []
            if tx: st.write("**Transactions**: " + ", ".join(tx))
            st.write(f"**Phone**: {yelp_details.get('display_phone') or yelp_details.get('phone') or '—'}")
            if yelp_details.get("url"): st.write(f"[Open on Yelp]({yelp_details['url']})")
            if yelp_details.get("hours"):
                import datetime as _dt
                todays = yelp_details["hours"][0].get("open", [])
                dow = _dt.datetime.today().weekday()
                todays_spans = [o for o in todays if o.get("day") == dow]
                st.markdown("**Hours (today)**")
                if todays_spans:
                    for span in todays_spans:
                        s, e = span.get("start",""), span.get("end","")
                        st.write(f"{s[:2]}:{s[2:]}–{e[:2]}:{e[2:]}" if len(s)>=4 and len(e)>=4 else "—")
                else:
                    st.write("—")
        with right:
            photos = yelp_details.get("photos") or []
            for p in photos[:3]:
                if p: st.image(p, use_container_width=True)
    elif not YELP_API_KEY:
        st.info("Add a Yelp API key in Secrets to enable Yelp Insights.")

    # Macros
    macros = fred_features() if FRED_API_KEY else {}
    st.subheader("🏦 Macro Snapshot (FRED)")
    if macros:
        col1, col2, col3 = st.columns(3)
        def fmt(v, pct=False):
            if v is None: return "—"
            return f"{v*100:.1f}%" if pct else f"{v:.2f}"
        with col1:
            st.markdown("**RRSFS (Retail & Food Services)**"); st.write("z:", f"{macros['RRSFS']['z']:.2f}" if macros['RRSFS']['z'] is not None else "—"); st.write("YoY:", fmt(macros['RRSFS']['yoy'], pct=True))
        with col2:
            st.markdown("**UMCSENT (Sentiment)**"); st.write("z:", f"{macros['UMCSENT']['z']:.2f}" if macros['UMCSENT']['z'] is not None else "—"); st.write("YoY:", fmt(macros['UMCSENT']['yoy'], pct=True))
        with col3:
            st.markdown("**UNRATE (Unemployment)**"); st.write("z:", f"{macros['UNRATE']['z']:.2f}" if macros['UNRATE']['z'] is not None else "—"); st.write("YoY:", fmt(macros['UNRATE']['yoy'], pct=True))
    else:
        st.info("Add a FRED_API_KEY in Secrets to enable macro-aware modeling.")

    # Predict & Analyze
    if st.button("📊 Predict & Analyze"):
        try:
            model_feats, aux_feats = build_feature_vector(image, coords, yelp_details, zip_code, fred=macros)
            model = load_real_data_model(); pred = hybrid_prediction(model, model_feats, aux_feats)
            st.markdown(f"## 💰 Predicted Weekly Sales: **${pred:,.2f}**")
            save_prediction(store, coords, pred, st.session_state['aux_feats'])
            plot_insights(store)
            st.subheader("📦 Strategy Recommendations (Robbinsville‑tuned)")
            for r in generate_recommendations(store, store_type, st.session_state['aux_feats']['foot'], st.session_state['aux_feats']['yelp_sent'], pred, macros):
                st.markdown(f"- {r}")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

