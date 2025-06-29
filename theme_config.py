import streamlit as st

# Sidebar config
st.sidebar.title("🧭 Navigation")
store_type = st.sidebar.selectbox("Filter by Store Type", ["Any", "Cafe", "Boutique", "Restaurant", "Fitness", "Retail"])

# Theme toggle
theme_mode = st.sidebar.radio("Choose Theme", ["Light", "Dark"], index=0)

# Apply theme via CSS
if theme_mode == "Dark":
    st.markdown("""
        <style>
        .main {
            background-color: #1e1e1e;
            color: white;
        }
        .stButton>button {
            background-color: #444;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .main {
            background-color: #f9fbfd;
        }
        </style>
    """, unsafe_allow_html=True)

# Save in session state for main app to read
st.session_state["store_type"] = store_type
st.session_state["theme"] = theme_mode
