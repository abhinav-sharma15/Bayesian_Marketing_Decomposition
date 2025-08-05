import streamlit as st
import pandas as pd
from tabs import upload_tab, trends_tab, decomposition_tab, simulator_tab

st.set_page_config(page_title="Bayesian Marketing Decomposition", layout="wide")

# --- Session States ---
if "data" not in st.session_state:
    st.session_state.data = None
if "selected_countries" not in st.session_state:
    st.session_state.selected_countries = []
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "Bayesian MMM"
if "target_choice" not in st.session_state:
    st.session_state.target_choice = ["Orders", "Revenue"]

# --- Sidebar Input ---
st.sidebar.title("Upload and Settings")
upload_tab()

if st.session_state.data is not None:
    with st.sidebar.expander("Configuration", expanded=True):
        countries = st.session_state.data["Country"].unique().tolist()
        st.session_state.selected_countries = st.multiselect("Select Country(ies)", countries, default=countries[:1])
        st.session_state.model_choice = st.radio("Model Type", ["Bayesian MMM", "XGBoost/LightGBM with SHAP"])
        st.session_state.target_choice = st.multiselect("Target Variable(s)", ["Orders", "Revenue"], default=["Orders"])

# --- Tabs ---
if st.session_state.data is not None and st.session_state.selected_countries:
    tab1, tab2, tab3 = st.tabs(["📈 Temporal Trends", "📊 Feature Decomposition", "🔧 What-if Simulator"])
    with tab1:
        trends_tab()
    with tab2:
        decomposition_tab()
    with tab3:
        simulator_tab()
else:
    st.info("Please upload a dataset and select at least one country to proceed.")
