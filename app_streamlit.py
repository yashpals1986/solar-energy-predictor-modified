import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ===============================  
# Page Config & Theme (CLEAN VERSION)
# ===============================
st.set_page_config(
    page_title="Solar Energy Predictor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        color: #FF6B35 !important;
        text-align: center;
        margin-bottom: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load
