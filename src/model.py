import streamlit as st
from surprise import KNNBaseline

@st.cache_resource
def load_model(_surprise_data):
    """Train KNNBaseline Item-Based model on full dataset and cache in memory. Only runs once per server session"""
    with st.spinner("Loading recommendation model..."):
        model = KNNBaseline(
            k=45,
            min_k=9,
            sim_options={"name": "pearson_baseline", "user_based": False}
        )
        model.fit(_surprise_data.build_full_trainset())
    return model