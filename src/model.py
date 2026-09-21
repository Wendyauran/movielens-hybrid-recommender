import os
import pickle
import urllib.request
import streamlit as st

MODEL_URL = "https://github.com/Wendyauran/movielens-hybrid-recommender/releases/download/model-knn-v1/best_knn_item.pkl"
MODEL_PATH = "models/best_knn_item.pkl"

def _download_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    tmp_path = MODEL_PATH + ".part"
    urllib.request.urlretrieve(MODEL_URL, tmp_path)
    os.replace(tmp_path, MODEL_PATH)

@st.cache_resource
def load_model():
    """Load pre-trained KNNBaseline model, downloading it from the GitHub release on first run"""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading pre-trained model (first run only)..."):
            _download_model()
    with st.spinner("Loading recommendation model..."):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
