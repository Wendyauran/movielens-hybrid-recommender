import streamlit as st
from src.data_loader import load_data


df_ratings, df_movies, df_merged = load_data()

st.title("🎬 Movie Recommendation System")
st.markdown(
    "Personalized movie recommendations using **Collaborative Filtering**, **Content-Based Filtering**, and **Hybrid Recommendation** " \
    "— built on the [MovieLens](https://grouplens.org/datasets/movielens/) dataset."
)

st.divider()

st.subheader("Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Users", f"{df_merged["userId"].nunique():,}")
col2.metric("Total Movies", f"{df_merged["movieId"].nunique():,}")
col3.metric("Total Ratings", f"{len(df_ratings):,}")
col4.metric("Avg Rating", f"{df_ratings["rating"].mean():.2f} / 5.0")

st.divider()

st.subheader("How It Works")
col1, col2, col3 = st.columns(3)
with col1:
    st.info("### 🧑‍🤝‍🧑 Collaborative Filtering\n\n"
            "Recommends movies based on **rating patterns** from similar users. The more users agree on past movies, the better the predictions.\n\n"
            "**Model:** KNNBaseline Item-Based\n\n"
            "**RMSE:** 0.8497")
    
with col2:
    st.success("### 🏷️ Content-Based Filtering\n\n"
               "Builds a **genre preference profile** from your watch history "
               "and scores unseen movies by genre similarity.\n\n"
               "**Method:** Weighted dot product\n\n"
               "**Features:** 19 genres")
    
with col3:
    st.warning("### ⚖️ Hybrid Recommendation\n\n"
               "Combines both approaches with a **weighted average** — "
               "balancing rating quality signals with genre relevance.\n\n"
               "**Formula:** α × Collaborative + (1-α) × Content\n\n"
               "**Default α:** 0.6")
    
st.divider()

st.subheader("Get Started")
st.markdown("Enter your **User ID** to recieve personalized recommendations across all three methods.")
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("👤 Get Recommendations", type="primary", width="stretch"):
        st.switch_page("pages/2_recommendations.py")
        
with col2:
    if st.button("🔍 Explore Movies", width="stretch"):
        st.switch_page("pages/3_movie_explorer.py")