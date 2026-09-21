import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_loader import load_data, build_genre_matrix


df_ratings, df_movies, df_merged = load_data()
df_genre_matrix, genre_cols = build_genre_matrix(df_movies)

LAYOUT = dict(
    plot_bgcolor="#18181B", 
    paper_bgcolor="#18181B",
    font_color="#FAFAFA",
    xaxis=dict(gridcolor="#3F3F46"),
    yaxis=dict(gridcolor="#3F3F46"),
    margin=dict(t=30, b=30),
    hoverlabel=dict(bgcolor="#27272A", font_color="#FAFAFA", bordercolor="#2DD4BF", font_size=13)
)

st.title("📊 Analytics")
st.markdown("Dataset overview, rating patterns, and model performance.")

st.divider()

st.subheader("Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Users", f"{df_merged["userId"].nunique():,}")
col2.metric("Total Movies", f"{df_merged["movieId"].nunique():,}")
col3.metric("Total Ratings", f"{len(df_ratings):,}")
col4.metric("Avg Rating", f"{df_ratings["rating"].mean():.2f} / 5.0")

st.divider()

col_rating, col_activity = st.columns(2)
with col_rating:
    st.subheader("Rating Distribution")
    rating_counts = df_ratings["rating"].value_counts().sort_index().reset_index()
    rating_counts.columns = ["Rating", "Count"]

    fig1 = px.bar(rating_counts, x="Rating", y="Count", color_discrete_sequence=["#FBBF24"])
    fig1.update_traces(hovertemplate="Rating: %{x}<br>Count: %{y:,}<extra></extra>")
    fig1.update_layout(bargap=0.2, **LAYOUT)
    st.plotly_chart(fig1, width="stretch")

with col_activity:
    st.subheader("User Activity Distribution")
    user_activity = df_ratings.groupby("userId").size().reset_index()
    user_activity.columns = ["userId", "Ratings Count"]

    fig2 = px.histogram(user_activity, x="Ratings Count", nbins=30, color_discrete_sequence=["#FB923C"])
    fig2.update_traces(hovertemplate="Ratings given: %{x}<br>Users: %{y}<extra></extra>")
    fig2.update_layout(**LAYOUT)
    st.plotly_chart(fig2, width="stretch")

st.divider()

st.subheader("Top 10 Genre Popularity")
st.markdown("Number of movies per genre in the dataset.")

genre_counts = df_genre_matrix[genre_cols].sum().sort_values().head(10).reset_index()
genre_counts.columns = ["Genre", "Movie Count"]
genre_counts["Genre"] = genre_counts["Genre"].str.capitalize()

fig3 = px.bar(genre_counts, x="Movie Count", y="Genre", orientation="h", color="Movie Count", color_continuous_scale=["#78350F", "#92400E", "#B45309", "#D97706", "#F59E0B", "#FBBF24", "#FCD34D", "#FDE68A"])
fig3.update_traces(hovertemplate="Genre: %{y}<br>Movie count: %{x:,}<extra></extra>")
fig3.update_layout(**LAYOUT)
st.plotly_chart(fig3, width="stretch")

st.divider()

st.subheader("Model Performance")
st.markdown("KNNBaseline Item-Based — RMSE before and after hyperparameter tuning.")

perf_data = pd.DataFrame({
    "Configuration": [
        "Default (k=40, min_k=1, MSD)",
        "Tuned (k=45, min_k=9, pearson_baseline)"
    ],
    "RMSE": [0.8532, 0.8497],
    "Improvement vs Default": ["-", "🔻0.41%"]
})
st.dataframe(perf_data, width="stretch", hide_index=True)

