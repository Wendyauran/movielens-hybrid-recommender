import streamlit as st
import plotly.express as px
from src.data_loader import load_data, build_genre_matrix, load_surprise_dataset
from src.model import load_model
from src.recommender import get_user_profile, get_content_recommendations, get_collab_recommendations, get_hybrid_recommendations, get_cold_start_recommendations


df_ratings, df_movies, df_merged = load_data()
df_genre_matrix, genre_cols = build_genre_matrix(df_movies)
surprise_data = load_surprise_dataset(df_merged)
model = load_model(surprise_data)

st.title("🎬 Movie Recommendations")
st.markdown("Enter your **User ID** to get personalized recommendations.")

col_input, col_btn = st.columns([3, 1], vertical_alignment="bottom")
with col_input:
    user_id = st.number_input("User ID", min_value=1, max_value=1000, value=1, step=1)

with col_btn:
    get_recom = st.button("Get Recommendations", type="primary", width="stretch")

valid_user_id = set(df_merged["userId"].unique())

if get_recom:
    if user_id in valid_user_id:
        with st.spinner("Computing recommendations..."):
            user_watched_movies, watch_ids, user_profile = get_user_profile(user_id, df_merged, df_genre_matrix, genre_cols)
            recom_collab_all = get_collab_recommendations(user_id, watch_ids, df_merged, model, top_n=None)
            recom_content_all = get_content_recommendations(df_genre_matrix, watch_ids, genre_cols, user_profile, top_n=None)

            st.session_state.update({
                "user_id": user_id,
                "is_new_user": False,
                "user_watched_movies": user_watched_movies,
                "watch_ids": watch_ids,
                "user_profile": user_profile,
                "recom_collab_all": recom_collab_all,
                "recom_content_all": recom_content_all
            })
    else:
        st.session_state.update({
            "user_id": user_id,
            "is_new_user": True
        })

if "user_id" not in st.session_state:
    st.info("Enter a **User ID** above and click **Get Recommendations** to start.")
    st.stop()

st.divider()

if not st.session_state["is_new_user"]:
    current_user = st.session_state["user_id"]
    user_watched_movies = st.session_state["user_watched_movies"]
    user_profile = st.session_state["user_profile"]
    recom_collab_all = st.session_state["recom_collab_all"]
    recom_content_all = st.session_state["recom_content_all"]

    st.subheader(f"👤 User {user_id} Profile")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Movies Rated", len(user_watched_movies))
    col2.metric("Avg Rating", f"{user_watched_movies["rating"].mean():.2f}")
    col3.metric("Highest Rated", f"{user_watched_movies["rating"].max():.1f}")
    col4.metric("Lowest Rated", f"{user_watched_movies["rating"].min():.1f}")

    with st.expander("📊 Genre Preference Profile", expanded=True):
        top_genres = user_profile[user_profile > 0].sort_values(ascending=False).head(10)
        top_genres_df = top_genres.reset_index()
        top_genres_df.columns = ["Genre", "Weight"]
        top_genres_df["Genre"] = top_genres_df["Genre"].str.capitalize()

        fig = px.bar(top_genres_df, x="Genre", y="Weight", color_discrete_sequence=["#FBBF24"])
        fig.update_traces(hovertemplate="Genre: %{x}<br>Preference: %{y:.1%}<extra></extra>")
        fig.update_layout(
            plot_bgcolor="#18181B", 
            paper_bgcolor="#18181B",
            font_color="#FAFAFA",
            xaxis=dict(gridcolor="#3F3F46", title="Genre"),
            yaxis=dict(gridcolor="#3F3F46", title="Profile Weight"),
            margin=dict(t=20, b=20),
            hoverlabel=dict(bgcolor="#27272A", font_color="#FAFAFA", bordercolor="#2DD4BF", font_size=13)
        )
        st.plotly_chart(fig, width="stretch")

    st.divider()

    st.subheader("🎬 Recommendations")
    top_n = st.slider("Top N recommendations", min_value=5, max_value=30, value=10, step=5)

    tab1, tab2, tab3 = st.tabs(["🤝 Collaborative", "🎭 Content-Based", "⚡ Hybrid"])
    with tab1:
        st.markdown("Predicts ratings based on **patterns from similar users**.")
        df_show = recom_collab_all.head(top_n)[["title", "collab_score"]].copy()
        df_show.columns = ["Movie Title", "Predicted Rating"]
        st.dataframe(df_show, width="stretch", hide_index=True, column_config={"Predicted Rating": st.column_config.NumberColumn(format="%.4f")})
        
    with tab2:
        st.markdown("Matches movies to your **genre preference profile**.")
        df_show = recom_content_all.head(top_n)[["title", "content_score"]].copy()
        df_show.columns = ["Movie Title", "Content Score"]
        st.dataframe(df_show, width="stretch", hide_index=True, column_config={"Content Score": st.column_config.NumberColumn(format="%.4f")})

    with tab3:
        st.markdown("Combines both methods using a **weighted average**.")

        col_alpha, _ = st.columns([2, 3])
        with col_alpha:
            alpha = st.slider(
                "α — Collaborative weight", 
                min_value=0.0, max_value=1.0, value=0.6, step=0.1, 
                help="α = 1.0 → pure collaborative | α = 0.0 → pure content-based"
            )

        recom_hybrid = get_hybrid_recommendations(recom_content_all, recom_collab_all, alpha=alpha, top_n=None)
        
        if recom_hybrid.empty:
            st.warning("Hybrid merge returned no results. Try adjusting α.")
        else:
            df_show = recom_hybrid.head(top_n)[["title", "collab_score", "content_score", "hybrid_score"]].copy()
            df_show.columns = ["Movie Title", "Collaborative", "Content", "Hybrid Score"]
            st.dataframe(df_show, width="stretch", hide_index=True,
                column_config={
                    "Collaborative": st.column_config.NumberColumn(format="%.4f"),
                    "Content": st.column_config.NumberColumn(format="%.4f"),
                    "Hybrid Score": st.column_config.NumberColumn(format="%.4f")
            })

else:
    current_user = st.session_state["user_id"]
    st.info(f"👋 Welcome, new user!")
    st.markdown(f"Select your **favorite genres** to get content-based recommendations:")

    selected_genres = st.multiselect("Choose genres", options=sorted(genre_cols.tolist()), placeholder="Pick at least one genre...")

    if not selected_genres:
        st.warning("Please select at least one genre to get recommendations.")
        st.stop()

    top_n_cold = st.slider("Top N recommendations", min_value=5, max_value=30, value=10, step=5, key="top_n_cold")
    recom_cold = get_cold_start_recommendations(selected_genres, genre_cols, df_genre_matrix, top_n=None)

    st.subheader("🎯 Recommendations For You")
    df_show = recom_cold.head(top_n_cold)[["title", "content_score"]].copy()
    df_show.columns = ["Movie Title", "Genre Match Score"]
    st.dataframe(df_show, width="stretch", hide_index=True, column_config={"Genre Match Score": st.column_config.NumberColumn(format="%.4f")})