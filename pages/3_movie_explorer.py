import streamlit as st
from src.data_loader import load_data, build_genre_matrix, load_surprise_dataset, get_movie_display_titles
from src.model import load_model
from src.recommender import get_similar_movies

st.set_page_config(
    page_title="Movie Explorer",
    page_icon="🔍",
    layout="wide"
)

df_ratings, df_movies, df_merged = load_data()
df_genre_matrix, genre_cols = build_genre_matrix(df_movies)
surprise_data = load_surprise_dataset(df_merged)
model = load_model(surprise_data)
df_display = get_movie_display_titles(df_genre_matrix)

st.title("🔍 Movie Explorer")
st.markdown("Search a movie to see its genre, find similar titles, and predict ratings.")

title_to_id = dict(zip(df_display["display_title"], df_display["movieId"]))

col_search, col_n = st.columns([3, 1], vertical_alignment="bottom")
with col_search:
    selected_title = st.selectbox("Search for a movie", options=sorted(df_display["display_title"].tolist()))
with col_n:
    top_n = st.slider("Number of similar movies", min_value=5, max_value=30, value=10, step=5)

selected_movie_id = title_to_id[selected_title]

st.divider()

movie_row = df_genre_matrix[df_genre_matrix["movieId"] == selected_movie_id].iloc[0]
movie_genres = [col.capitalize() for col in genre_cols if movie_row[col] == 1]

st.subheader(f"🎬 {selected_title}")
if movie_genres:
    st.markdown("**Genres:** " + ", ".join(movie_genres))
else:
    st.markdown("**Genres: No genre information available.")

st.divider()

st.subheader("🎭 Similar Movies")
st.markdown("Based on **genre similarity** using cosine similarity.")

df_similar = get_similar_movies(selected_movie_id, df_genre_matrix, genre_cols, top_n=top_n)

if df_similar.empty:
    st.info("No similar movies found.")
else:
    df_similar = df_similar.merge(df_display[["movieId", "display_title"]], on="movieId", how="left")
    df_show = df_similar[["display_title", "similarity"]].copy()
    df_show.columns = ["Movie Title", "Similarity Score"]
    st.dataframe(df_show, width="stretch", hide_index=True)

st.divider()

st.subheader("🤝 Collaborative Rating Prediction")
st.markdown("Enter a **User ID** to see the predicted rating this user would give.")

valid_user_id = set(df_merged["userId"].unique())

col_uid, col_btn = st.columns([2, 1], vertical_alignment="bottom")
with col_uid:
    pred_user_id = st.number_input("User ID", min_value=1, max_value=1000, value=1, step=1, key="pred_user_id")
with col_btn:
    predict = st.button("Predict Rating", type="primary", width="stretch")

if predict:
    if pred_user_id in valid_user_id:
        predicted = model.predict(pred_user_id, selected_movie_id).est
        st.metric(f"Predicted rating for User {pred_user_id}", f"{predicted:.2f} / 5.0")
        if predicted >= 4.0:
            st.success("This user would likely **love** this movie!")
        elif predicted >= 3.0:
            st.info("This user would likely **enjoy** this movie.")
        else:
            st.warning("This movie might **not be the best fit** for this user.")
    else:
        st.warning(f"User ID {pred_user_id} is not in the dataset.")