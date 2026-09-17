import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

def get_user_profile(user_id, df_merged, df_genre_matrix, genre_cols):
    user_watched_movies = df_merged[df_merged["userId"] == user_id][["movieId", "title", "rating"]]
    watched_ids = user_watched_movies["movieId"].tolist()

    genre_watched = df_genre_matrix[df_genre_matrix["movieId"].isin(watched_ids)].copy()
    genre_watched = genre_watched.merge(user_watched_movies[["movieId", "rating"]], on="movieId")

    weighted_genres = genre_watched[genre_cols].multiply(genre_watched["rating"].values, axis=0)
    total = weighted_genres.sum().sum()
    user_profile = weighted_genres.sum() / total if total > 0 else weighted_genres.sum()
    
    return user_watched_movies, watched_ids, user_profile

    
def get_content_recommendations(df_genre_matrix, watched_ids, genre_cols, user_profile, top_n=10):
    """Score unwatched movies via dot product with user genre profile and return Top-N content-based recommendations"""
    user_unwatched_movies = df_genre_matrix[~df_genre_matrix["movieId"].isin(watched_ids)].copy().reset_index(drop=True)
    content_score = user_unwatched_movies[genre_cols].values.dot(user_profile.values)

    df_recom_content = pd.DataFrame({
        "movieId": user_unwatched_movies["movieId"].values,
        "title": user_unwatched_movies["title"].values,
        "content_score": content_score
    }).sort_values("content_score", ascending=False).reset_index(drop=True)
    return df_recom_content.head(top_n) if top_n else df_recom_content


def get_collab_recommendations(user_id, watched_ids, df_merged, model, top_n=10):
    """Top-N collaborative filtering predictions for a user"""
    all_movies = df_merged.drop_duplicates("movieId").set_index("movieId")["title"]
    title_counts = all_movies.value_counts()

    def disambiguate(movie_id, title):
        return f"{title} [id:{movie_id}]" if title_counts[title] > 1 else title

    unwatched_ids = [mid for mid in all_movies.index if mid not in set(watched_ids)]
    collab_score = [model.predict(user_id, mid).est for mid in unwatched_ids]

    df_recom_collab = pd.DataFrame({
        "movieId": unwatched_ids,
        "title": [disambiguate(mid, all_movies[mid]) for mid in unwatched_ids],
        "collab_score": collab_score 
    }).sort_values("collab_score", ascending=False).reset_index(drop=True)
    return df_recom_collab.head(top_n) if top_n else df_recom_collab


def get_hybrid_recommendations(recom_content, recom_collab, alpha=0.6, top_n=10):
    """Combine content-based and collaborative score with weighted average"""
    merged = recom_content[["movieId", "title", "content_score"]].merge(
        recom_collab[["movieId", "collab_score"]],
        on="movieId",
        how="inner"
    )
    if merged.empty:
        return merged

    scaler = MinMaxScaler()
    merged[["content_score_norm", "collab_score_norm"]] = scaler.fit_transform(merged[["content_score", "collab_score"]])
    merged["hybrid_score"] = (
        (1 - alpha) * merged["content_score_norm"] + 
        alpha * merged["collab_score_norm"]
    )
    df_recom_hybrid = merged[["movieId", "title", "content_score", "collab_score", "hybrid_score"]].sort_values("hybrid_score", ascending=False).head(10).reset_index(drop=True)
    return df_recom_hybrid


def get_similar_movies(movie_id, df_genre_matrix, genre_cols, top_n=10):
    """Find similar movies via cosine similarity"""
    if movie_id not in df_genre_matrix["movieId"].values:
        return pd.DataFrame()

    movies_target = df_genre_matrix[df_genre_matrix["movieId"] == movie_id][genre_cols].values
    all_genre_vecs = df_genre_matrix[genre_cols].values

    cosine_score = cosine_similarity(all_genre_vecs, movies_target).flatten()

    results = df_genre_matrix[["movieId", "title"]].copy()
    results["similarity"] = cosine_score

    df_recom_cosine = results[results["movieId"] != movie_id].sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)
    return df_recom_cosine


def get_cold_start_recommendations(selected_genres, genre_cols, df_genre_matrix, top_n=10):
    """Content-based recommendations for new users based on selected genres"""
    profile_user = pd.Series(0.0, index=genre_cols)
    for genre in selected_genres:
        if genre in profile_user.index:
            profile_user[genre] = 1.0

    total = profile_user.sum()
    profile_user = profile_user / total if total > 0 else profile_user

    result_scores = df_genre_matrix[genre_cols].values.dot(profile_user.values)
    df_recom_cold = pd.DataFrame({
        "movieId": df_genre_matrix["movieId"].values,
        "title": df_genre_matrix["title"].values,
        "content_score": result_scores
    }).sort_values("content_score", ascending=False).head(top_n).reset_index(drop=True)
    return df_recom_cold