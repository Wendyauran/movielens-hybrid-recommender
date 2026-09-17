import pandas as pd
import streamlit as st 
from sklearn.feature_extraction.text import CountVectorizer
from surprise import Reader, Dataset

DATA_PATH = "dataset/"

@st.cache_data
def load_data():
    """Load ratings and movies datasets, return the datasets and merged dataframe"""
    df_ratings = pd.read_csv(DATA_PATH + "ratings_collab.csv")
    df_movies = pd.read_csv(DATA_PATH + "movies_collab.csv")
    df_merged = pd.merge(
        df_ratings[["userId", "movieId", "rating"]],
        df_movies[["movieId", "title"]],
        on="movieId",
        how="left"
    )
    return df_ratings, df_movies, df_merged


@st.cache_data
def build_genre_matrix(_df_movies):
    """Build binary genre feature matrix using CountVectorizer"""
    df_movies_genre = _df_movies[_df_movies["genres"] != "(no genres listed)"].copy().reset_index(drop=True)
    
    vect = CountVectorizer(tokenizer=lambda x: x.split("|"), token_pattern=None)
    df_genre_matrix = vect.fit_transform(df_movies_genre["genres"])
    genre_cols = vect.get_feature_names_out()

    df_genre_matrix = pd.DataFrame(
        df_genre_matrix.toarray(),
        columns=genre_cols
    )
    df_genre_matrix = pd.concat([df_movies_genre[["movieId", "title"]], df_genre_matrix], axis=1)
    return df_genre_matrix, genre_cols


@st.cache_data
def load_surprise_dataset(_df_merged):
    reader_data = Reader(rating_scale=(0.5,5))
    recom_data = Dataset.load_from_df(_df_merged[["userId", "movieId", "rating"]], reader_data)
    return recom_data


@st.cache_data
def get_movie_display_titles(_df_genre_matrix):
    """Return df_genre_matrix with with disambiguated display titles for duplicate movie names"""
    title_counts = _df_genre_matrix["title"].value_counts()
    
    df_display = _df_genre_matrix[["movieId", "title"]].copy()
    df_display["display_title"] = df_display.apply(
        lambda row: f"{row["title"]} [id:{row["movieId"]}]"
        if title_counts[row["title"]] > 1 else row["title"],
        axis=1
    )
    return df_display