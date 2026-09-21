import streamlit as st


st.title("ℹ️ About This Project")
st.markdown(
    "A production-ready Movie Recommendation System built on the **MovieLens** dataset, "
    "combining three recommendation approaches into a unified Streamlit application."
)

st.divider()

tab1, tab2, tab3 = st.tabs(["🧠 Methodology", "🛠️ Tech Stack", "📦 Dataset"])
with tab1:
    st.subheader("Recommendation Methods")
    with st.expander("🧑‍🤝‍🧑 Collaborative Filtering", expanded=True):
        st.markdown("""
        Recommends movies based on **rating patterns from similar users**.

        **Algorithm:** KNNBaseline — Item-Based  
        **Similarity metric:** Pearson Baseline  
        **Parameters:** k=45 neighbors, min_k=9  
        **RMSE after tuning:** 0.8497

        How it works:
        1. Computes item-to-item similarity from the full rating matrix
        2. For each unseen movie, finds the k most similar items the user has rated
        3. Predicts rating using a baseline-adjusted weighted average
        """)

    with st.expander("🏷️ Content-Based Filtering"):
        st.markdown("""
        Recommends movies based on a **genre preference profile** built from past ratings.

        **Feature extraction:** CountVectorizer on pipe-separated genre tags  
        **Profile building:** Weighted sum of genre vectors (weight = user rating)  
        **Scoring:** Dot product between user profile and movie genre vector

        How it works:
        1. Builds a binary genre matrix (19 genres × 9,724 movies)
        2. Weights each movie's genre vector by the user's rating for it
        3. Normalizes the profile so all genre weights sum to 1
        4. Scores unseen movies via dot product with the profile
        """)

    with st.expander("⚖️ Hybrid Recommendation"):
        st.markdown("""
        Combines collaborative and content-based scores using a **weighted average**.

        **Normalization:** MinMaxScaler — both scores scaled to [0, 1]  
        **Formula:** hybrid = α × collab_norm + (1−α) × content_norm  
        **Default α:** 0.6 (60% collaborative, 40% content-based)

        Normalization is necessary because collaborative scores are predicted ratings (0.5–5.0)
        while content scores are dot products with no fixed scale.
        """)

    with st.expander("❄️ Cold Start"):
        st.markdown("""
        New users with no rating history cannot use collaborative filtering.

        **Flow:**
        1. User selects favorite genres via multiselect
        2. A uniform genre profile is built (equal weight per selected genre)
        3. Movies are scored via dot product with the genre profile
        4. Results are content-based only, until the user rates movies
        """)

with tab2:
    st.subheader("Technology Stack")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**🐍 Python**\n\nCore language for all data processing, modeling, and application logic.")
        st.info("**pandas / numpy**\n\nDataFrame operations, matrix computations, and user profile building.")

    with col2:
        st.success("**🎯 Surprise**\n\nScikit-Surprise for collaborative filtering — KNNBaseline, cross-validation, and hyperparameter tuning via RandomizedSearchCV.")
        st.success("**🔬 scikit-learn**\n\nCountVectorizer for genre matrix, MinMaxScaler for hybrid normalization, cosine_similarity for item-to-item.")

    with col3:
        st.warning("**🌐 Streamlit**\n\nFull-stack web framework — caching, multi-page routing, and all UI components.")
        st.warning("**📈 Plotly Express**\n\nInteractive charts for the Analytics page.")

with tab3:
    st.subheader("Dataset")

    col_info, col_stats = st.columns([2, 1])
    with col_info:
        st.markdown("""
        **MovieLens** is a dataset collected by the GroupLens Research lab at the
        University of Minnesota — a standard benchmark for recommendation systems.

        This project uses the **MovieLens Latest Small** dataset:
        - Ratings collected from real MovieLens users
        - Each user has rated at least 20 movies
        - Rating scale: 0.5 to 5.0 (half-star increments)
        - Genres are pipe-separated tags per movie (e.g. `Action|Comedy|Drama`)
        """)
        st.link_button("📥 MovieLens Dataset", "https://grouplens.org/datasets/movielens/")

    with col_stats:
        st.metric("Users",   "610")
        st.metric("Movies",  "9,724")
        st.metric("Ratings", "100,836")
        st.metric("Genres",  "19")

    st.divider()

    st.subheader("Source Code")
    st.markdown("The full source code for this project is available on GitHub.")
    st.link_button("🔗 View on GitHub", "https://github.com/Wendyauran/movielens-hybrid-recommender")
