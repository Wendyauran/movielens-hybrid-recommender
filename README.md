# Movie Recommendation System — MovieLens

[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://python.org)
[![scikit-surprise](https://img.shields.io/badge/scikit--surprise-1.1.4+-orange)](http://surpriselib.com)

A production-ready movie recommendation web app built on the [MovieLens 100K](https://grouplens.org/datasets/movielens/) dataset, combining three recommendation approaches into a single interactive Streamlit interface.

**Live demo:** [movielens-hybrid-recommender.streamlit.app](https://movielens-hybrid-recommender.streamlit.app)

---

## Preview

### **Home**
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/a2f3b3a3-30df-4938-a30a-55f20ba881de" />

### **Recommendations**
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/bf5bf1d6-79cb-4c84-b1ab-578eb75d9f65" />

### **Analytics**
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/113033ee-a193-4d82-b820-236708451150" />

---

## Background

Recommendation systems are central to how users discover relevant content at scale. 
This project explores three complementary approaches — Collaborative Filtering, Content-Based Filtering, and Hybrid — applied to real movie rating data, and asks:

> *How does each method approach personalized recommendations, and where does each fall short?*

The result is a tuned **KNNBaseline Item-Based** model (RMSE 0.8497) served through an interactive multi-page Streamlit app, with a cold start solution for users without rating history.

---

## Features

- **Collaborative Filtering** — KNNBaseline Item-Based model predicts ratings based on patterns from similar users
- **Content-Based Filtering** — Builds a genre preference profile from watch history and scores unseen movies by similarity
- **Hybrid Recommendation** — Weighted combination of both methods with adjustable α slider
- **Cold Start** — New users without history select favorite genres to receive content-based recommendations

---

## Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Dataset overview and method summary |
| 🎬 Recommendations | Personalized recommendations for existing or new users |
| 🔍 Movie Explorer | Find similar movies and predict collaborative ratings |
| 📊 Analytics | Rating distribution, genre popularity, model performance |
| ℹ️ About | Methodology, tech stack, and dataset info |

---

## Usage

**Existing user**
1. Go to the **Recommendations** page
2. Enter a User ID (1–610) and click **Get Recommendations**
3. View results across **Collaborative**, **Content-Based**, and **Hybrid** tabs
4. Adjust the **Top N** slider and **α weight** to explore different outputs

**New user (cold start)**
1. Enter a User ID not in the dataset
2. Select favorite genres from the multiselect
3. Receive content-based recommendations based on your genre profile

**Movie Explorer**
1. Search for a movie by title
2. View top similar movies by genre cosine similarity
3. Enter a User ID to get a predicted collaborative rating for that movie

---

## Model Performance

| Configuration | RMSE |
|--------------|------|
| Default (k=40, min_k=1, MSD) | 0.8532 |
| **Tuned (k=45, min_k=9, pearson_baseline)** | **0.8497** |

Improvement: **0.41%** after hyperparameter tuning via `RandomizedSearchCV`.

---

## Dataset

[MovieLens 100K](https://grouplens.org/datasets/movielens/) by GroupLens Research:

- 610 users · 9,742 movies · 100,836 ratings
- Rating scale: 0.5 – 5.0
- 19 genres

---

## Tech Stack

| Library | Usage |
|---------|-------|
| [Streamlit](https://streamlit.io) | Web app framework |
| [scikit-surprise](http://surpriselib.com) | Collaborative filtering model |
| [scikit-learn](https://scikit-learn.org) | Genre matrix, cosine similarity, normalization |
| [pandas](https://pandas.pydata.org) | Data manipulation |
| [Plotly](https://plotly.com) | Interactive charts |

---

## Run Locally

```bash
git clone https://github.com/Wendyauran/movielens-hybrid-recommender.git
cd movielens-hybrid-recommender
pip install -r requirements.txt
streamlit run app.py
```

> Note: The model is trained at startup (~30–60 seconds on first load). No pre-trained file required.

---

## Project Structure

```
├── app.py                    # Entry point, navigation setup
├── pages/
│   ├── 1_overview.py
│   ├── 2_recommendations.py
│   ├── 3_movie_explorer.py
│   ├── 4_analytics.py
│   └── 5_about.py
├── src/
│   ├── data_loader.py        # Data loading and caching
│   ├── model.py              # KNNBaseline training
│   └── recommender.py        # All recommendation logic
├── dataset/
│   ├── movies_collab.csv
│   └── ratings_collab.csv
└── requirements.txt
```

---

## Author

**Wendy Auran Petraz**
