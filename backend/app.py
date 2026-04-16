from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# ---------------- GLOBAL VARIABLES ----------------
movies_df = None
cosine_sim = None
indices = None
item_similarity_df = None
valid_users = None
ratings = None
top_rated_movies = None


# ---------------- LOAD MODELS ----------------
@app.on_event("startup")
def load_all_models():
    global movies_df, cosine_sim, indices, item_similarity_df
    global valid_users, ratings, top_rated_movies

    print("🚀 Loading artifacts...")

    # -------- LOAD ARTIFACTS --------
    movies_df = joblib.load("artifacts/movies.pkl")
    cosine_sim = joblib.load("artifacts/content_model.pkl")
    indices = joblib.load("artifacts/indices.pkl")
    item_similarity_df = joblib.load("artifacts/collaborative_model.pkl")
    valid_users = joblib.load("artifacts/valid_users.pkl")
    ratings = joblib.load("artifacts/ratings.pkl")

    # ---------------- TOP RATED (CORRECT VERSION) ----------------
    movie_stats = ratings.groupby("movieId")["rating"].mean()

    # top 50 highest rated movies
    top_movie_ids = movie_stats.sort_values(ascending=False).head(50).index

    top_rated_movies = movies_df[
        movies_df["movieId"].isin(top_movie_ids)
    ]["title"].tolist()

    print("🎉 All models loaded successfully!")


# ---------------- CONTENT BASED ----------------
def content_recommend(title, top_n=10):
    if title not in indices:
        return []

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    movie_indices = [i[0] for i in sim_scores]

    return movies_df.iloc[movie_indices]["title"].tolist()


# ---------------- COLLAB FILTERING ----------------
def collab_recommend(movie_id, top_n=10):
    if movie_id not in item_similarity_df.index:
        return []

    sim_scores = item_similarity_df[movie_id].sort_values(ascending=False)[1:top_n+1]

    return sim_scores.index.tolist()


# ---------------- HYBRID RECOMMENDATION ----------------
def hybrid_recommend(user_id, top_n=10):

    # ---------------- COLD START ----------------
    if valid_users is None or user_id not in valid_users:
        return {
            "type": "cold_start",
            "recommendations": top_rated_movies[:top_n]
        }

    # ---------------- USER HISTORY (FIXED LOGIC) ----------------
    user_history = ratings[
        ratings["userId"] == user_id
    ].sort_values("rating", ascending=False)

    liked_movies = user_history["movieId"].head(5).tolist()

    scores = {}

    for movie_id in liked_movies:

        movie_row = movies_df[movies_df["movieId"] == movie_id]

        if movie_row.empty:
            continue

        title = movie_row["title"].values[0]

        if title not in indices:
            continue

        recs = content_recommend(title, top_n=10)

        for rank, rec in enumerate(recs):
            scores[rec] = scores.get(rec, 0) + (10 - rank)

    sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    final_recs = [r[0] for r in sorted_recs[:top_n]]

    # ---------------- FINAL FALLBACK ----------------
    if not final_recs:
        final_recs = top_rated_movies[:top_n]

    return {
        "type": "hybrid",
        "recommendations": final_recs
    }


# ---------------- API ENDPOINT ----------------
@app.get("/recommend/{user_id}")
def recommend(user_id: int):
    return hybrid_recommend(user_id)


# ---------------- DEBUG USERS ----------------
@app.get("/debug/users")
def get_users():
    return {
        "users": [int(u) for u in valid_users[:50]] if valid_users is not None else []
    }


# ---------------- TOP RATED ----------------
@app.get("/top-rated")
def top_rated():
    return {
        "type": "top_rated",
        "recommendations": top_rated_movies[:20]
    }


# ---------------- HEALTH CHECK ----------------
@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Movie Recommender API is live 🚀"
    }