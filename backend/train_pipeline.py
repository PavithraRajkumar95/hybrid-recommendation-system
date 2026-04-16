import os
import pandas as pd
import numpy as np
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data")
ARTIFACT_PATH = os.path.join(BASE_DIR, "artifacts")

os.makedirs(ARTIFACT_PATH, exist_ok=True)


# ---------------- LOAD DATA ----------------
print("🚀 Loading data...")

movies = pd.read_csv(os.path.join(DATA_PATH, "movie.csv"))
ratings = pd.read_csv(
    os.path.join(DATA_PATH, "rating.csv"),
    usecols=["userId", "movieId", "rating"],
    engine="python"
)

print("✔ Data loaded")


# ---------------- CLEANING + REDUCTION ----------------
print("⚡ Cleaning + reducing dataset...")

movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

# Keep only active movies + users
top_movies = ratings["movieId"].value_counts().head(300).index
ratings = ratings[ratings["movieId"].isin(top_movies)]

top_users = ratings["userId"].value_counts().head(800).index
ratings = ratings[ratings["userId"].isin(top_users)]

movies = movies[movies["movieId"].isin(ratings["movieId"])].reset_index(drop=True)

print(f"✔ Movies: {len(movies)}, Ratings: {len(ratings)}")


# ---------------- CONTENT-BASED MODEL ----------------
print("⚡ Building TF-IDF model...")

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])

# IMPORTANT: cosine similarity is still heavy but now reduced dataset
cosine_sim = cosine_similarity(tfidf_matrix)

print("✔ Content model ready")


# ---------------- INDEX MAP ----------------
indices = pd.Series(
    movies.index,
    index=movies["title"]
).drop_duplicates()

print("✔ Indices created")


# ---------------- COLLAB FILTERING ----------------
print("⚡ Building collaborative filtering model...")

user_item_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0).astype(np.float32)   # 🔥 memory optimization

item_similarity = cosine_similarity(user_item_matrix.T)

item_similarity_df = pd.DataFrame(
    item_similarity,
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

print("✔ Collaborative model ready")


# ---------------- VALID USERS ----------------
valid_users = ratings["userId"].unique()

print("✔ Valid users extracted")


# ---------------- SAVE ARTIFACTS (OPTIMIZED) ----------------
print("💾 Saving artifacts...")

joblib.dump(movies, os.path.join(ARTIFACT_PATH, "movies.pkl"))
joblib.dump(tfidf, os.path.join(ARTIFACT_PATH, "tfidf.pkl"))
joblib.dump(cosine_sim.astype(np.float16), os.path.join(ARTIFACT_PATH, "content_model.pkl"))
joblib.dump(indices, os.path.join(ARTIFACT_PATH, "indices.pkl"))
joblib.dump(item_similarity_df.astype(np.float16), os.path.join(ARTIFACT_PATH, "collaborative_model.pkl"))
joblib.dump(valid_users, os.path.join(ARTIFACT_PATH, "valid_users.pkl"))

print("🎉 TRAINING COMPLETE - ALL ARTIFACTS SAVED")