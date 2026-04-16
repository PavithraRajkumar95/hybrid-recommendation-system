import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def build_user_item_matrix(df):
    return df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

def build_item_similarity(user_item_matrix):
    item_similarity = cosine_similarity(user_item_matrix.T)
    return pd.DataFrame(
        item_similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )

def recommend_cf(movie_id, item_similarity_df, n=10):
    if movie_id not in item_similarity_df.columns:
        return []
    similar_scores = item_similarity_df[movie_id].sort_values(ascending=False)
    return similar_scores.index[1:n+1].tolist()