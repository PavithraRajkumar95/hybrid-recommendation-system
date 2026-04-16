from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def build_tfidf_matrix(movies_df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genres'])
    return tfidf_matrix

def build_cosine_sim(tfidf_matrix):
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

def build_indices(movies_df):
    return pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()

def recommend_content(title, movies_df, cosine_sim, indices, n=10):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices].tolist()