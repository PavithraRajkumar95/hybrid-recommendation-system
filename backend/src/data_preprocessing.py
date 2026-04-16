import pandas as pd
import os

def load_and_preprocess_movies():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    movies_path = os.path.join(BASE_DIR, "..", "data", "movie.csv")
    
    movies_df = pd.read_csv(movies_path)
    
    # Clean genres
    movies_df['genres'] = movies_df['genres'].str.replace('|', ' ', regex=False)
    
    return movies_df


def load_ratings():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ratings_path = os.path.join(BASE_DIR, "..", "data", "rating.csv")
    
    ratings_df = pd.read_csv(ratings_path)
    return ratings_df