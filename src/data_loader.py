import pandas as pd

def load_movies(path="data/movies.csv"):
    return pd.read_csv(path)

