from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import load_movies


class ContentBasedRecommender:
    def __init__(self):
        self.df = load_movies()
        self._prepare()

    def _prepare(self):
        self.df["combined_features"] = (
            self.df["genre"] + " " + self.df["description"]
        )

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df["combined_features"]
        )

        self.similarity_matrix = cosine_similarity(
            self.tfidf_matrix, self.tfidf_matrix
        )

    def recommend(self, title, top_n=3):
        if title not in self.df["title"].values:
            return []

        idx = self.df[self.df["title"] == title].index[0]
        scores = list(enumerate(self.similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        return [
            self.df.iloc[i[0]]["title"]
            for i in scores[1 : top_n + 1]
        ]


if __name__ == "__main__":
    model = ContentBasedRecommender()
    print(model.recommend("Inception"))
