def hybrid_weighted(title, movies_df, cosine_sim, indices, item_similarity_df,
                    n=30, weight_content=0.5, weight_cf=0.5):

    # ---------------- CONTENT ----------------
    content_scores = {}

    if title in indices:
        idx = indices[title]

        sim_scores = list(enumerate(cosine_sim[idx].flatten()))

        
        sim_scores = [i for i in sim_scores if i[0] < len(movies_df)]

        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # limit top N
        sim_scores = sim_scores[1:n+1]

        content_scores = {
            movies_df['title'].iloc[i[0]]: i[1]
            for i in sim_scores
        }

  
    cf_scores = {}

    movie_id_list = movies_df[movies_df['title'] == title]['movieId'].values

    if len(movie_id_list) > 0:
        movie_id = movie_id_list[0]

        if movie_id in item_similarity_df.columns:

            cf_series = item_similarity_df[movie_id].sort_values(ascending=False)
            cf_series = cf_series.drop(movie_id, errors='ignore')

            #  limit CF top N
            cf_series = cf_series.head(n)

            # FAST mapping (no loop filtering)
            movieId_to_title = dict(zip(movies_df['movieId'], movies_df['title']))

            cf_scores = {
                movieId_to_title.get(mid): score
                for mid, score in cf_series.items()
                if mid in movieId_to_title
            }

    # ---------------- NORMALIZATION ----------------
    #  normalize scores
    def normalize(scores):
        if not scores:
            return {}
        max_score = max(scores.values())
        if max_score == 0:
            return scores
        return {k: v / max_score for k, v in scores.items()}

    content_scores = normalize(content_scores)
    cf_scores = normalize(cf_scores)

    # ---------------- COMBINE ----------------
    combined_scores = {}
    all_movies = set(content_scores.keys()).union(cf_scores.keys())

    for movie in all_movies:
        c_score = content_scores.get(movie, 0)
        cf_score = cf_scores.get(movie, 0)

        combined_scores[movie] = (
            weight_content * c_score +
            weight_cf * cf_score
        )

    # ---------------- FINAL ----------------
    final_recommendations = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [movie for movie, _ in final_recommendations[:n]]
