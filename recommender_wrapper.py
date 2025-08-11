
import pickle
import pandas as pd

# ===== Load Pickle =====
with open("hybrid_recommender.pkl", "rb") as f:
    recommender_data = pickle.load(f)

movie_metadata = recommender_data["movie_metadata"]
recommend_for_user_func = recommender_data["recommend_for_user"]

# ===== Ensure genres_clean exists =====
if "genres_clean" not in movie_metadata.columns:
    movie_metadata["genres_clean"] = movie_metadata["genres"].apply(
        lambda g: [genre.strip() for genre in g.split("|")] if isinstance(g, str) else []
    )

# ===== Wrapper Function =====
def recommend_movies_for_user(user_id, top_n=10):
    """
    Recommend top N movies for a given user ID.
    """
    recommendations = recommend_for_user_func(user_id, movie_metadata, top_n=top_n)
    rec_df = pd.DataFrame(recommendations, columns=["movie_id", "title", "score"])
    
    # Merge genres info
    rec_df = rec_df.merge(
        movie_metadata[["movie_id", "genres_clean"]],
        on="movie_id",
        how="left"
    )
    
    return rec_df
