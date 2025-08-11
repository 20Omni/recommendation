
import streamlit as st
import pickle
import pandas as pd

# ===== Load saved recommender =====
with open("hybrid_recommender.pkl", "rb") as f:
    data = pickle.load(f)

recommend_for_user_func = data["recommend_for_user"]
movie_metadata = data["movie_metadata"]

# ===== Streamlit UI =====
st.title("🎬 Movie Recommender System")

# Sidebar inputs
user_id = st.number_input("Enter User ID", min_value=1, value=1)
top_n = st.slider("Number of recommendations", 1, 20, 10)

# Optional genre filter
all_genres = sorted(set(g for genres in movie_metadata["genres_clean"] for g in genres))
selected_genre = st.selectbox("Filter by Genre (optional)", ["All"] + all_genres)

# Recommend function wrapper
def recommend_movies_for_user(user_id, top_n=10):
    recommendations = recommend_for_user_func(user_id, movie_metadata, top_n=top_n)
    rec_df = pd.DataFrame(recommendations, columns=["movie_id", "title", "score"])
    
    # Merge genres if missing
    if "genres_clean" not in rec_df.columns:
        genres_map = dict(zip(movie_metadata["movie_id"], movie_metadata["genres_clean"]))
        rec_df["genres_clean"] = rec_df["movie_id"].map(genres_map)
    
    return rec_df

# Generate recommendations
if st.button("Get Recommendations"):
    rec_df = recommend_movies_for_user(user_id, top_n)

    # Apply genre filter
    if selected_genre != "All":
        rec_df = rec_df[rec_df["genres_clean"].apply(lambda g_list: selected_genre in g_list)]

    if not rec_df.empty:
        st.dataframe(rec_df, use_container_width=True)
    else:
        st.warning("No recommendations match the selected genre.")

