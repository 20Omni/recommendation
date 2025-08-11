import streamlit as st
import pickle
import pandas as pd
import os
import gdown

# ===== Download pickle from Google Drive if missing =====
file_id = "16xePMUk_UXm_Bc2HAtMFiCfIUD0i2zkD"
pickle_path = "hybrid_recommender.pkl"

if not os.path.exists(pickle_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    st.info("Downloading recommender model from Google Drive...")
    gdown.download(url, pickle_path, quiet=False)

# ===== Load saved recommender =====
with open(pickle_path, "rb") as f:
    data = pickle.load(f)

recommend_for_user_func = data["recommend_for_user"]
movie_metadata = data["movie_metadata"]

# Ensure genres_clean exists
if "genres_clean" not in movie_metadata.columns:
    movie_metadata["genres_clean"] = movie_metadata["genres"].apply(
        lambda g: [genre.strip() for genre in g.split("|")] if isinstance(g, str) else []
    )

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
    
    # Merge genres info
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
