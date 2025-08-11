import streamlit as st
import os
import pickle
import pandas as pd
import gdown

file_id = "16xePMUk_UXm_Bc2HAtMFiCfIUD0i2zkD"
pickle_path = "hybrid_recommender.pkl"
url = f"https://drive.google.com/uc?id={file_id}"

def download_pickle():
    st.info("Downloading model from Google Drive, please wait...")
    # Use gdown with confirm option enabled internally
    gdown.download(url, pickle_path, quiet=False)

def is_valid_pickle(path):
    try:
        with open(path, "rb") as f:
            pickle.load(f)
        return True
    except Exception:
        return False

if not os.path.exists(pickle_path) or not is_valid_pickle(pickle_path):
    download_pickle()

if not is_valid_pickle(pickle_path):
    st.error("Failed to download a valid model file. Please try again later.")
    st.stop()

# Load the pickle file safely now
with open(pickle_path, "rb") as f:
    data = pickle.load(f)

recommend_for_user_func = data["recommend_for_user"]
movie_metadata = data["movie_metadata"]

# Your existing Streamlit UI code follows...


# Load the pickle file
with open(pickle_path, "rb") as f:
    data = pickle.load(f)

recommend_for_user_func = data["recommend_for_user"]
movie_metadata = data["movie_metadata"]

# Make sure genres_clean exists
if "genres_clean" not in movie_metadata.columns:
    movie_metadata["genres_clean"] = movie_metadata["genres"].apply(
        lambda g: [genre.strip() for genre in g.split("|")] if isinstance(g, str) else []
    )

# Streamlit UI
st.title("🎬 Movie Recommender System")

user_id = st.number_input("Enter User ID", min_value=1, value=1)
top_n = st.slider("Number of recommendations", 1, 20, 10)

all_genres = sorted(set(g for genres in movie_metadata["genres_clean"] for g in genres))
selected_genre = st.selectbox("Filter by Genre (optional)", ["All"] + all_genres)

def recommend_movies_for_user(user_id, top_n=10):
    recommendations = recommend_for_user_func(user_id, movie_metadata, top_n=top_n)
    rec_df = pd.DataFrame(recommendations, columns=["movie_id", "title", "score"])
    genres_map = dict(zip(movie_metadata["movie_id"], movie_metadata["genres_clean"]))
    rec_df["genres_clean"] = rec_df["movie_id"].map(genres_map)
    return rec_df

if st.button("Get Recommendations"):
    rec_df = recommend_movies_for_user(user_id, top_n)
    if selected_genre != "All":
        rec_df = rec_df[rec_df["genres_clean"].apply(lambda g_list: selected_genre in g_list)]
    if not rec_df.empty:
        st.dataframe(rec_df, use_container_width=True)
    else:
        st.warning("No recommendations match the selected genre.")
