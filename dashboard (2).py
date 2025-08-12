import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gdown

# -----------------
# CONFIG
# -----------------
MODEL_FILE_ID = "16xePMUk_UXm_Bc2HAtMFiCfIUD0i2zkD"  # Your Google Drive file ID
DATA_FILE = "cleaned_movielens.csv"

# -----------------
# Load Dataset
# -----------------
@st.cache_data
def load_metadata():
    df = pd.read_csv(DATA_FILE)
    return df

# -----------------
# Download & Load Model
# -----------------
@st.cache_resource
def download_and_load_model():
    url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
    output = "hybrid_recommender.pkl"
    gdown.download(url, output, quiet=False)

    with open(output, "rb") as f:
        model_data = pickle.load(f)
    return model_data

# -----------------
# Recommendation Function
# -----------------
def recommend_for_user(user_id, model_data, movies_meta, top_n=10):
    hybrid_preds = model_data  # Assuming it's a list of (user, item, score)
    user_preds = [(u, i, s) for (u, i, s) in hybrid_preds if u == user_id]

    # Sort & take top N
    top_preds = sorted(user_preds, key=lambda x: x[2], reverse=True)[:top_n]

    results = []
    for (_, movie_id, score) in top_preds:
        title_row = movies_meta[movies_meta["movie_id"] == movie_id]
        if not title_row.empty:
            title = title_row.iloc[0]["title"]
        else:
            title = f"Movie ID {movie_id}"
        results.append((title, score))
    return results

# -----------------
# Streamlit UI
# -----------------
st.title("🎬 Hybrid Movie Recommender")

movies_meta = load_metadata()
model_data = download_and_load_model()

# User input
user_id = st.number_input("Enter User ID", min_value=1, step=1, value=196)
top_n = st.slider("Number of Recommendations", 5, 20, 10)

if st.button("Get Recommendations"):
    recs = recommend_for_user(user_id, model_data, movies_meta, top_n)
    st.subheader(f"Top {top_n} recommendations for User {user_id}")
    for title, score in recs:
        st.write(f"**{title}** — Predicted Rating: {score:.2f}")
