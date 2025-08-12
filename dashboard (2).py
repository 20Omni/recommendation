import streamlit as st
import pandas as pd
import pickle
import os
import gdown

# ---------------------
# Download model from Google Drive if not present
# ---------------------
MODEL_URL = "https://drive.google.com/uc?id=16xePMUk_UXm_Bc2HAtMFiCfIUD0i2zkD"
MODEL_FILE = "hybrid_recommender.pkl"

if not os.path.exists(MODEL_FILE):
    st.info("Downloading hybrid recommender model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_FILE, quiet=False)

# ---------------------
# Load the hybrid model
# ---------------------
with open(MODEL_FILE, "rb") as f:
    hybrid_model = pickle.load(f)

# ---------------------
# Load your movies metadata
# ---------------------
movies_meta = pd.read_csv("cleaned_movielens.csv")  # Update to your actual metadata file path

# ---------------------
# Recommendation function
# ---------------------
def recommend_for_user(uid, top_n=10):
    # Assume `hybrid_model` has the same recommend_for_user method from your code
    recs = hybrid_model.recommend_for_user(uid, top_n=top_n)
    return recs

# ---------------------
# Streamlit UI
# ---------------------
st.title("🎬 Hybrid Movie Recommendation System")

user_id = st.number_input("Enter User ID", min_value=1, step=1)
top_n = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10)

if st.button("Get Recommendations"):
    try:
        recs = recommend_for_user(user_id, top_n=top_n)
        st.subheader(f"Top {top_n} Recommendations for User {user_id}")
        for movie_id, title, score in recs:
            st.write(f"**{title}** (ID: {movie_id}) — Predicted Rating: {score:.2f}")
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
