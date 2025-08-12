import streamlit as st
import pandas as pd
import pickle
import gdown

# --------------------------
# CONFIG
# --------------------------
GDRIVE_URL = "https://drive.google.com/uc?id=16xePMUk_UXm_Bc2HAtMFiCfIUD0i2zkD"
MODEL_FILE = "hybrid_recommender.pkl"
MOVIE_METADATA_FILE = "cleaned_movielens.csv"

# --------------------------
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# --------------------------
@st.cache_resource
def load_model():
    gdown.download(GDRIVE_URL, MODEL_FILE, quiet=False)
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    return model

# --------------------------
# LOAD MOVIE METADATA
# --------------------------
@st.cache_data
def load_metadata():
    df = pd.read_csv(MOVIE_METADATA_FILE)
    return df

# --------------------------
# RECOMMENDATION FUNCTION
# --------------------------
def recommend_movies(user_id, model, movies_df, top_n=10):
    if user_id not in model:
        st.warning(f"User {user_id} not found in model!")
        return []
    
    recs = model[user_id]
    recs_df = pd.DataFrame(recs, columns=["movie_id", "score"])
    merged = recs_df.merge(movies_df, on="movie_id", how="left")
    return merged.sort_values(by="score", ascending=False).head(top_n)

# --------------------------
# LOGIN SYSTEM
# --------------------------
users_db = {"user1": "pass1", "user2": "pass2", "admin": "admin123"}

def login():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in users_db and users_db[username] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success(f"Welcome, {username}!")
        else:
            st.error("Invalid username or password")

# --------------------------
# MAIN APP
# --------------------------
st.title("🎬 Movie Recommendation Dashboard")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
else:
    st.sidebar.success(f"Logged in as {st.session_state['username']}")
    model = load_model()
    movies_df = load_metadata()

    st.subheader("Get Recommendations")
    user_id = st.number_input("Enter User ID", min_value=1, step=1)

    if st.button("Recommend"):
        recs = recommend_movies(user_id, model, movies_df, top_n=10)
        if len(recs) > 0:
            st.table(recs[["movie_id", "title", "score"]])
