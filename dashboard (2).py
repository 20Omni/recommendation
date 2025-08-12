# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import gdown
from werkzeug.security import generate_password_hash, check_password_hash

# -------------------------
# CONFIG
# -------------------------
GDRIVE_FILE_ID = "16xePMUk_UXm_Bc2HAtMFiCfIUD0i2zkD"
MODEL_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
MODEL_LOCAL = "hybrid_recommender.pkl"

MOVIES_METADATA_CSV = "cleaned_movielens.csv"   # you uploaded this
USERS_CSV = "users.csv"                         # will be created if missing

# -------------------------
# Utility: load / download model
# -------------------------
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_LOCAL):
        st.info("Downloading hybrid model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_LOCAL, quiet=False)
    # load pickle
    with open(MODEL_LOCAL, "rb") as f:
        data = pickle.load(f)
    return data

# -------------------------
# Users file helpers
# -------------------------
def ensure_users_file():
    if not os.path.exists(USERS_CSV):
        df = pd.DataFrame(columns=["username", "password_hash", "preferred_genres"])
        df.to_csv(USERS_CSV, index=False)

def load_users():
    ensure_users_file()
    return pd.read_csv(USERS_CSV)

def save_user(username, password_hash, preferred_genres_list):
    df = load_users()
    pref_str = ",".join(preferred_genres_list) if preferred_genres_list else ""
    if username in df['username'].values:
        # update
        df.loc[df['username'] == username, ['password_hash', 'preferred_genres']] = [password_hash, pref_str]
    else:
        df = df.append({"username": username, "password_hash": password_hash, "preferred_genres": pref_str}, ignore_index=True)
    df.to_csv(USERS_CSV, index=False)

def get_user_record(username):
    df = load_users()
    row = df[df['username'] == username]
    if row.empty:
        return None
    return row.iloc[0].to_dict()

def update_user_genres(username, genres_list):
    df = load_users()
    if username in df['username'].values:
        df.loc[df['username'] == username, 'preferred_genres'] = ",".join(genres_list)
        df.to_csv(USERS_CSV, index=False)

# -------------------------
# Recommendation boosting
# -------------------------
def boost_recommendations(recs_df, user_pref_genres, movie_metadata_df, boost_factor=0.25):
    """
    recs_df: DataFrame containing at least ['movie_id','score']
    user_pref_genres: list of genres strings
    movie_metadata_df: contains 'movie_id' and 'genres_clean' (as pipe or list)
    boost_factor: relative score boost (fraction of score range) for matches
    """
    if not user_pref_genres:
        return recs_df

    # Ensure metadata genres in list form
    meta = movie_metadata_df.copy()
    if 'genres_clean' in meta.columns:
        # if genres_clean is string like "Action|Comedy" -> list
        meta['genres_list'] = meta['genres_clean'].apply(
            lambda g: [gg.strip() for gg in g.split("|")] if isinstance(g, str) else []
        )
    elif 'genres' in meta.columns:
        meta['genres_list'] = meta['genres'].apply(
            lambda g: [gg.strip() for gg in g.split("|")] if isinstance(g, str) else []
        )
    else:
        meta['genres_list'] = [[]]*len(meta)

    # merge
    merged = recs_df.merge(meta[['movie_id', 'genres_list']], on='movie_id', how='left')

    # compute overlap ratio and boost
    def overlap_boost(genres_list):
        if not isinstance(genres_list, list) or not genres_list:
            return 0.0
        overlap = len(set(genres_list).intersection(set(user_pref_genres)))
        if overlap == 0:
            return 0.0
        # boost proportional to overlap count
        return overlap / max(len(user_pref_genres), 1)

    merged['overlap'] = merged['genres_list'].apply(overlap_boost)

    # Normalize current score range (0..5 expected). We'll add boost_factor * overlap *  (max_score-min_score)
    min_s, max_s = merged['score'].min(), merged['score'].max()
    score_range = max(1e-6, (max_s - min_s))
    merged['score_boosted'] = merged['score'] + (boost_factor * score_range * merged['overlap'])

    # sort and return top
    merged = merged.sort_values(by='score_boosted', ascending=False).reset_index(drop=True)
    return merged

# -------------------------
# App: load model & metadata
# -------------------------
st.set_page_config(layout="wide", page_title="Hybrid Recommender")

st.title("🎬 Hybrid Recommender — Login & Personalized Recommendations")

with st.spinner("Loading model and metadata (this may take a moment)..."):
    model_data = download_and_load_model()   # expects same structure you used earlier
    # The pickle you created earlier (hybrid_recommender.pkl) should include:
    # - "movie_metadata" (DataFrame) OR you can rely on uploaded cleaned_movielens.csv
    # - "recommend_for_user" function OR a dictionary of predictions per-user
    # For safety, we'll handle both shapes below.

# Load movie metadata from uploaded dataset
if os.path.exists(MOVIES_METADATA_CSV):
    movies_df = pd.read_csv(MOVIES_METADATA_CSV)
else:
    st.error(f"Missing {MOVIES_METADATA_CSV}. Upload it to app folder.")
    st.stop()

# Discover recommend function in the pickle
recommend_func = None
movie_metadata_from_model = None
if isinstance(model_data, dict):
    # common earlier structure: { "movie_metadata": df, "recommend_for_user": func, ...}
    recommend_func = model_data.get("recommend_for_user", None)
    movie_metadata_from_model = model_data.get("movie_metadata", None)
elif hasattr(model_data, "recommend_for_user"):
    recommend_func = model_data.recommend_for_user
    # try to extract metadata attribute if available
    movie_metadata_from_model = getattr(model_data, "movie_metadata", None)
else:
    # maybe you saved a mapping user->list of recs
    # we'll accept a dict of user_id -> list[(movie_id,score)] as fallback
    recommend_func = None

# Use metadata from model if present to enrich display
if movie_metadata_from_model is not None:
    # ensure contains 'movie_id','title','genres' or 'genres_clean'
    meta = movie_metadata_from_model.copy()
    if 'title' in meta.columns and 'movie_id' in meta.columns:
        movies_df = meta[['movie_id','title']].drop_duplicates().merge(movies_df.drop_duplicates(subset=['movie_id','title']),
                                                                      on=['movie_id','title'], how='right') \
                    if 'title' in movies_df.columns else meta[['movie_id','title']]
    # keep genres too if available
    if 'genres' in meta.columns and 'genres_clean' not in movies_df.columns:
        movies_df['genres_clean'] = meta['genres']

# -------------------------
# Authentication UI: signup / login
# -------------------------
ensure_users_file()
st.sidebar.title("Account")

mode = st.sidebar.radio("Choose action", ["Login", "Sign up", "Logout", "Update genres"])

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None

if mode == "Sign up":
    st.sidebar.subheader("Create a new account")
    new_user = st.sidebar.text_input("Choose username")
    new_pass = st.sidebar.text_input("Choose password", type="password")
    genres_all = sorted({g.strip() for gs in movies_df['genres_clean'].dropna().unique() for g in str(gs).split("|")})
    pref = st.sidebar.multiselect("Select preferred genres (optional)", genres_all)

    if st.sidebar.button("Create account"):
        if not new_user or not new_pass:
            st.sidebar.error("Username & password required")
        else:
            existing = get_user_record(new_user)
            if existing is not None:
                st.sidebar.error("Username already exists — choose another")
            else:
                hashed = generate_password_hash(new_pass)
                save_user(new_user, hashed, pref)
                st.sidebar.success("Account created — now login")

elif mode == "Login":
    st.sidebar.subheader("Log in")
    username = st.sidebar.text_input("Username", key="login_user")
    password = st.sidebar.text_input("Password", type="password", key="login_pass")
    if st.sidebar.button("Login"):
        rec = get_user_record(username)
        if rec is None:
            st.sidebar.error("User not found — please sign up")
        else:
            if check_password_hash(rec["password_hash"], password):
                st.sidebar.success("Logged in")
                st.session_state.logged_in = True
                st.session_state.username = username
                # load preferred genres into session
                pref = rec["preferred_genres"]
                st.session_state.pref_genres = pref.split(",") if isinstance(pref, str) and pref else []
            else:
                st.sidebar.error("Invalid username or password")

elif mode == "Logout":
    if st.sidebar.button("Confirm logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.pref_genres = []
        st.sidebar.success("Logged out")

elif mode == "Update genres":
    if not st.session_state.logged_in:
        st.sidebar.info("Login first to update genres")
    else:
        st.sidebar.subheader("Update your favorite genres")
        genres_all = sorted({g.strip() for gs in movies_df['genres_clean'].dropna().unique() for g in str(gs).split("|")})
        current_pref = st.session_state.get("pref_genres", [])
        new_pref = st.sidebar.multiselect("Pick genres", genres_all, default=current_pref)
        if st.sidebar.button("Save genres"):
            update_user_genres(st.session_state.username, new_pref)
            st.session_state.pref_genres = new_pref
            st.sidebar.success("Preferences updated")

# -------------------------
# Main app area
# -------------------------
st.markdown("---")
if not st.session_state.logged_in:
    st.header("Welcome!")
    st.write("Please sign up or login from the left sidebar to get personalized recommendations.")
else:
    st.header(f"Hello, {st.session_state.username} 👋")
    st.write("Your saved favorite genres:", ", ".join(st.session_state.get("pref_genres", []) or ["(none)"]))

    # user_id mapping — allow user to use their internal user_id (numeric)
    # Provide a number input defaulting to a sample user ID
    user_id_in = st.number_input("Enter numeric user_id (to pull historical interactions)", min_value=1, step=1, value=196)

    top_n = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)

    if st.button("Get personalized recommendations"):
        # 1) obtain base recommendations from hybrid model
        # If pickle provided recommend_for_user function:
        try:
            if recommend_func is not None:
                # earlier wrappers had signature recommend_for_user_func(user_id, movie_metadata, top_n)
                try:
                    recs = recommend_func(int(user_id_in), movies_df, top_n=500)  # get many then filter/boost
                    # recs expected as list of tuples (movie_id, title, score) or (movie_id, score,...)
                except TypeError:
                    # alternate signature
                    recs = recommend_func(int(user_id_in), top_n=500)
            else:
                # fallback: model_data might be dict {user_id: [(mid,score),...]}
                if int(user_id_in) in model_data:
                    recs = model_data[int(user_id_in)]
                else:
                    st.error("No recommendations found for this user in the saved model.")
                    recs = []
        except Exception as e:
            st.error(f"Failed to get base recommendations from model: {e}")
            recs = []

        # Normalize recs into DataFrame with movie_id, title, score
        # Handle multiple possible tuple formats
        rec_rows = []
        for it in recs:
            if isinstance(it, (list, tuple)):
                if len(it) >= 3:
                    movie_id = int(it[0])
                    title = str(it[1])
                    score = float(it[2])
                elif len(it) == 2:
                    movie_id = int(it[0])
                    score = float(it[1])
                    # title lookup
                    title = movies_df.loc[movies_df['movie_id'] == movie_id, 'title'].squeeze() if 'title' in movies_df.columns else str(movie_id)
                else:
                    continue
                rec_rows.append({"movie_id": movie_id, "title": title, "score": score})
        recs_df = pd.DataFrame(rec_rows)

        if recs_df.empty:
            st.info("No recommendations returned by the model.")
        else:
            # 2) boost by user preferred genres
            user_pref = st.session_state.get("pref_genres", [])
            boosted = boost_recommendations(recs_df, user_pref, movies_df, boost_factor=0.25)
            top_boosted = boosted.head(top_n)

            # Display
            st.subheader("Top recommendations (boosted by your genres)")
            for _, row in top_boosted.iterrows():
                genres = row.get("genres_list") or []
                st.markdown(f"**{row['title']}** — ⭐ {row['score_boosted']:.2f}  \n🎭 Genres: {', '.join(genres)}")
