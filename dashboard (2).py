import re
import streamlit as st
import pandas as pd
import pickle
import sqlite3
from collections import Counter

# ---------------- Config ----------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

# ---------------- Paths ----------------
HYBRID_MODEL_PATH = "hybrid_recommender.pkl"
MOVIE_METADATA_PATH = "movie_metadata.csv"
DB_PATH = "users.db"
TOP_N = 12  # show more to fill grid nicely

# ---------------- Load Data ----------------
with open(HYBRID_MODEL_PATH, "rb") as f:
    hybrid_data = pickle.load(f)

# {user_id (int): [movie titles]}
final_recs = hybrid_data["final_recs"]

# movies_df columns expected: title, genres_clean, avg_rating
movies_df = pd.read_csv(MOVIE_METADATA_PATH)

# Safety: ensure required columns exist
required_cols = {"title", "genres_clean", "avg_rating"}
missing = required_cols - set(movies_df.columns)
if missing:
    st.error(f"movie_metadata.csv is missing columns: {missing}")
    st.stop()

# ---------------- Database Setup ----------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
""")
c.execute("""
CREATE TABLE IF NOT EXISTS watched (
    username TEXT,
    movie_title TEXT,
    FOREIGN KEY(username) REFERENCES users(username)
)
""")
conn.commit()

# ---------------- Helpers ----------------
def slug_key(text: str) -> str:
    """Make a safe, short key fragment from any text."""
    return re.sub(r'[^A-Za-z0-9]+', '_', str(text))[:40]

def signup(username, password):
    if not username or not password:
        st.error("Please enter both username and password.")
        return
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        st.success("Signup successful! Please log in.")
    except sqlite3.IntegrityError:
        st.error("Username already exists.")

def login(username, password):
    if not username or not password:
        return False
    c.execute("SELECT 1 FROM users WHERE username=? AND password=?", (username, password))
    return c.fetchone() is not None

def mark_watched(username, movie_title):
    # Avoid duplicate rows
    c.execute("SELECT 1 FROM watched WHERE username=? AND movie_title=?", (username, movie_title))
    if c.fetchone() is None:
        c.execute("INSERT INTO watched (username, movie_title) VALUES (?, ?)", (username, movie_title))
        conn.commit()

def get_watched(username):
    c.execute("SELECT movie_title FROM watched WHERE username=?", (username,))
    return [x[0] for x in c.fetchall()]

def get_top_genres(username, k=4):
    watched = get_watched(username)
    if not watched:
        return []
    watched_genres = movies_df[movies_df['title'].isin(watched)]['genres_clean'].str.split('|').explode()
    # Filter out NAs
    watched_genres = watched_genres.dropna()
    top = [g for g, _ in Counter(watched_genres).most_common(k)]
    return top

def get_genre_recommendations(username, top_n=TOP_N):
    """
    Fallback recs for new users: top-rated among user's most-seen genres.
    If user hasn't watched anything, return global top-rated.
    """
    watched = set(get_watched(username))
    if not watched:
        return movies_df.sort_values(by="avg_rating", ascending=False).head(top_n)["title"].tolist()

    top_genres = get_top_genres(username, k=4)

    # Recommend high-rated movies that match any of top genres and not watched
    cand = movies_df[~movies_df['title'].isin(watched)].copy()
    cand = cand[cand['genres_clean'].fillna("").apply(lambda g: any(gen in g for gen in top_genres))]
    cand = cand.sort_values(by="avg_rating", ascending=False).head(top_n)
    return cand["title"].tolist()

def order_df_by_titles(df: pd.DataFrame, ordered_titles: list) -> pd.DataFrame:
    """Preserve given title order (e.g., from ML) when subsetting DataFrame."""
    order_map = {t: i for i, t in enumerate(ordered_titles)}
    df = df[df["title"].isin(ordered_titles)].copy()
    df["__order"] = df["title"].map(order_map)
    return df.sort_values("__order").drop(columns="__order")

def render_cards_by_titles(titles: list, username: str, prefix: str, cols_count: int = 3):
    """Render a 3-column card grid for given movie titles."""
    if not titles:
        st.info("No items to show yet.")
        return

    df = order_df_by_titles(movies_df, titles)
    watched_set = set(get_watched(username))
    cols = st.columns(cols_count)

    for i, row in enumerate(df.itertuples(index=False)):
        col = cols[i % cols_count]
        title = row.title
        genres = row.genres_clean
        rating = row.avg_rating
        is_watched = title in watched_set

        with col:
            st.markdown(
                f"""
                <div class="card">
                  <div class="card-title">{title}{' <span class="badge watched">Watched</span>' if is_watched else ''}</div>
                  <div class="card-genres">{genres}</div>
                  <div class="card-rating">⭐ {rating:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            btn_key = f"{prefix}_btn_{slug_key(title)}_{i}"
            clicked = st.button("Watched ✅", key=btn_key, disabled=is_watched)
            if clicked and not is_watched:
                mark_watched(username, title)
                st.toast(f"Marked '{title}' as watched!")
                st.rerun()

# ---------------- Tiny CSS for nice cards ----------------
st.markdown(
    """
    <style>
    .card {
        border: 1px solid #eee;
        border-radius: 14px;
        padding: 12px 14px;
        margin-bottom: 14px;
        background: #ffffff;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
        min-height: 110px;
    }
    .card-title {
        font-weight: 700;
        font-size: 1rem;
        line-height: 1.2;
        margin-bottom: 6px;
    }
    .card-genres {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 8px;
    }
    .card-rating {
        font-size: 0.95rem;
        color: #333;
    }
    .badge {
        display: inline-block;
        margin-left: 8px;
        padding: 2px 8px;
        border-radius: 999px;
        background: #eef2ff;
        color: #3730a3;
        font-size: 0.75rem;
        vertical-align: middle;
    }
    .badge.watched {
        background: #d1fae5;
        color: #065f46;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Session State ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

st.title("🎬 Movie Recommender System")

# ---------------- Sidebar: Login / Signup / Logout ----------------
if st.session_state.logged_in:
    st.sidebar.markdown(f"**👤 {st.session_state.username}**")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()
else:
    choice = st.sidebar.radio("Menu", ["Login", "Signup"])
    if choice == "Signup":
        st.subheader("Create New Account")
        new_user = st.text_input("Username", key="signup_user")
        new_pass = st.text_input("Password", type="password", key="signup_pass")
        if st.button("Sign Up"):
            signup(new_user, new_pass)

    if choice == "Login":
        st.subheader("Login")
        username_in = st.text_input("Username", key="login_user")
        password_in = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if login(username_in, password_in):
                st.session_state.logged_in = True
                st.session_state.username = username_in
                st.success(f"Welcome {username_in}!")
                st.rerun()
            else:
                st.error("Invalid username or password")

# ---------------- Main App (when logged in) ----------------
if st.session_state.logged_in:
    tabs = st.tabs(["🌟 Top Rated", "🎯 Recommendations", "📖 Watched History"])

    # ---------- Top Rated ----------
    with tabs[0]:
        st.subheader("Top Rated")
        top_titles = (
            movies_df.sort_values(by="avg_rating", ascending=False)
            .head(TOP_N)["title"]
            .tolist()
        )
        render_cards_by_titles(top_titles, st.session_state.username, prefix="top", cols_count=3)

    # ---------- Recommendations ----------
    with tabs[1]:
        st.subheader("For You")
        # Try ML first (if username is numeric & present in model)
        username = st.session_state.username
        try:
            user_id = int(username)
        except ValueError:
            user_id = None

        if user_id is not None and user_id in final_recs:
            rec_titles = final_recs[user_id][:TOP_N]
        else:
            rec_titles = get_genre_recommendations(username, top_n=TOP_N)

        render_cards_by_titles(rec_titles, username, prefix="rec", cols_count=3)

    # ---------- Watched History ----------
    with tabs[2]:
        st.subheader("Your Watched History")
        watched_list = get_watched(st.session_state.username)
        if not watched_list:
            st.info("You haven't watched anything yet.")
        else:
            # Preserve watched order by when it was added (approx by current order from DB)
            render_cards_by_titles(watched_list, st.session_state.username, prefix="hist", cols_count=3)
