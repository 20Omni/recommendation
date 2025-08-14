import streamlit as st
import pandas as pd
import pickle
import sqlite3
import hashlib
from collections import Counter
from typing import List

# ==========================
# Paths / Constants
# ==========================
HYBRID_MODEL_PATH = "hybrid_recommender.pkl"   # expects keys: final_recs (dict), weights (optional)
MOVIE_METADATA_PATH = "movie_metadata.csv"     # columns: title, genres_clean, avg_rating
DB_PATH = "users.db"
TOP_N = 10

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")


# ==========================
# Data Loaders (cached)
# ==========================
@st.cache_data(show_spinner=False)
def load_hybrid(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    # Expected: {"final_recs": dict, "weights": ...}
    final_recs = data.get("final_recs", {})
    weights = data.get("weights", {})
    return final_recs, weights

@st.cache_data(show_spinner=False)
def load_movies(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"title", "genres_clean", "avg_rating"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"`movie_metadata.csv` is missing columns: {missing}.")
    # Keep only needed columns (and drop duplicates just in case)
    df = df[list(required & set(df.columns))].drop_duplicates(subset=["title"]).copy()
    # Ensure types
    df["title"] = df["title"].astype(str)
    df["genres_clean"] = df["genres_clean"].astype(str)
    df["avg_rating"] = pd.to_numeric(df["avg_rating"], errors="coerce").fillna(0.0)
    return df

final_recs, weights = load_hybrid(HYBRID_MODEL_PATH)
movies_df = load_movies(MOVIE_METADATA_PATH)


# ==========================
# Database Setup (SQLite)
# ==========================
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
""")

# Enforce uniqueness so the same movie isn't added twice for a user
c.execute("""
CREATE TABLE IF NOT EXISTS watched (
    username TEXT,
    movie_title TEXT,
    UNIQUE(username, movie_title),
    FOREIGN KEY(username) REFERENCES users(username)
)
""")
conn.commit()


# ==========================
# Helpers (Auth, Watched)
# ==========================
def signup(username: str, password: str):
    if not username or not password:
        st.error("Username and password are required.")
        return
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        st.success("Signup successful! Please log in.")
    except sqlite3.IntegrityError:
        st.error("Username already exists.")

def login(username: str, password: str) -> bool:
    c.execute("SELECT 1 FROM users WHERE username=? AND password=?", (username, password))
    return c.fetchone() is not None

def mark_watched(username: str, movie_title: str):
    try:
        c.execute("INSERT OR IGNORE INTO watched (username, movie_title) VALUES (?, ?)", (username, movie_title))
        conn.commit()
    except Exception as e:
        st.error(f"Failed to mark watched: {e}")

def unwatch_movie(username: str, movie_title: str):
    try:
        c.execute("DELETE FROM watched WHERE username=? AND movie_title=?", (username, movie_title))
        conn.commit()
    except Exception as e:
        st.error(f"Failed to unwatch: {e}")

def get_watched(username: str) -> List[str]:
    c.execute("SELECT movie_title FROM watched WHERE username=?", (username,))
    return [row[0] for row in c.fetchall()]


# ==========================
# Recommendation Logic
# ==========================
def split_genres(genres_str: str) -> List[str]:
    """Supports 'Action|Drama' or 'Action, Drama' formats."""
    if pd.isna(genres_str):
        return []
    s = str(genres_str).replace("|", ",")
    return [g.strip() for g in s.split(",") if g.strip()]

def rating_to_stars(rating: float) -> str:
    """Assumes rating on ~1–5 scale; rounds to nearest star."""
    full = int(round(float(rating)))
    full = max(0, min(5, full))
    return "⭐" * full + "☆" * (5 - full)

def get_genre_top_rated(genres: List[str], exclude_titles: set, top_n: int) -> List[str]:
    if not genres:
        # Fallback to overall top rated
        return (
            movies_df[~movies_df["title"].isin(exclude_titles)]
            .sort_values("avg_rating", ascending=False)
            .head(top_n)["title"]
            .tolist()
        )
    mask = movies_df["genres_clean"].apply(lambda g: any(x in g for x in genres))
    return (
        movies_df[mask & ~movies_df["title"].isin(exclude_titles)]
        .sort_values("avg_rating", ascending=False)
        .head(top_n)["title"]
        .tolist()
    )

def get_combined_recommendations(username: str, top_n: int = TOP_N) -> List[str]:
    watched_list = set(get_watched(username))
    # 1) Hybrid model recs if available
    hybrid = final_recs.get(username, []) if isinstance(final_recs, dict) else []
    # 2) Genre preferences from watched
    watched_genres_series = movies_df[movies_df["title"].isin(watched_list)]["genres_clean"] \
                                .astype(str).apply(split_genres).explode()
    top_genres = [g for g, _ in Counter(watched_genres_series).most_common(3)] if not watched_genres_series.empty else []

    genre_recs = get_genre_top_rated(top_genres, watched_list, top_n * 3)  # oversample before sorting
    # Merge: hybrid first, then genre, then overall top, then de-dup
    overall_top = (
        movies_df[~movies_df["title"].isin(watched_list)]
        .sort_values("avg_rating", ascending=False)
        .head(top_n * 3)["title"].tolist()
    )
    merged = list(dict.fromkeys(hybrid + genre_recs + overall_top))
    # Rank by avg_rating and return top_n
    ranked = (
        movies_df[movies_df["title"].isin(merged)]
        .sort_values("avg_rating", ascending=False)
        .head(top_n)["title"].tolist()
    )
    return ranked


# ==========================
# UI Components
# ==========================
def md5_key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:10]

def movie_card(
    tab_prefix: str,
    title: str,
    genres_clean: str,
    avg_rating: float,
    watched: bool,
    on_watch=None,
    on_unwatch=None,
    allow_unwatch: bool = True,
):
    """Card-style movie block with safe unique keys per tab."""
    key_base = md5_key(title)
    with st.container():
        # Visual cue for watched
        badge = " · ✅ **Watched**" if watched else ""
        st.markdown(f"### {title}{badge}")
        st.markdown(f"**Genres:** {genres_clean}")
        st.markdown(f"**Rating:** {avg_rating:.2f} {rating_to_stars(avg_rating)}")

        cols = st.columns([1, 1, 6])
        # Buttons: show only one appropriate action
        if not watched and on_watch:
            if cols[0].button("✅ Mark Watched", key=f"{tab_prefix}_watch_{key_base}"):
                on_watch(title)
                st.experimental_rerun()
        elif watched and allow_unwatch and on_unwatch:
            if cols[0].button("❌ Unwatch", key=f"{tab_prefix}_unwatch_{key_base}"):
                on_unwatch(title)
                st.experimental_rerun()

        st.markdown("---")


# ==========================
# Session State
# ==========================
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""


# ==========================
# Auth Screens
# ==========================
st.title("🎬 Movie Recommender System")

if not st.session_state["logged_in"]:
    choice = st.sidebar.radio("Menu", ["Login", "Signup"], index=0)

    if choice == "Signup":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_pass = st.text_input("Password", type="password")
        if st.button("Sign Up"):
            signup(new_user, new_pass)

    else:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if login(username, password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.success(f"Welcome {username}!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password.")

# ==========================
# Main Dashboard
# ==========================
if st.session_state["logged_in"]:
    st.sidebar.success(f"Logged in as: {st.session_state['username']}")
    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
        st.experimental_rerun()

    st.subheader(f"Hello, {st.session_state['username']}!")

    tab1, tab2, tab3 = st.tabs(["🌟 Top Rated", "🎯 Recommendations", "📖 Watched History"])

    # ---- Tab 1: Top Rated ----
    with tab1:
        top_movies = movies_df.sort_values("avg_rating", ascending=False).head(12)
        watched_now = set(get_watched(st.session_state["username"]))
        for _, row in top_movies.iterrows():
            title = row["title"]
            movie_card(
                tab_prefix="top",
                title=title,
                genres_clean=row["genres_clean"],
                avg_rating=row["avg_rating"],
                watched=(title in watched_now),
                on_watch=lambda t, u=st.session_state["username"]: mark_watched(u, t),
                on_unwatch=lambda t, u=st.session_state["username"]: unwatch_movie(u, t),
                allow_unwatch=True,   # allowed here
            )

    # ---- Tab 2: Recommendations ----
    with tab2:
        recs = get_combined_recommendations(st.session_state["username"], top_n=TOP_N)
        if not recs:
            st.info("No personalized recommendations yet — watch a few movies first!")
        watched_now = set(get_watched(st.session_state["username"]))
        for title in recs:
            row = movies_df[movies_df["title"] == title]
            if row.empty:
                continue
            row = row.iloc[0]
            movie_card(
                tab_prefix="rec",
                title=title,
                genres_clean=row["genres_clean"],
                avg_rating=row["avg_rating"],
                watched=(title in watched_now),
                on_watch=lambda t, u=st.session_state["username"]: mark_watched(u, t),
                on_unwatch=lambda t, u=st.session_state["username"]: unwatch_movie(u, t),
                allow_unwatch=True,   # allowed here
            )

    # ---- Tab 3: Watched History ----
    with tab3:
        watched_list = get_watched(st.session_state["username"])
        if not watched_list:
            st.info("You haven't watched anything yet.")
        else:
            # Show watched movies as read-only cards (NO unwatch button here)
            for title in watched_list:
                row = movies_df[movies_df["title"] == title]
                if row.empty:
                    # If a watched title isn't in metadata (mismatch), still show the title plainly
                    st.markdown(f"### {title} · ✅ **Watched**")
                    st.markdown("---")
                    continue
                row = row.iloc[0]
                movie_card(
                    tab_prefix="hist",
                    title=title,
                    genres_clean=row["genres_clean"],
                    avg_rating=row["avg_rating"],
                    watched=True,
                    on_watch=None,
                    on_unwatch=None,
                    allow_unwatch=False,  # 🚫 as requested
                )
