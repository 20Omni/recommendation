import streamlit as st
import pandas as pd
import pickle
import sqlite3
from collections import Counter

# ---------------- Paths ----------------
HYBRID_MODEL_PATH = "hybrid_recommender.pkl"
MOVIE_METADATA_PATH = "movie_metadata.csv"   # columns: title, genres_clean, avg_rating
DB_PATH = "users.db"
TOP_N = 10

st.set_page_config(page_title="🎬 Movie Recommender System", layout="wide")

# ---------------- Load Data ----------------
with open(HYBRID_MODEL_PATH, "rb") as f:
    hybrid_data = pickle.load(f)
final_recs = hybrid_data.get("final_recs", {})

movies_df = pd.read_csv(MOVIE_METADATA_PATH).copy()
# Normalize columns
movies_df["title"] = movies_df["title"].astype(str)
movies_df["genres_clean"] = movies_df["genres_clean"].astype(str)
movies_df["avg_rating"] = pd.to_numeric(movies_df["avg_rating"], errors="coerce").fillna(0.0)
movies_df.drop_duplicates(subset=["title"], inplace=True)

# ---------------- Database Setup ----------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)""")
# Unique pair prevents duplicates if we use INSERT OR IGNORE
c.execute("""CREATE TABLE IF NOT EXISTS watched (
    username TEXT,
    movie_title TEXT,
    UNIQUE(username, movie_title),
    FOREIGN KEY(username) REFERENCES users(username)
)""")
conn.commit()

# ---------------- Helpers ----------------
def signup(username, password):
    if not username or not password:
        st.error("Username and password are required.")
        return
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        st.success("Signup successful! Please log in.")
    except sqlite3.IntegrityError:
        st.error("Username already exists.")

def login(username, password):
    c.execute("SELECT 1 FROM users WHERE username=? AND password=?", (username, password))
    return c.fetchone() is not None

def mark_watched(username, movie_title):
    c.execute("INSERT OR IGNORE INTO watched (username, movie_title) VALUES (?, ?)", (username, movie_title))
    conn.commit()

def get_watched(username):
    c.execute("SELECT movie_title FROM watched WHERE username=?", (username,))
    return [x[0] for x in c.fetchall()]

def parse_genres(g):
    """Supports formats like `Drama|Crime`, `Drama, Crime`, or \"['Drama','Crime']\"."""
    s = str(g).strip()
    if s.startswith("[") and s.endswith("]"):
        # safe-ish parse: remove brackets/quotes and split by comma
        s = s[1:-1].replace("'", "").replace('"', "")
    s = s.replace("|", ",")
    return ", ".join([x.strip() for x in s.split(",") if x.strip()])

def rating_to_stars(r):
    # assumes 0-5 scale; round to nearest star
    try:
        r = float(r)
    except:
        r = 0.0
    stars = int(round(min(max(r, 0.0), 5.0)))
    return "⭐" * stars + "☆" * (5 - stars)

def movie_card(title, genres, rating, show_button=False, button_label="Watched ✅", button_key=None, on_click=None):
    """Pretty multi-line movie display with optional button."""
    # Simple card styling via HTML (safe to use in Streamlit)
    st.markdown(
        f"""
        <div style="border:1px solid #e6e6e6;border-radius:10px;padding:14px;margin:8px 0;background-color:#fbfbfb;">
            <div style="font-weight:700;font-size:18px;color:#222;">{title}</div>
            <div style="margin-top:6px;color:#666;"><b>Genres:</b> {genres}</div>
            <div style="margin-top:4px;color:#444;"><b>Rating:</b> {rating:.2f} &nbsp; {rating_to_stars(rating)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if show_button and on_click:
        if st.button(button_label, key=button_key):
            on_click()
            st.rerun()

def get_genre_recommendations(username, top_n=TOP_N):
    watched = get_watched(username)
    if not watched:
        # No history yet -> overall top rated
        return movies_df.sort_values(by="avg_rating", ascending=False).head(top_n)["title"].tolist()

    # Count favorite genres from watched
    watched_genres = movies_df[movies_df["title"].isin(watched)]["genres_clean"].astype(str)
    watched_genres = watched_genres.str.replace("|", ",")
    watched_genres = watched_genres.str.split(",").explode().str.strip()
    top_genres = [g for g, _ in Counter(watched_genres.dropna()).most_common(3)]

    # Recommend top-rated movies matching favorite genres (not yet watched)
    recs_df = movies_df[~movies_df["title"].isin(watched)].copy()
    if top_genres:
        recs_df = recs_df[recs_df["genres_clean"].apply(lambda g: any(tg in str(g) for tg in top_genres))]
    recs_df = recs_df.sort_values(by="avg_rating", ascending=False).head(top_n)
    return recs_df["title"].tolist()

# ---------------- Streamlit UI ----------------
st.title("🎬 Movie Recommender System")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""

menu = ["Login", "Signup"]
choice = st.sidebar.selectbox("Menu", menu)

if not st.session_state["logged_in"]:
    if choice == "Signup":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_pass = st.text_input("Password", type="password")
        if st.button("Sign Up"):
            signup(new_user, new_pass)

    else:  # Login
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if login(username, password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.success(f"Welcome {username}!")
                st.rerun()
            else:
                st.error("Invalid username or password")

# ---------------- Main Dashboard ----------------
if st.session_state["logged_in"]:
    tabs = st.tabs(["🌟 Top Rated", "🎯 Recommendations", "📖 Watched History"])

    # Top Rated Tab
    with tabs[0]:
        st.subheader("Top Rated Movies")
        top_movies = movies_df.sort_values(by="avg_rating", ascending=False).head(15)
        for _, row in top_movies.iterrows():
            title = row["title"]
            genres = parse_genres(row["genres_clean"])
            rating = row["avg_rating"]
            movie_card(
                title,
                genres,
                rating,
                show_button=True,
                button_label="Watched ✅",
                button_key=f"top_watch_{title}",
                on_click=lambda t=title: mark_watched(st.session_state["username"], t),
            )

    # Recommendations Tab
    with tabs[1]:
        st.subheader("Recommended for You")
        rec_titles = get_genre_recommendations(st.session_state["username"])
        rec_df = movies_df[movies_df["title"].isin(rec_titles)].copy()
        for _, row in rec_df.iterrows():
            title = row["title"]
            genres = parse_genres(row["genres_clean"])
            rating = row["avg_rating"]
            movie_card(
                title,
                genres,
                rating,
                show_button=True,
                button_label="Watched ✅",
                button_key=f"rec_watch_{title}",
                on_click=lambda t=title: mark_watched(st.session_state["username"], t),
            )

    # Watched History Tab (no unwatch)
    with tabs[2]:
        st.subheader("Your Watched Movies")
        watched_list = get_watched(st.session_state["username"])
        if watched_list:
            for title in watched_list:
                row = movies_df[movies_df["title"] == title]
                if row.empty:
                    # fallback if metadata missing
                    movie_card(title, "Unknown", 0.0, show_button=False)
                else:
                    r = row.iloc[0]
                    movie_card(
                        title,
                        parse_genres(r["genres_clean"]),
                        r["avg_rating"],
                        show_button=False  # 🚫 no unwatch here
                    )
        else:
            st.info("You haven't watched anything yet.")
