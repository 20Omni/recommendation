import streamlit as st
import pandas as pd
import pickle
import sqlite3
from collections import Counter

# ---------------- Paths ----------------
HYBRID_MODEL_PATH = "hybrid_recommender.pkl"
MOVIE_METADATA_PATH = "movie_metadata.csv"
DB_PATH = "users.db"
TOP_N = 10

# ---------------- Load Data ----------------
with open(HYBRID_MODEL_PATH, "rb") as f:
    hybrid_data = pickle.load(f)

final_recs = hybrid_data["final_recs"]  # {user_id: [movie titles]}
movies_df = pd.read_csv(MOVIE_METADATA_PATH)  # has title, genres_clean, avg_rating

# ---------------- Database Setup ----------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)""")
c.execute("""CREATE TABLE IF NOT EXISTS watched (
    username TEXT,
    movie_title TEXT,
    FOREIGN KEY(username) REFERENCES users(username)
)""")
conn.commit()

# ---------------- Helper Functions ----------------
def signup(username, password):
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        st.success("Signup successful! Please log in.")
    except sqlite3.IntegrityError:
        st.error("Username already exists.")

def login(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    return c.fetchone() is not None

def mark_watched(username, movie_title):
    c.execute("INSERT INTO watched (username, movie_title) VALUES (?, ?)", (username, movie_title))
    conn.commit()

def get_watched(username):
    c.execute("SELECT movie_title FROM watched WHERE username=?", (username,))
    return [x[0] for x in c.fetchall()]

def get_genre_recommendations(username, top_n=TOP_N):
    watched = get_watched(username)
    if not watched:
        return movies_df.sort_values(by="avg_rating", ascending=False).head(top_n)["title"].tolist(), None

    watched_genres = movies_df[movies_df['title'].isin(watched)]['genres_clean'].str.split('|').explode()
    top_genre = Counter(watched_genres).most_common(1)[0][0] if not watched_genres.empty else None
    top_genres = [g for g, _ in Counter(watched_genres).most_common(4)]

    recs = movies_df[~movies_df['title'].isin(watched)]
    recs = recs[recs['genres_clean'].apply(lambda g: any(genre in g for genre in top_genres))]
    recs = recs.sort_values(by="avg_rating", ascending=False).head(top_n)
    return recs["title"].tolist(), top_genre

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.markdown("<h1 style='color:red; font-size: 40px;'>🎬 Movie Recommender System</h1>", unsafe_allow_html=True)

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''

# Sidebar menu
if st.session_state['logged_in']:
    st.sidebar.write(f"👤 Logged in as **{st.session_state['username']}**")
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = ''
        st.experimental_rerun()
else:
    menu = ["Login", "Signup"]
    choice = st.sidebar.radio("Menu", menu)

    if choice == "Signup":
        st.subheader("🆕 Create New Account")
        new_user = st.text_input("Username")
        new_pass = st.text_input("Password", type='password')
        if st.button("Sign Up"):
            signup(new_user, new_pass)

    elif choice == "Login":
        st.subheader("🔑 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            if login(username, password):
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.success(f"Welcome {username}!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")

# ---------------- Main Content ----------------
if st.session_state['logged_in']:
    tabs = st.tabs(["🎯 Recommendations", "🌟 Top Rated", "📖 Watched History"])

    # Recommendations Tab
    with tabs[0]:
        username = st.session_state['username']

        try:
            user_id = int(username)  # if username is numeric for ML model
        except ValueError:
            user_id = None

        if user_id in final_recs:
            st.markdown("<h2 style='color:red;'>🤖 For You</h2>", unsafe_allow_html=True)
            st.caption("Based on your past ratings & similar users")
            recs = final_recs[user_id]
            top_genre = None
        else:
            recs, top_genre = get_genre_recommendations(username)
            if top_genre:
                st.markdown(f"<h2 style='color:red;'>🎭 Because you watched {top_genre} movies</h2>", unsafe_allow_html=True)
            else:
                st.markdown("<h2 style='color:red;'>🎭 Recommended For You</h2>", unsafe_allow_html=True)

        rec_df = movies_df[movies_df['title'].isin(recs)]
        for _, row in rec_df.iterrows():
            col1, col2 = st.columns([4, 1])
            col1.write(f"**{row['title']}** ({row['genres_clean']}) — ⭐ {row['avg_rating']:.2f}")
            if col2.button("Watched ✅", key=f"rec_{row['title']}"):
                mark_watched(st.session_state['username'], row['title'])
                st.success(f"Marked '{row['title']}' as watched!")
                st.experimental_rerun()

    # Top Rated Tab
    with tabs[1]:
        st.markdown("<h2 style='color:red;'>🌟 Top Rated</h2>", unsafe_allow_html=True)
        watched_movies = get_watched(st.session_state['username'])
        top_movies = movies_df[~movies_df['title'].isin(watched_movies)].sort_values(by="avg_rating", ascending=False).head(10)
        for _, row in top_movies.iterrows():
            col1, col2 = st.columns([4, 1])
            col1.write(f"**{row['title']}** ({row['genres_clean']}) — ⭐ {row['avg_rating']:.2f}")
            if col2.button("Watched ✅", key=f"top_{row['title']}"):
                mark_watched(st.session_state['username'], row['title'])
                st.success(f"Marked '{row['title']}' as watched!")
                st.experimental_rerun()

    # Watched History Tab
    with tabs[2]:
        st.markdown("<h2 style='color:red;'>📖 Your Watched History</h2>", unsafe_allow_html=True)
        watched_list = get_watched(st.session_state['username'])
        if watched_list:
            for movie in watched_list:
                genres = movies_df.loc[movies_df['title'] == movie, 'genres_clean'].values
                rating = movies_df.loc[movies_df['title'] == movie, 'avg_rating'].values
                genres_str = genres[0] if len(genres) else "Unknown"
                rating_val = rating[0] if len(rating) else 0
                st.write(f"**{movie}** ({genres_str}) — ⭐ {rating_val:.2f}")
        else:
            st.info("You haven't watched anything yet.")
