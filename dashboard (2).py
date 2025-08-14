import streamlit as st
import pandas as pd
import pickle
import sqlite3
from collections import Counter

# ---------------- Paths ----------------
HYBRID_MODEL_PATH = "hybrid_recommender.pkl"  # directly from GitHub repo or local
MOVIE_METADATA_PATH = "movie_metadata.csv"   # directly from GitHub repo or local
DB_PATH = "users.db"

TOP_N = 10

# ---------------- Load Data ----------------
with open(HYBRID_MODEL_PATH, "rb") as f:
    hybrid_data = pickle.load(f)

final_recs = hybrid_data["final_recs"]
weights = hybrid_data["weights"]

movies_df = pd.read_csv(MOVIE_METADATA_PATH)  # must have 'title' and 'genres_clean'

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
        return final_recs.get(username, [])[:top_n]

    # Count genres from watched movies
    watched_genres = movies_df[movies_df['title'].isin(watched)]['genres_clean'].str.split('|').explode()
    top_genres = [g for g, _ in Counter(watched_genres).most_common(3)]

    # Recommend movies matching top genres not yet watched
    recs = []
    for _, row in movies_df.iterrows():
        if row['title'] in watched:
            continue
        if any(g in row['genres_clean'].split('|') for g in top_genres):
            recs.append(row['title'])
        if len(recs) >= top_n:
            break
    return recs

# ---------------- Streamlit UI ----------------
st.title("🎬 Movie Recommender System")

# Session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''

menu = ["Login", "Signup"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Signup":
    st.subheader("Create New Account")
    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type='password')
    if st.button("Sign Up"):
        signup(new_user, new_pass)

elif choice == "Login":
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        if login(username, password):
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success(f"Welcome {username}!")
        else:
            st.error("Invalid username or password")

# ---------------- Main Dashboard ----------------
if st.session_state['logged_in']:
    st.subheader(f"Hello, {st.session_state['username']}!")

    # ---------------- Top Movies ----------------
    st.markdown("### 🌟 Top Movies")
    top_movies = movies_df.sort_values('title').head(10)
    for _, row in top_movies.iterrows():
        col1, col2 = st.columns([3,1])
        col1.write(f"{row['title']} ({row['genres_clean']})")
        if col2.button(f"Watched ✅", key=f"top_{row['title']}"):
            mark_watched(st.session_state['username'], row['title'])
            st.success(f"Marked '{row['title']}' as watched!")

    # ---------------- Personalized Recommendations ----------------
    st.markdown("### 🎯 Your Recommendations")
    recs = get_genre_recommendations(st.session_state['username'])
    for movie in recs:
        col1, col2 = st.columns([3,1])
        col1.write(movie)
        if col2.button(f"Watched ✅", key=f"rec_{movie}"):
            mark_watched(st.session_state['username'], movie)
            st.success(f"Marked '{movie}' as watched!")

    # ---------------- Watched History ----------------
    st.markdown("### 📖 Your Watched History")
    watched_list = get_watched(st.session_state['username'])
    st.write(watched_list if watched_list else "You haven't watched anything yet.")
