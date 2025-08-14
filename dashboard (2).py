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

final_recs = hybrid_data["final_recs"]
weights = hybrid_data["weights"]

# movie_metadata.csv now has title, genres_clean, avg_rating
movies_df = pd.read_csv(MOVIE_METADATA_PATH)

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
        return movies_df.sort_values('avg_rating', ascending=False).head(top_n)['title'].tolist()
    
    watched_genres = movies_df[movies_df['title'].isin(watched)]['genres_clean'].str.split(',').explode()
    top_genres = [g.strip() for g, _ in Counter(watched_genres).most_common(3)]
    
    recs = []
    for _, row in movies_df.sort_values('avg_rating', ascending=False).iterrows():
        if row['title'] in watched:
            continue
        if any(g in row['genres_clean'] for g in top_genres):
            recs.append(row['title'])
        if len(recs) >= top_n:
            break
    return recs

# ---------------- Streamlit UI ----------------
st.title("🎬 Movie Recommender System")

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''

menu = ["Login", "Signup"]
choice = st.sidebar.selectbox("Menu", menu)

if not st.session_state['logged_in']:
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

if st.session_state['logged_in']:
    st.subheader(f"Hello, {st.session_state['username']}!")

    tab1, tab2, tab3 = st.tabs(["🌟 Top Rated Movies", "🎯 Your Recommendations", "📖 Watched History"])

    # Tab 1: Top Rated Movies
    with tab1:
        top_movies = movies_df.sort_values('avg_rating', ascending=False).head(10)
        watched_list = get_watched(st.session_state['username'])

        for _, row in top_movies.iterrows():
            col1, col2 = st.columns([3, 1])
            movie_display = f"~~{row['title']} ({row['genres_clean']})~~" if row['title'] in watched_list else f"{row['title']} ({row['genres_clean']})"
            col1.markdown(movie_display)
            if row['title'] not in watched_list:
                if col2.button("Watched ✅", key=f"top_{row['title']}"):
                    mark_watched(st.session_state['username'], row['title'])
                    st.success(f"Marked '{row['title']}' as watched!")

    # Tab 2: Recommendations
    with tab2:
        recs = get_genre_recommendations(st.session_state['username'])
        watched_list = get_watched(st.session_state['username'])
        for movie in recs:
            movie_display = f"~~{movie}~~" if movie in watched_list else movie
            col1, col2 = st.columns([3, 1])
            col1.markdown(movie_display)
            if movie not in watched_list:
                if col2.button("Watched ✅", key=f"rec_{movie}"):
                    mark_watched(st.session_state['username'], movie)
                    st.success(f"Marked '{movie}' as watched!")

    # Tab 3: Watched History
    with tab3:
        watched_list = get_watched(st.session_state['username'])
        if watched_list:
            st.write(watched_list)
        else:
            st.info("You haven't watched anything yet.")
