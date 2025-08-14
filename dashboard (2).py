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
        return movies_df.sort_values(by="avg_rating", ascending=False).head(top_n)["title"].tolist()

    watched_genres = movies_df[movies_df['title'].isin(watched)]['genres_clean'].str.split('|').explode()
    top_genres = [g for g, _ in Counter(watched_genres).most_common(4)]  # top 4 genres

    recs = movies_df[~movies_df['title'].isin(watched)]
    recs = recs[recs['genres_clean'].apply(lambda g: any(genre in g for genre in top_genres))]
    recs = recs.sort_values(by="avg_rating", ascending=False).head(top_n)
    return recs["title"].tolist()

def display_movie_cards(movie_titles, username, prefix):
    rec_df = movies_df[movies_df['title'].isin(movie_titles)]
    cols = st.columns(3)
    for i, (_, row) in enumerate(rec_df.iterrows()):
        with cols[i % 3]:
            st.markdown(f"**{row['title']}**  \n_{row['genres_clean']}_  \n⭐ {row['avg_rating']:.2f}")
            if st.button("Watched ✅", key=f"{prefix}_{row['title']}"):
                mark_watched(username, row['title'])
                st.success(f"Marked '{row['title']}' as watched!")

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
else:
    tabs = st.tabs(["🌟 Top Rated", "🎯 Recommendations", "📖 Watched History"])

    # Top Rated Tab
    with tabs[0]:
        top_movies = movies_df.sort_values(by="avg_rating", ascending=False).head(TOP_N)
        display_movie_cards(top_movies["title"].tolist(), st.session_state['username'], "top")

    # Recommendations Tab
    with tabs[1]:
        username = st.session_state['username']
        try:
            user_id = int(username)  # Assuming username is user_id for trained users
        except ValueError:
            user_id = None

        if user_id in final_recs:
            recs = final_recs[user_id]  # ML-based
        else:
            recs = get_genre_recommendations(username)  # Silent fallback

        display_movie_cards(recs, username, "rec")

    # Watched History Tab
    with tabs[2]:
        watched_list = get_watched(st.session_state['username'])
        if watched_list:
            cols = st.columns(3)
            for i, movie in enumerate(watched_list):
                genres = movies_df.loc[movies_df['title'] == movie, 'genres_clean'].values
                rating = movies_df.loc[movies_df['title'] == movie, 'avg_rating'].values
                genres_str = genres[0] if len(genres) else "Unknown"
                rating_val = rating[0] if len(rating) else 0
                with cols[i % 3]:
                    st.markdown(f"**{movie}**  \n_{genres_str}_  \n⭐ {rating_val:.2f}")
        else:
            st.info("You haven't watched anything yet.")
