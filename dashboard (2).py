import streamlit as st
import pandas as pd
import pickle
import sqlite3
from collections import Counter

# ---------------- Paths ----------------
HYBRID_MODEL_PATH = "hybrid_recommender.pkl"
MOVIE_METADATA_PATH = "movie_metadata.csv"  # Has title, genres_clean, avg_rating
DB_PATH = "users.db"

TOP_N = 10

# ---------------- Load Data ----------------
with open(HYBRID_MODEL_PATH, "rb") as f:
    hybrid_data = pickle.load(f)

final_recs = hybrid_data["final_recs"]
weights = hybrid_data["weights"]

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

def unwatch_movie(username, movie_title):
    c.execute("DELETE FROM watched WHERE username=? AND movie_title=?", (username, movie_title))
    conn.commit()

def get_watched(username):
    c.execute("SELECT movie_title FROM watched WHERE username=?", (username,))
    return [x[0] for x in c.fetchall()]

def get_combined_recommendations(username, top_n=TOP_N):
    watched = get_watched(username)

    # Hybrid model recommendations (if available)
    hybrid_recs = final_recs.get(username, [])

    # Genre-based recommendations
    watched_genres = movies_df[movies_df['title'].isin(watched)]['genres_clean'].str.split(',').explode()
    top_genres = [g.strip() for g, _ in Counter(watched_genres).most_common(3)]
    genre_recs = [
        row['title'] for _, row in movies_df.iterrows()
        if any(g in row['genres_clean'] for g in top_genres) and row['title'] not in watched
    ]

    # Merge + sort by avg_rating
    combined = list(dict.fromkeys(hybrid_recs + genre_recs))  # remove duplicates
    combined_sorted = (
        movies_df[movies_df['title'].isin(combined)]
        .sort_values('avg_rating', ascending=False)
        .head(top_n)['title']
        .tolist()
    )
    return combined_sorted

def rating_to_stars(rating):
    """Convert numeric rating (0-5) to star emojis"""
    full_stars = int(round(rating))
    return "⭐" * full_stars + "☆" * (5 - full_stars)

def movie_card(title, genres, rating, watched, action_button_key, action_label, action_func):
    """Reusable movie display card"""
    with st.container():
        st.markdown(f"**{title}**")
        st.markdown(f"*Genres:* {genres}")
        st.markdown(f"*Rating:* {rating:.2f} {rating_to_stars(rating)}")
        if st.button(action_label, key=action_button_key):
            action_func(title)
            st.experimental_rerun()
        st.markdown("---")

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
        watched_list = get_watched(st.session_state['username'])
        top_movies = movies_df.sort_values('avg_rating', ascending=False).head(10)
        for _, row in top_movies.iterrows():
            if row['title'] in watched_list:
                movie_card(row['title'], row['genres_clean'], row['avg_rating'], True,
                           f"unwatch_{row['title']}", "❌ Unwatch", 
                           lambda title=row['title']: unwatch_movie(st.session_state['username'], title))
            else:
                movie_card(row['title'], row['genres_clean'], row['avg_rating'], False,
                           f"watch_{row['title']}", "✅ Mark Watched", 
                           lambda title=row['title']: mark_watched(st.session_state['username'], title))

    # Tab 2: Recommendations
    with tab2:
        watched_list = get_watched(st.session_state['username'])
        recs = get_combined_recommendations(st.session_state['username'])
        for movie in recs:
            row = movies_df[movies_df['title'] == movie].iloc[0]
            if movie in watched_list:
                movie_card(movie, row['genres_clean'], row['avg_rating'], True,
                           f"unwatch_rec_{movie}", "❌ Unwatch", 
                           lambda title=movie: unwatch_movie(st.session_state['username'], title))
            else:
                movie_card(movie, row['genres_clean'], row['avg_rating'], False,
                           f"watch_rec_{movie}", "✅ Mark Watched", 
                           lambda title=movie: mark_watched(st.session_state['username'], title))

    # Tab 3: Watched History
    with tab3:
        watched_list = get_watched(st.session_state['username'])
        if watched_list:
            for movie in watched_list:
                row = movies_df[movies_df['title'] == movie].iloc[0]
                movie_card(movie, row['genres_clean'], row['avg_rating'], True,
                           f"unwatch_hist_{movie}", "❌ Unwatch", 
                           lambda title=movie: unwatch_movie(st.session_state['username'], title))
        else:
            st.info("You haven't watched anything yet.")
