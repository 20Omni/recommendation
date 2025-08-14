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
# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: white;
        }
        .movie-card {
            background-color: #1e1e1e;
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            transition: transform 0.2s;
        }
        .movie-card:hover {
            transform: scale(1.03);
            box-shadow: 0 4px 20px rgba(255, 0, 0, 0.4);
        }
        .movie-title {
            font-size: 18px;
            font-weight: bold;
            color: #ff4b4b;
        }
        .movie-genre {
            font-size: 14px;
            color: #cccccc;
        }
        .movie-rating {
            font-size: 14px;
            color: gold;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color:#ff4b4b; text-align:center;'>🎬 Movie Recommender System</h1>", unsafe_allow_html=True)

# Sidebar login/logout
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''

if st.session_state['logged_in']:
    st.sidebar.success(f"👤 Logged in as **{st.session_state['username']}**")
    if st.sidebar.button("🚪 Logout"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = ''
        st.rerun()
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
                st.success(f"Welcome {username}! 🎉")
                st.rerun()
            else:
                st.error("Invalid username or password ❌")

# ---------------- Main Content ----------------
if st.session_state['logged_in']:
    tabs = st.tabs(["🎯 Recommendations", "🌟 Top Rated", "📖 Watched History"])

    # Recommendations Tab
    with tabs[0]:
        username = st.session_state['username']

        try:
            user_id = int(username)
        except ValueError:
            user_id = None

        if user_id in final_recs:
            st.subheader("🤖 Personalized For You")
            recs = final_recs[user_id]
        else:
            recs, top_genre = get_genre_recommendations(username)
            if top_genre:
                st.subheader(f"🎭 Because you like **{top_genre}** movies")
            else:
                st.subheader("🎭 Recommended For You")

        rec_df = movies_df[movies_df['title'].isin(recs)]

        # Display movies in a grid (3 per row)
        cols = st.columns(3)
        for i, (_, row) in enumerate(rec_df.iterrows()):
            with cols[i % 3]:
                st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-title">{row['title']}</div>
                        <div class="movie-genre">{row['genres_clean']}</div>
                        <div class="movie-rating">⭐ {row['avg_rating']:.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
                if st.button("✅ Watched", key=f"rec_{row['title']}"):
                    mark_watched(username, row['title'])
                    st.experimental_rerun()

    # Top Rated Tab
    with tabs[1]:
        st.subheader("🌟 Top Rated Movies")
        watched_movies = get_watched(username)
        top_movies = movies_df[~movies_df['title'].isin(watched_movies)].sort_values(by="avg_rating", ascending=False).head(12)

        cols = st.columns(3)
        for i, (_, row) in enumerate(top_movies.iterrows()):
            with cols[i % 3]:
                st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-title">{row['title']}</div>
                        <div class="movie-genre">{row['genres_clean']}</div>
                        <div class="movie-rating">⭐ {row['avg_rating']:.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
                if st.button("✅ Watched", key=f"top_{row['title']}"):
                    mark_watched(username, row['title'])
                    st.experimental_rerun()

    # Watched History Tab
    with tabs[2]:
        st.subheader("📖 Your Watched History")
        watched_list = get_watched(username)
        if watched_list:
            for movie in watched_list:
                genres = movies_df.loc[movies_df['title'] == movie, 'genres_clean'].values
                rating = movies_df.loc[movies_df['title'] == movie, 'avg_rating'].values
                genres_str = genres[0] if len(genres) else "Unknown"
                rating_val = rating[0] if len(rating) else 0
                st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-title">{movie}</div>
                        <div class="movie-genre">{genres_str}</div>
                        <div class="movie-rating">⭐ {rating_val:.2f}</div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("You haven't watched anything yet.")

