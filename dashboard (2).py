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
    top_genres = [g for g, _ in Counter(watched_genres).most_common(3)]
    
    recs = movies_df[~movies_df['title'].isin(watched)]
    recs = recs[recs['genres_clean'].apply(lambda g: any(genre in g for genre in top_genres))]
    recs = recs.sort_values(by="avg_rating", ascending=False).head(top_n)
    return recs["title"].tolist()

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

    # ---------------- Top Rated Tab ----------------
    with tabs[0]:
        top_movies = movies_df.sort_values(by="avg_rating", ascending=False).head(10)
        for _, row in top_movies.iterrows():
            col1, col2 = st.columns([3, 1])
            col1.write(f"**{row['title']}** ({row['genres_clean']}) — ⭐ {row['avg_rating']:.2f}")
            if col2.button("Watched ✅", key=f"top_{row['title']}"):
                mark_watched(st.session_state['username'], row['title'])
                st.success(f"Marked '{row['title']}' as watched!")

    # ---------------- Recommendations Tab ----------------
    with tabs[1]:
        username = st.session_state['username']

        # Use hybrid model recommendations if available
        if username in final_recs:
            recs = final_recs[username]
        else:
            recs = get_genre_recommendations(username)

        rec_df = movies_df[movies_df['title'].isin(recs)]
        watched_list = get_watched(username)

        for _, row in rec_df.iterrows():
            col1, col2 = st.columns([3, 1])

            # Find explanation
            explanation = "Recommended based on your viewing history."
            if watched_list:
                rec_genres = set(str(row['genres_clean']).split("|"))
                for watched_movie in watched_list:
                    watched_genres = set(str(movies_df.loc[movies_df['title'] == watched_movie, 'genres_clean'].values[0]).split("|"))
                    if rec_genres & watched_genres:
                        explanation = f"Because you watched **{watched_movie}** ({', '.join(watched_genres)}), we recommend this."
                        break

            col1.markdown(f"**{row['title']}** ({row['genres_clean']}) — ⭐ {row['avg_rating']:.2f}")
            col1.caption(explanation)

            if col2.button("Watched ✅", key=f"rec_{row['title']}"):
                mark_watched(username, row['title'])
                st.success(f"Marked '{row['title']}' as watched!")

    # ---------------- Watched History Tab ----------------
    with tabs[2]:
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
