import streamlit as st
import pandas as pd
import hashlib

# ==========================
# Data Preparation
# ==========================
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

# ==========================
# User Management
# ==========================
users_db = {}

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def signup(username, password):
    if username in users_db:
        return False
    users_db[username] = {"password": hash_password(password), "watched": []}
    return True

def login(username, password):
    if username in users_db and users_db[username]["password"] == hash_password(password):
        return True
    return False

def watch_movie(username, title):
    if title not in users_db[username]["watched"]:
        users_db[username]["watched"].append(title)

# ==========================
# Utility Functions
# ==========================
def rating_to_stars(rating):
    full_stars = int(round(rating))
    return "⭐" * full_stars + "☆" * (5 - full_stars)

def movie_card(tab_prefix, title, genres, rating, action_label=None, action_func=None):
    """Reusable movie display card with optional action button"""
    with st.container():
        st.markdown(f"### {title}")
        st.markdown(f"**Genres:** {genres}")
        st.markdown(f"**Rating:** {rating:.2f} {rating_to_stars(rating)}")
        if action_label and action_func:
            if st.button(action_label, key=f"{tab_prefix}_{title}"):
                action_func(title)
                st.experimental_rerun()
        st.markdown("---")

# ==========================
# Streamlit App
# ==========================
st.set_page_config(page_title="Movie Recommender", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

if not st.session_state.logged_in:
    st.title("🎬 Movie Recommender")

    choice = st.radio("Login / Signup", ["Login", "Signup"])

    if choice == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if login(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Welcome back, {username}!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password.")

    elif choice == "Signup":
        username = st.text_input("Choose a Username")
        password = st.text_input("Choose a Password", type="password")
        if st.button("Signup"):
            if signup(username, password):
                st.success("Signup successful! Please login.")
            else:
                st.error("Username already exists.")

else:
    st.sidebar.success(f"Logged in as {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.experimental_rerun()

    st.title("🎥 Your Movie Dashboard")

    tab1, tab2, tab3 = st.tabs(["⭐ Top Rated", "🎯 Recommendations", "📜 Watched History"])

    # Top Rated Tab
    with tab1:
        st.subheader("Top Rated Movies")
        top_movies = movie_data.sort_values(by="avg_rating", ascending=False).head(15)
        for _, row in top_movies.iterrows():
            movie_card("top", row["title"], row["genres_clean"], row["avg_rating"],
                       "✅ Watch", lambda title=row["title"]: watch_movie(st.session_state.username, title))

    # Recommendations Tab (simple example: recommend same genre as first watched movie)
    with tab2:
        st.subheader("Recommended for You")
        watched = users_db[st.session_state.username]["watched"]
        if watched:
            first_genre = movie_data[movie_data["title"] == watched[0]]["genres_clean"].values[0]
            recs = movie_data[movie_data["genres_clean"] == first_genre].sort_values(by="avg_rating", ascending=False).head(15)
            for _, row in recs.iterrows():
                if row["title"] not in watched:
                    movie_card("rec", row["title"], row["genres_clean"], row["avg_rating"],
                               "✅ Watch", lambda title=row["title"]: watch_movie(st.session_state.username, title))
        else:
            st.info("Watch some movies to get recommendations.")

    # Watched History Tab
    with tab3:
        st.subheader("Your Watched Movies")
        watched = users_db[st.session_state.username]["watched"]
        if watched:
            for movie in watched:
                row = movie_data[movie_data["title"] == movie].iloc[0]
                movie_card("hist", movie, row["genres_clean"], row["avg_rating"])
        else:
            st.info("You haven't watched any movies yet.")
