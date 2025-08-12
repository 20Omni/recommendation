import streamlit as st
import pandas as pd
import pickle
import os

# ------------------------
# Load hybrid recommendations
# ------------------------
@st.cache_data
def load_hybrid():
    with open("hybrid_recommender_light.pkl", "rb") as f:
        data = pickle.load(f)
    return data["recommendations"], data["movies_meta"]

recommendations_dict, movies_meta = load_hybrid()

# ------------------------
# User storage file
# ------------------------
USERS_FILE = "users.csv"
if not os.path.exists(USERS_FILE):
    pd.DataFrame(columns=["username", "password", "preferred_genres"]).to_csv(USERS_FILE, index=False)

def load_users():
    return pd.read_csv(USERS_FILE)

def save_users(df):
    df.to_csv(USERS_FILE, index=False)

# ------------------------
# Login / Signup
# ------------------------
st.sidebar.title("🎬 Movie Recommender Login")
menu = st.sidebar.radio("Menu", ["Login", "Signup"])

if menu == "Signup":
    st.subheader("Create an Account")
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    genres = st.multiselect("Select Your Favorite Genres", 
                            movies_meta["genres"].explode().unique())

    if st.button("Signup"):
        users = load_users()
        if username in users["username"].values:
            st.error("Username already exists. Please login.")
        else:
            users = pd.concat([users, pd.DataFrame({
                "username": [username],
                "password": [password],
                "preferred_genres": [",".join(genres)]
            })], ignore_index=True)
            save_users(users)
            st.success("Account created! Please login now.")

elif menu == "Login":
    st.subheader("Login to Your Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        users = load_users()
        if ((users["username"] == username) & (users["password"] == password)).any():
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            user_row = users[users["username"] == username].iloc[0]
            st.session_state["preferred_genres"] = user_row["preferred_genres"].split(",") if pd.notna(user_row["preferred_genres"]) else []
            st.success(f"Welcome back, {username}!")
        else:
            st.error("Invalid username or password.")

# ------------------------
# Recommendations
# ------------------------
if st.session_state.get("logged_in", False):
    st.title("🍿 Personalized Movie Recommendations")

    # Show user's genre prefs
    if st.session_state["preferred_genres"]:
        st.write("Your Preferred Genres:", ", ".join(st.session_state["preferred_genres"]))
    else:
        st.write("You haven’t set any preferred genres yet.")

    user_id = st.number_input("Enter your User ID", min_value=1, step=1)

    if st.button("Get Recommendations"):
        if user_id in recommendations_dict:
            recs = recommendations_dict[user_id]
            rec_df = pd.DataFrame(recs, columns=["movie_id", "score"])
            rec_df = rec_df.merge(movies_meta, on="movie_id", how="left")

            # Boost preferred genres
            if st.session_state["preferred_genres"]:
                rec_df["boost"] = rec_df["genres"].apply(
                    lambda g: 1 if any(pref in g for pref in st.session_state["preferred_genres"]) else 0
                )
                rec_df["score"] = rec_df["score"] + rec_df["boost"] * 0.2

            rec_df = rec_df.sort_values(by="score", ascending=False).head(10)
            st.table(rec_df[["title", "genres", "score"]])
        else:
            st.warning("No recommendations found for this user ID.")
