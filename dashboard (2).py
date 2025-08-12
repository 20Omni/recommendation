import streamlit as st
import pandas as pd
import pickle
import os

# -----------------
# Load hybrid recommender (lightweight)
# -----------------
@st.cache_data
def load_hybrid():
    with open("hybrid_recommender_light.pkl", "rb") as f:
        return pickle.load(f)

model_data = load_hybrid()
hybrid_preds = pd.DataFrame(model_data["recommendations"], columns=["user_id", "movie_id", "score"])
movies_meta = model_data["movies_meta"]

# -----------------
# User data (login / signup)
# -----------------
USERS_FILE = "users.csv"

def load_users():
    if os.path.exists(USERS_FILE):
        return pd.read_csv(USERS_FILE)
    return pd.DataFrame(columns=["username", "password", "genres"])

def save_users(df):
    df.to_csv(USERS_FILE, index=False)

users_df = load_users()

def signup(username, password):
    if username in users_df["username"].values:
        return False, "Username already exists!"
    new_user = pd.DataFrame([[username, password, ""]], columns=["username", "password", "genres"])
    updated_df = pd.concat([users_df, new_user], ignore_index=True)
    save_users(updated_df)
    return True, "Signup successful!"

def login(username, password):
    user = users_df[(users_df["username"] == username) & (users_df["password"] == password)]
    if not user.empty:
        return True
    return False

# -----------------
# Recommend function with genre boost
# -----------------
def get_recommendations(username, top_n=10):
    # Get this user's id → simulate mapping username to numeric user_id
    # For demo purposes, we assume all users share same hybrid_preds user_ids
    # You could later map usernames to real IDs in your ratings data
    user_id = hybrid_preds["user_id"].iloc[0]  # dummy

    # Base recommendations
    user_recs = hybrid_preds[hybrid_preds["user_id"] == user_id].sort_values("score", ascending=False)
    recs = pd.merge(user_recs, movies_meta, on="movie_id")

    # Apply genre boost
    preferred_genres = users_df.loc[users_df["username"] == username, "genres"].values[0]
    if preferred_genres:
        preferred_genres = preferred_genres.split(",")
        recs["boost"] = recs["title"].apply(lambda x: any(g.lower() in x.lower() for g in preferred_genres))
        recs["score"] += recs["boost"] * 0.5  # boost score if genre match
        recs = recs.sort_values("score", ascending=False)

    return recs.head(top_n)

# -----------------
# Streamlit UI
# -----------------
st.title("🎬 Hybrid Movie Recommender")

menu = ["Login", "Signup"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Signup":
    st.subheader("Create New Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Signup"):
        success, msg = signup(username, password)
        st.success(msg) if success else st.error(msg)

elif choice == "Login":
    st.subheader("Login to your account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login(username, password):
            st.success(f"Welcome {username}!")
            # Genre preferences
            genres = st.text_input("Enter your favorite genres (comma-separated)", 
                                   value=users_df.loc[users_df["username"] == username, "genres"].values[0] 
                                   if username in users_df["username"].values else "")
            if st.button("Save Preferences"):
                users_df.loc[users_df["username"] == username, "genres"] = genres
                save_users(users_df)
                st.success("Preferences updated!")

            # Show recommendations
            top_n = st.slider("Number of recommendations", 5, 20, 10)
            recs = get_recommendations(username, top_n=top_n)
            st.write("### Your Recommendations")
            st.table(recs[["title", "score"]])

        else:
            st.error("Invalid username or password")
