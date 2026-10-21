import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 Movie Recommender")

movies = pd.read_csv("movies.csv")
movies["tags"] = movies["genres"].str.replace("|", " ", regex=False)
matrix = TfidfVectorizer().fit_transform(movies["tags"])

choice = st.selectbox("Pick a movie you like:", movies["title"])

if st.button("Recommend"):
    i = movies[movies["title"] == choice].index[0]
    scores = cosine_similarity(matrix[i], matrix).flatten()
    top = scores.argsort()[::-1][1:6]
    st.subheader("You may also like:")
    for t in movies["title"].iloc[top]:
        st.write("🎥", t)