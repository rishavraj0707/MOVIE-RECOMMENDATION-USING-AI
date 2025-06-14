
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import pyperclip

def load_and_prepare_data():
    try:
        movies = pd.read_csv("movies.csv").head(1000)
    except FileNotFoundError:
        st.error("Error: 'movies.csv' file not found.")
        st.stop()
    
    required_columns = ["title", "genres"]
    if not all(col in movies.columns for col in required_columns):
        st.error("Missing required columns in dataset.")
        st.stop()

    if "overview" not in movies.columns:
        movies["overview"] = ""
    
    # Clean missing rows and combine content
    movies.dropna(subset=["title", "genres"], inplace=True)
    movies["genres"] = movies["genres"].str.replace("|", " ", regex=False).str.lower()
    movies["overview"] = movies["overview"].fillna("").str.lower()
    
    # New: Better content representation
    movies["content"] = movies["title"].str.lower() + " " + movies["genres"] + " " + movies["overview"]
    
    return movies

def initialize_similarity_matrix(movies):
    tfidf = TfidfVectorizer(stop_words="english", max_df=0.8)
    tfidf_matrix = tfidf.fit_transform(movies["content"])
    return cosine_similarity(tfidf_matrix)

def recommend_movies(movies, cosine_sim, movie_indices, title, preferred_genre=None, top_n=5):
    idx = movie_indices.get(title.lower())
    if idx is None:
        return ["Movie not found in database."]
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i, score in sim_scores[1:]:
        if preferred_genre:
            movie_genres = movies.iloc[i]["genres"]
            if preferred_genre.lower() not in movie_genres:
                continue
        recommendations.append((movies.iloc[i]["title"], score))
        if len(recommendations) == top_n:
            break

    if not recommendations:
        return ["No similar movies found."]
    
    return [title for title, score in recommendations]

def main():
    st.set_page_config("Movie Recommender Pro", layout="centered")
    st.title("🎬 Movie Recommendation Engine")
    
    movies = load_and_prepare_data()
    cosine_sim = initialize_similarity_matrix(movies)
    movie_indices = pd.Series(movies.index, index=movies["title"].str.lower()).drop_duplicates()
    all_genres = sorted(set(g for genres in movies["genres"].str.split() for g in genres))

    col1, col2 = st.columns(2)
    with col1:
        movie_input = st.selectbox("🎞 Choose a movie:", movies["title"].sort_values().unique())
    with col2:
        genre_input = st.selectbox("🎯 Optional genre filter:", ["Any"] + all_genres)

    if st.button("🔍 Get Recommendations"):
        selected_genre = None if genre_input == "Any" else genre_input
        results = recommend_movies(movies, cosine_sim, movie_indices, movie_input, selected_genre)
        st.subheader("📽 Recommendations:")
        for i, movie in enumerate(results, 1):
            st.markdown(f"{i}. **{movie}**")
        
        if results and "not found" not in results[0].lower():
            if st.button("📋 Copy Top Recommendation"):
                pyperclip.copy(results[0])
                st.success("Copied to clipboard!")

if __name__ == "__main__":
    main()
