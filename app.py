import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# TMDb API Key
TMDB_API_KEY = os.getenv('TMDB_API_KEY')

# Set up the Streamlit page
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# Load the dataset
@st.cache_data
def load_data():
    movies = pd.read_csv('tmdb_movies_data.csv')
    movies['combined_features'] = (
        movies['overview'].fillna('') + ' ' +
        movies['keywords'].fillna('') + ' ' +
        movies['tagline'].fillna('')
    )
    return movies

movies = load_data()

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the combined features into TF-IDF vectors
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['combined_features'])

# Calculate cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations based on similarity
def get_recommendations(movie_title, cosine_sim=cosine_sim):
    try:
        idx = movies[movies['original_title'] == movie_title].index[0]
    except IndexError:
        return []

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # Sort by similarity score
    sim_scores = sim_scores[1:11]  # Exclude the first item (itself) and get 10 recommendations
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices]

# Function to fetch movie details from TMDb
def fetch_movie_details(movie_title):
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
    try:
        search_response = requests.get(search_url).json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error during TMDb API request: {e}")
        return None

    if 'results' in search_response and search_response['results']:
        movie_id = search_response['results'][0]['id']
        details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=credits"
        try:
            details_response = requests.get(details_url).json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching movie details from TMDb: {e}")
            return None

        poster_path = details_response.get('poster_path')
        movie_data = {
            "title": details_response.get('original_title'),
            "summary": details_response.get('overview'),
            "poster": f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None,
            "actors": details_response['credits']['cast'][:3] if 'credits' in details_response and 'cast' in details_response['credits'] else []
        }
        return movie_data
    else:
        st.error(f"No results found for '{movie_title}' in TMDb.")
    return None

# Title
st.title("Movie Recommendation System")

# Search box with dropdown list and typing feature
query = st.selectbox("Search for a movie", movies['original_title'].tolist(), format_func=lambda x: x, index=0, key="movie_select")

if query:
    movie_details = fetch_movie_details(query)
    if movie_details:
        # Display movie details
        col1, col2 = st.columns([1, 2])
        with col1:
            if movie_details['poster']:
                st.image(movie_details['poster'], width=300, caption=movie_details['title'])
        with col2:
            st.subheader(movie_details['title'])
            st.write(movie_details['summary'])

            # Display top actors below summary
            if movie_details['actors']:
                st.write("Top Cast:")
                actor_cols = st.columns(len(movie_details['actors']))
                for col, actor in zip(actor_cols, movie_details['actors']):
                    if actor.get('profile_path'):
                        col.image(f"https://image.tmdb.org/t/p/w500{actor.get('profile_path')}", width=100)
                    col.write(actor.get('name'))

        # Get recommendations and display them
        recommendations = get_recommendations(query)
        if not recommendations.empty:
            st.write("### Recommended Movies")
            for _, row in recommendations.iterrows():
                recommended_movie_details = fetch_movie_details(row['original_title'])
                if recommended_movie_details:
                    with st.container():
                        rec_col1, rec_col2 = st.columns([1, 2])
                        with rec_col1:
                            if recommended_movie_details['poster']:
                                st.image(recommended_movie_details['poster'], width=250)  # Increased poster size to 250 pixels
                        with rec_col2:
                            st.subheader(recommended_movie_details['title'])
                            st.write(recommended_movie_details['summary'])

                            # Display top actors for each recommended movie with larger photos
                            if recommended_movie_details['actors']:
                                st.write("Top Cast:")
                                rec_actor_cols = st.columns(len(recommended_movie_details['actors']))
                                for rec_col, rec_actor in zip(rec_actor_cols, recommended_movie_details['actors']):
                                    if rec_actor.get('profile_path'):
                                        rec_col.image(f"https://image.tmdb.org/t/p/w500{rec_actor.get('profile_path')}", width=70)
                                    rec_col.write(rec_actor.get('name'))

                    st.markdown("---")  # Line separator after each recommended movie
