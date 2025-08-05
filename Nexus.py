import os
import json
import time
import random
import joblib
import requests
from PIL import Image
import pandas as pd
import streamlit as st
from io import BytesIO
from dotenv import load_dotenv
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


with open("./static/style.css") as f:
    css = f.read()

st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_dotenv()
API_KEY = os.environ.get("API_KEY")

if "page" not in st.session_state:
    st.session_state.page = "main"

if "movie_name" not in st.session_state:
    st.session_state.movie_name = ""

if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

if "suggestions" not in st.session_state:
    st.session_state.suggestions = []

data = pd.read_csv("../datasets/main_data.csv")
movie_titles = data["movie_title"].tolist()

session = requests.Session()


def create_session():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.2,
        status_forcelist=[429, 500, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


@st.cache_data
def create_similarity():
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data["comb"])
    similarity = cosine_similarity(count_matrix)
    return similarity


@st.cache_resource
def load_model():
    return joblib.load("nlp_model.pkl")


def rcmd(m):
    m = m.lower()
    try:
        similarity = create_similarity()
    except Exception as e:
        st.error(f"Error creating similarity matrix: {e}")
        return []

    if m not in data["movie_title"].unique():
        return "Sorry! The movie you requested is not in our database. Please check the spelling or try with another movie."
    else:
        i = data.loc[data["movie_title"] == m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        lst = lst[1:11]
        l = [data["movie_title"][a] for a in [x[0] for x in lst]]
        return l


def rcmd_with_model(m):
    m = m.lower()
    try:
        # Get the movie's description and vectorize it
        if m not in data["movie_title"].unique():
            return "Sorry! The movie you requested is not in our database. Please check the spelling or try with another movie."

        similarity = create_similarity()  # Existing cosine similarity logic
        i = data.loc[data["movie_title"] == m].index[0]

        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        lst = lst[1:11]  # Get the top 10 similar movies

        recommended_movies = [data["movie_title"][a] for a, _ in lst]

        # Process recommendations with the NLP model
        nlp_model = load_model()
        processed_recommendations = [
            {
                "title": title,
                "predicted_score": nlp_model.predict(
                    [data.loc[data["movie_title"] == title]["description"].values[0]]
                )[0],
            }
            for title in recommended_movies
        ]

        processed_recommendations.sort(key=lambda x: x["predicted_score"], reverse=True)
        return [movie["title"] for movie in processed_recommendations]
    except Exception as e:
        st.error(f"Error in recommendations: {e}")
        return []


def fetch_movie_details(movie_name):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_name}"
    session = create_session()

    try:
        response = session.get(url, timeout=20)  # Increased timeout
        response.raise_for_status()
        data = response.json()
        if data["results"]:
            movie = data["results"][0]
            movie_id = movie["id"]
            movie["original_title"] = movie.get("original_title", movie["title"])
            movie_details_url = (
                f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
            )
            movie_details_response = session.get(movie_details_url, timeout=20)
            movie_details_response.raise_for_status()
            movie_details = movie_details_response.json()

            # Extract the genres
            genres = movie_details.get("genres", [])
            movie["genres"] = [genre["name"] for genre in genres]
            return movie
        else:
            st.warning("No movies found. Try another search.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching movie details: {e}")
        return None


# Function to fetch recommendations from TMDb
@st.cache_data
def fetch_recommendations(movie_id, retries=3, delay=5):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/recommendations?api_key={API_KEY}"
    session = create_session()
    attempt = 0
    while attempt < retries:
        try:
            response = session.get(url, timeout=20)
            response.raise_for_status()
            data = response.json()
            if data["results"]:
                return data["results"][:5]
            else:
                st.warning("No recommendations available.")
                return []
        except requests.exceptions.RequestException as e:
            st.warning(
                f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds..."
            )
            time.sleep(delay)
            attempt += 1
    st.error(
        f"Failed to fetch recommendations for movie ID {movie_id} after {retries} attempts."
    )
    return []


# Function to fetch cast details
def fetch_cast_details(movie_id, retries=3, delay=5):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={API_KEY}"
    session = create_session()
    attempt = 0
    while attempt < retries:
        try:
            response = session.get(url, timeout=20)
            response.raise_for_status()
            data = response.json()
            cast = data.get("cast", [])[:5]
            return [
                {
                    "name": member["name"],
                    "character": member["character"],
                    "profile_path": member.get("profile_path"),
                }
                for member in cast
            ]
        except requests.exceptions.RequestException as e:
            st.warning(
                f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds..."
            )
            time.sleep(delay)
            attempt += 1
    st.error(
        f"Failed to fetch cast details for movie ID {movie_id} after {retries} attempts."
    )
    return []


def fetch_poster(movie_name):
    session = create_session()
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_name}"
    try:
        response = session.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()

        if data["results"]:
            first_movie = data["results"][0]
            title = first_movie.get("title", movie_name)
            poster_path = first_movie.get("poster_path")

            if poster_path:
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                poster_response = session.get(poster_url, timeout=5)
                if poster_response.status_code == 200:
                    return poster_response.content, title
                else:
                    return None, title

            else:
                print(f"No poster path for {movie_name}.")
                return None, title
        else:
            print(f"No results for {movie_name}.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching poster for {movie_name}: {e}")
        return fetch_poster(movie_name), movie_name
    return None, movie_name


def fetch_posters_in_parallel(movie_names):
    results = []
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(fetch_poster, movie_name, API_KEY): movie_name
            for movie_name in movie_names
        }

        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")

    return results


# Function to display movie details
def display_movie_details(movie_details):
    mov_details = {}
    with st.spinner("Loading..."):
        if movie_details.get("poster_path"):
            poster_url = (
                f"https://image.tmdb.org/t/p/w500{movie_details['poster_path']}"
            )
            response = requests.get(poster_url)
            if response.status_code == 200:
                poster_image = Image.open(BytesIO(response.content))

                mov_details["poster"] = poster_image
        else:
            st.text("No poster available")

        mov_details["details"] = [
            movie_details[x]
            for x in [
                "overview",
                "vote_average",
                "release_date",
                "vote_count",
                "original_language",
            ]
        ]
        genres = movie_details.get("genres", [])
        if genres:
            genre_names = [genre for genre in genres]
            mov_details["genre"] = genre_names

        cast_details = fetch_cast_details(movie_details["id"])
        if cast_details:
            mov_details["cast"] = []
            cast_cols = st.columns(len(cast_details))
            for i, cast_member in enumerate(cast_details):
                with cast_cols[i]:
                    if cast_member["profile_path"]:
                        cast_image_url = f"https://image.tmdb.org/t/p/w500{cast_member['profile_path']}"
                        response = requests.get(cast_image_url)
                        if response.status_code == 200:
                            mov_details["cast"].append(
                                [cast_member, Image.open(BytesIO(response.content))]
                            )

    # Move the Overview text above the poster
    st.markdown("## Overview:")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(mov_details["poster"], width=200)
    with col2:
        st.markdown(f"\t {mov_details['details'][0]}")
        st.markdown(f"**Rating:** {mov_details['details'][1]} / 10")
        st.markdown(f"**Release Date:** {mov_details['details'][2]}")
        st.markdown(f"**Vote Count:** {mov_details['details'][3]}")
        st.markdown(f"**Original Language:** {mov_details['details'][4]}")
        st.markdown(f"**Genres:** {', '.join(mov_details['genre'])}")

    # Ensure the "Top Casts" text is displayed before the images
    st.markdown("## Top Casts:")

    cols = st.columns(len(mov_details["cast"]))
    for i, cast in enumerate(mov_details["cast"]):
        with cols[i]:  # Place each image in its respective column
            st.image(cast[1], width=100)  # Resize images as needed
            st.text(cast[0]["name"])


# Function to display recommended movies with posters


def display_recommended_movie(recommendations):
    st.markdown("## Recommendations:")

    row_count = 2
    movies_per_row = 5
    rows = [
        recommendations[i : i + movies_per_row]
        for i in range(0, len(recommendations), movies_per_row)
    ][:row_count]

    recommend_details = []
    with st.spinner("Fetching recommendations..."):

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(fetch_poster, rec): rec for rec in recommendations
            }

            for future in futures:
                movie_name = futures[future]
                try:
                    poster_data, fetched_movie_name = future.result()
                    if poster_data:
                        recommend_details.append(
                            [fetched_movie_name, Image.open(BytesIO(poster_data))]
                        )
                    else:
                        recommend_details.append([movie_name, None])
                except Exception as e:
                    recommend_details.append([movie_name, None])

    # Display each row of movies
    for row_index, row in enumerate(rows):
        cols = st.columns(movies_per_row)  # Create columns for the row
        for i, rec in enumerate(row):
            with cols[i]:
                # Use only the movies in the current row
                movie_name, poster_image = recommend_details[
                    row_index * movies_per_row + i
                ]
                if poster_image:  # Only show the image if it's available
                    st.image(poster_image)
                else:
                    st.text("No Poster Available")  # Display a message if no poster
                unique_key = f"{movie_name}-{row_index}-{i}"  # Generate a unique key
                if st.button(f"{movie_name}", key=unique_key):
                    st.session_state.selected_movie = movie_name
                    print_data(movie_name)
                    st.session_state.movie_name = movie_name
                    with st.spinner("Redirecting..."):
                        st.rerun()


def print_data(movie_name):
    suggestion = []
    movie_details = fetch_movie_details(movie_name)
    get_recs = st.button("Get recommendations")
    if get_recs:
        suggestion = []
        if movie_details:
            st.session_state.selected_movie = movie_details
            display_movie_details(movie_details)
            recommendations = rcmd(st.session_state.selected_movie)
            if isinstance(recommendations, list):
                display_recommended_movie(recommendations)
            else:
                st.warning(recommendations)


# Streamlit app layout
def main_page():
    st.title("Nexus")
    movie_name = st.text_input(
        "Enter a Movie Name", value=st.session_state.get("movie_name", "")
    )

    if movie_name:
        st.session_state.movie_name = movie_name
        st.session_state.page = "recommendations"
        with st.spinner("Redirecting..."):
            st.rerun()

    # Random suggestions if not already stored
    if "suggestions" not in st.session_state or not st.session_state.suggestions:
        st.write("Suggested Movies:")
        suggestion = random.sample(movie_titles, 5)
        st.session_state.suggestions = suggestion

    cols = st.columns(len(st.session_state.suggestions))
    suggestions_data = []

    if st.session_state.suggestions:
        with st.spinner("Fetching suggestions... "):
            # Parallel fetch posters using ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(fetch_poster, suggestion): suggestion
                    for suggestion in st.session_state.suggestions
                }

                for future in futures:
                    movie_name = futures[future]
                    try:
                        poster_data, fetched_movie_name = future.result()
                        if poster_data:
                            suggestions_data.append(
                                [fetched_movie_name, Image.open(BytesIO(poster_data))]
                            )
                        else:
                            suggestions_data.append([movie_name, None])
                    except Exception as e:
                        suggestions_data.append([movie_name, None])

        for i, suggestion in enumerate(suggestions_data):
            with cols[i]:
                if suggestion[1]:  # If the poster is available
                    st.image(suggestion[1], width=200)
                else:
                    st.text("No Poster Available")
                if st.button(f"{suggestion[0]}", key=f"random-{suggestion[0]}"):
                    st.session_state.movie_name = suggestion[0]
                    st.session_state.page = "recommendations"
                    with st.spinner("Redirecting..."):
                        st.rerun()


def recommendations_page():
    if st.button("Home"):
        st.session_state.page = "main"
        st.session_state.movie_name = ""
        st.session_state.suggestions = []
        with st.spinner("Redirecting..."):
            st.rerun()

    movie_name = st.session_state.movie_name
    movie_details = fetch_movie_details(movie_name)

    st.title(f"{movie_details['title']}")
    if movie_details:
        display_movie_details(movie_details)
        recommendations = rcmd(movie_name)
        if isinstance(recommendations, list):
            display_recommended_movie(recommendations)


if st.session_state.page == "main":
    st.session_state.selected_movie = None
    st.session_state.movie_name = ""
    main_page()
elif st.session_state.page == "recommendations":
    st.session_state.suggestions = []
    recommendations_page()
