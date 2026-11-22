import os
import random

import requests
import streamlit as st
from dotenv import load_dotenv

from recommender import MovieRecommender

load_dotenv()

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

POSTER_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_MOVIE = "https://www.themoviedb.org/movie"

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; max-width: 1150px; }
    .no-poster {
        aspect-ratio: 2 / 3;
        display: flex; align-items: center; justify-content: center;
        border-radius: 10px;
        background: linear-gradient(140deg, #2c2c36, #17171d);
        color: #8d8d99;
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the movie index...")
def load_model():
    return MovieRecommender("data/movies_processed.csv")


def get_api_key():
    try:
        return st.secrets["TMDB_API_KEY"]
    except Exception:
        return os.getenv("TMDB_API_KEY", "")


@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    key = get_api_key()
    if not key:
        return None

    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(movie_id)}",
            params={"api_key": key},
            timeout=5,
        )
        response.raise_for_status()
        path = response.json().get("poster_path")
        return f"{POSTER_BASE}{path}" if path else None
    except (requests.RequestException, ValueError, TypeError):
        return None


def show_poster(movie_id):
    url = fetch_poster(movie_id)
    if url:
        st.image(url, width="stretch")
    else:
        st.markdown('<div class="no-poster">No poster</div>', unsafe_allow_html=True)


def year_of(row):
    year = row.get("release_year")
    return "" if year is None or year != year else f" ({int(year)})"


def shorten(text, length=190):
    text = " ".join(str(text).split())
    if len(text) <= length:
        return text
    return text[:length].rsplit(" ", 1)[0] + "…"


model = load_model()
has_api_key = bool(get_api_key())

st.title("🎬 Movie Recommendation System")
st.caption(
    "Content-based filtering over the TMDB 5000 dataset. "
    "Pick a film and get the ones that share its story, people and mood."
)

with st.sidebar:
    st.header("Refine")
    n = st.slider("Recommendations", 3, 12, 6)
    min_votes = st.slider("Minimum votes", 0, 2000, 100, step=50)
    min_rating = st.slider("Minimum rating", 0.0, 8.0, 0.0, step=0.5)
    genres = st.multiselect("Genres (any of)", model.genres, placeholder="All genres")

    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. Overview, genres, keywords, cast and director are merged into one text blob.\n"
        "2. TF-IDF turns each blob into a weighted vector.\n"
        "3. Cosine similarity ranks the closest movies."
    )

    if not has_api_key:
        st.info(
            "No TMDB API key found, so posters are hidden. "
            "Add `TMDB_API_KEY` to `.streamlit/secrets.toml` to enable them."
        )

def surprise():
    st.session_state.movie = random.choice(model.titles)


picker, surprise_col = st.columns([4, 1], vertical_alignment="bottom")
with picker:
    selected = st.selectbox(
        "Pick a movie you like:",
        model.titles,
        index=None,
        placeholder="Start typing a title...",
        key="movie",
    )
with surprise_col:
    st.button("Surprise me", width="stretch", icon="🎲", on_click=surprise)

if selected is None:
    st.info("Choose a movie above to see similar titles.")
    st.stop()

info = model.details(selected)

with st.container(border=True):
    poster_col, text_col = st.columns([1, 3], gap="large")
    with poster_col:
        show_poster(info["id"])
    with text_col:
        st.markdown(f"### {info['title']}{year_of(info)}")
        st.markdown(
            f"⭐ **{info['vote_average']:.1f}** &nbsp;·&nbsp; "
            f"{int(info['vote_count']):,} votes &nbsp;·&nbsp; "
            f"**Directed by** {info['director'] or 'unknown'}"
        )
        st.markdown(f"**Genres:** {info['genres']}")
        st.markdown(f"**Cast:** {info['cast']}")
        st.write(info["overview"])

recs = model.recommend(
    selected,
    n=n,
    min_votes=min_votes,
    min_rating=min_rating,
    genres=genres,
)

st.subheader("You might also like")

if recs.empty:
    st.warning("Nothing matched those filters. Try lowering the vote or rating cut-off.")
    st.stop()

for start in range(0, len(recs), 3):
    columns = st.columns(3, gap="medium")
    for column, (_, row) in zip(columns, recs.iloc[start : start + 3].iterrows()):
        with column, st.container(border=True):
            show_poster(row["id"])
            st.markdown(f"**[{row['title']}{year_of(row)}]({TMDB_MOVIE}/{int(row['id'])})**")
            st.progress(
                float(max(0.0, min(1.0, row["similarity"]))),
                text=f"Match {row['similarity']:.0%}",
            )
            st.caption(
                f"⭐ {row['vote_average']:.1f} · {int(row['vote_count']):,} votes"
            )
            st.write(row["genres"])
            st.caption(shorten(row["overview"]))

st.divider()
st.caption(
    "Data: TMDB 5000 Movie Dataset · Movie info and posters © TMDB · "
    "Built with Streamlit, scikit-learn and pandas."
)
