import json
import os
import random
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

from recommender import MovieRecommender

load_dotenv()

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

POSTER_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_MOVIE = "https://www.themoviedb.org/movie"
REPO_URL = "https://github.com/kushagrat052-sketch/movie-recommender"

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1240px; }
    h1 { letter-spacing: -0.02em; }
    h3 { margin-top: 1.6rem; }
    .pill-row { display: flex; flex-wrap: wrap; gap: 0.45rem; margin: 0.5rem 0 1.6rem; }
    .pill {
        padding: 0.25rem 0.7rem;
        border: 1px solid rgba(128, 128, 128, 0.35);
        border-radius: 999px;
        font-size: 0.78rem;
        opacity: 0.85;
    }
    .no-poster {
        aspect-ratio: 2 / 3;
        display: flex; align-items: center; justify-content: center;
        border-radius: 10px;
        background: linear-gradient(140deg, #2c2c36, #17171d);
        color: #9a9aa6;
        font-size: 0.8rem;
        letter-spacing: 0.02em;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the movie index...", ttl=86400)
def load_model():
    return MovieRecommender("data/movies_processed.csv")


@st.cache_data(show_spinner=False, ttl=60)
def load_meta():
    try:
        return json.loads(Path("data/last_updated.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def get_api_key():
    try:
        return st.secrets["TMDB_API_KEY"]
    except Exception:
        return os.getenv("TMDB_API_KEY", "")


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_poster(movie_id):
    key = get_api_key()
    if not key:
        return None

    url = f"https://api.themoviedb.org/3/movie/{int(movie_id)}"
    for attempt in range(2):
        try:
            response = requests.get(
                url, params={"api_key": key}, timeout=8
            )
            response.raise_for_status()
            path = response.json().get("poster_path")
            return f"{POSTER_BASE}{path}" if path else None
        except (requests.RequestException, ValueError, TypeError):
            if attempt == 1:
                return None


def show_poster(movie_id):
    url = fetch_poster(movie_id)
    if url:
        st.image(url, width="stretch")
    else:
        st.markdown(
            '<div class="no-poster">Poster unavailable</div>', unsafe_allow_html=True
        )


def year_of(row):
    year = row.get("release_year")
    return "" if year is None or year != year else f" ({int(year)})"


def shorten(text, length=190):
    text = " ".join(str(text).split())
    if len(text) <= length:
        return text
    return text[:length].rsplit(" ", 1)[0] + "…"


def updated_label(meta):
    raw = meta.get("updated_at")
    if not raw:
        return "recently"
    try:
        moment = datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone()
    except ValueError:
        return "recently"
    return f"{moment:%b} {moment.day}, {moment.year}"


def pick_movie(title):
    st.session_state.movie = title
    st.toast(f"Showing films similar to {title}")


def surprise():
    st.session_state.movie = random.choice(model.titles)


model = load_model()
meta = load_meta()
has_api_key = bool(get_api_key())

st.title("Movie Recommendation System")
st.caption(
    "Content-based recommendations across a growing catalogue. "
    "Pick a film, or start from a category, and get the closest matches."
)
st.markdown(
    f'<div class="pill-row">'
    f'<span class="pill">{len(model.df):,} films</span>'
    f'<span class="pill">{len(model.genres)} categories</span>'
    f'<span class="pill">Updated {updated_label(meta)}</span>'
    f"</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Browse by category
# ---------------------------------------------------------------------------
st.subheader("Browse by category")
st.caption(
    "Best-rated titles in each genre, ranked with a weighted score so "
    "low-vote outliers stay down."
)

counts = model.genre_counts
categories = ["Top rated"] + model.genres
category = st.pills(
    "Category",
    categories,
    default="Top rated",
    key="category",
    label_visibility="collapsed",
)

if category is None:
    st.caption("Select a category to explore its best films.")
else:
    picks = model.top_rated(None if category == "Top rated" else category, n=6)
    if category == "Top rated":
        st.caption("Highest weighted rating across the whole catalogue")
    else:
        st.caption(f"{counts.get(category, 0):,} {category} titles · highest weighted rating")

    if picks.empty:
        st.info("No films matched that category yet. Try another one.")
    else:
        for start in range(0, len(picks), 3):
            columns = st.columns(3, gap="medium")
            for column, (_, row) in zip(columns, picks.iloc[start : start + 3].iterrows()):
                with column, st.container(border=True):
                    show_poster(row["id"])
                    st.markdown(
                        f"**[{row['title']}{year_of(row)}]({TMDB_MOVIE}/{int(row['id'])})**"
                    )
                    st.caption(
                        f"⭐ {row['vote_average']:.1f} · "
                        f"{int(row['vote_count']):,} votes · {row['genres']}"
                    )
                    st.button(
                        "Find similar",
                        key=f"category-{row['id']}",
                        width="stretch",
                        on_click=pick_movie,
                        args=(row["title"],),
                    )

# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Find similar movies")

picker, surprise_col, filter_col = st.columns(
    [5, 1.3, 1], vertical_alignment="bottom"
)
with picker:
    selected = st.selectbox(
        "Pick a movie you like:",
        model.titles,
        index=None,
        placeholder="Start typing a title...",
        key="movie",
        label_visibility="collapsed",
    )
with surprise_col:
    st.button("Surprise me", width="stretch", icon="🎲", on_click=surprise)
with filter_col:
    with st.popover("Filters", width="stretch"):
        n = st.slider("Recommendations", 3, 12, 6)
        min_votes = st.slider("Minimum votes", 0, 2000, 100, step=50)
        min_rating = st.slider("Minimum rating", 0.0, 8.0, 0.0, step=0.5)
        genres = st.multiselect(
            "Genres (any of)", model.genres, placeholder="All genres"
        )

if selected is None:
    st.info("Pick a film above, or use “Find similar” on a category card.")
else:
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
    filter_bits = [f"{len(recs)} matches", f"min {min_votes:,} votes", f"min rating {min_rating:.1f}"]
    if genres:
        filter_bits.append(", ".join(genres))
    st.caption(" · ".join(filter_bits))

    if recs.empty:
        st.warning("Nothing matched those filters. Try lowering the cut-offs.")
    else:
        for start in range(0, len(recs), 3):
            columns = st.columns(3, gap="medium")
            for column, (_, row) in zip(columns, recs.iloc[start : start + 3].iterrows()):
                with column, st.container(border=True):
                    show_poster(row["id"])
                    st.markdown(
                        f"**[{row['title']}{year_of(row)}]({TMDB_MOVIE}/{int(row['id'])})**"
                    )
                    st.progress(
                        float(max(0.0, min(1.0, row["similarity"]))),
                        text=f"Match {row['similarity']:.0%}",
                    )
                    st.caption(
                        f"⭐ {row['vote_average']:.1f} · "
                        f"{int(row['vote_count']):,} votes"
                    )
                    st.write(row["genres"])
                    st.caption(shorten(row["overview"]))

st.divider()
left, right = st.columns([3, 1], vertical_alignment="center")
with left:
    st.caption(
        f"Data updated {updated_label(meta)} · refreshed weekly via GitHub Actions · "
        "Movie data and posters © TMDB"
    )
    if not has_api_key:
        st.caption("Posters are off until a TMDB API key is added to `.streamlit/secrets.toml`.")
with right:
    st.link_button("View on GitHub", REPO_URL, width="stretch")
