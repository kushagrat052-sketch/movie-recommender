"""
Step 1: clean the raw TMDB 5000 dataset and build a compact CSV for the app.

Download the two Kaggle files ("TMDB 5000 Movie Dataset") and put them in
data/raw/ :

    tmdb_5000_movies.csv
    tmdb_5000_credits.csv

Then run:  python prepare_data.py
"""

import ast
from pathlib import Path

import pandas as pd

from features import make_tags

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/movies_processed.csv")


def parse_names(value, limit=None):
    """Parse a TMDB JSON-ish column and keep the `name` fields."""
    try:
        items = ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError):
        return []

    names = [item.get("name", "") for item in items if isinstance(item, dict)]
    names = [name for name in names if name]
    return names[:limit] if limit else names


def find_director(value):
    """The crew column is a list of everyone on the film; we only want the director."""
    try:
        crew = ast.literal_eval(value)
    except (ValueError, SyntaxError, TypeError):
        return ""

    for member in crew:
        if isinstance(member, dict) and member.get("job") == "Director":
            return member.get("name", "")
    return ""


def main():
    movies = pd.read_csv(RAW_DIR / "tmdb_5000_movies.csv")
    credits = pd.read_csv(RAW_DIR / "tmdb_5000_credits.csv")

    df = movies.merge(
        credits[["movie_id", "cast", "crew"]], left_on="id", right_on="movie_id"
    )
    df = df.dropna(subset=["overview", "title"])
    df = df.drop_duplicates(subset="title").reset_index(drop=True)

    df["genres_list"] = df["genres"].apply(parse_names)
    df["keywords_list"] = df["keywords"].apply(parse_names)
    df["cast_list"] = df["cast"].apply(lambda value: parse_names(value, limit=3))
    df["director"] = df["crew"].apply(find_director)

    df["tags"] = df.apply(
        lambda row: make_tags(
            row["overview"],
            row["genres_list"],
            row["keywords_list"],
            row["cast_list"],
            row["director"],
        ),
        axis=1,
    )

    df["release_year"] = pd.to_datetime(
        df["release_date"], errors="coerce"
    ).dt.year.astype("Int64")
    df["genres"] = df["genres_list"].str.join(", ")
    df["cast"] = df["cast_list"].str.join(", ")
    df["vote_average"] = df["vote_average"].round(1)

    out = df[
        [
            "id",
            "title",
            "overview",
            "genres",
            "cast",
            "director",
            "vote_average",
            "vote_count",
            "release_year",
            "tags",
        ]
    ]
    out.to_csv(OUT_PATH, index=False)

    size_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"Saved {len(out)} movies to {OUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
