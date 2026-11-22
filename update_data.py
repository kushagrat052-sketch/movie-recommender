"""Refresh the catalogue with current titles and vote counts from TMDB.

The GitHub Actions workflow runs this every Monday, but you can also run it
by hand:

    python update_data.py

New releases come from TMDB's popular / now-playing lists. Existing films in
those lists get their vote counts refreshed. Set TMDB_API_KEY in .env (or in
the environment); without a key the script exits quietly so scheduled runs
on a fresh clone do not fail.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from features import make_tags

load_dotenv()

API_URL = "https://api.themoviedb.org/3"
DATA_PATH = Path("data/movies_processed.csv")
STAMP_PATH = Path("data/last_updated.json")

API_KEY = os.getenv("TMDB_API_KEY", "")
PAGES = int(os.getenv("UPDATE_PAGES", "5"))


def api_get(path, **params):
    params["api_key"] = API_KEY
    response = requests.get(f"{API_URL}{path}", params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def collect_candidates():
    """Popular and currently-playing films, keyed by TMDB id."""
    candidates = {}
    for list_name, pages in (("popular", PAGES), ("now_playing", 2)):
        for page in range(1, pages + 1):
            try:
                payload = api_get(f"/movie/{list_name}", page=page)
            except requests.RequestException as error:
                print(f"  {list_name} page {page} failed: {error}")
                continue
            for movie in payload.get("results", []):
                candidates[movie["id"]] = movie
    return candidates


def parse_year(release_date):
    if not release_date:
        return None
    try:
        return int(str(release_date)[:4])
    except ValueError:
        return None


def build_row(movie_id, summary):
    """Fetch full metadata for one film and turn it into a dataset row."""
    details = api_get(f"/movie/{movie_id}", append_to_response="credits,keywords")
    credits = details.get("credits") or {}

    cast = [person.get("name", "") for person in credits.get("cast", [])[:3]]
    director = next(
        (
            person.get("name", "")
            for person in credits.get("crew", [])
            if person.get("job") == "Director"
        ),
        "",
    )
    genres = [genre.get("name", "") for genre in details.get("genres", [])]
    keywords = [
        keyword.get("name", "")
        for keyword in (details.get("keywords") or {}).get("keywords", [])
    ]
    overview = details.get("overview") or summary.get("overview") or ""
    title = details.get("title") or summary.get("title")

    if not title:
        return None

    return {
        "id": int(movie_id),
        "title": title,
        "overview": overview,
        "genres": ", ".join(genres),
        "cast": ", ".join(cast),
        "director": director,
        "vote_average": round(float(details.get("vote_average") or 0), 1),
        "vote_count": int(details.get("vote_count") or 0),
        "release_year": parse_year(details.get("release_date")),
        "tags": make_tags(overview, genres, keywords, cast, director),
    }


def main():
    if not API_KEY:
        print("TMDB_API_KEY is not set - skipping data refresh.")
        return

    if not DATA_PATH.exists():
        raise SystemExit(f"{DATA_PATH} not found. Run prepare_data.py first.")

    df = pd.read_csv(DATA_PATH)
    known_ids = {int(movie_id) for movie_id in df["id"]}
    known_titles = {title.lower() for title in df["title"]}

    candidates = collect_candidates()
    print(f"Found {len(candidates)} current films to check")

    new_rows = []
    refreshed = 0

    for movie_id, summary in candidates.items():
        if movie_id in known_ids:
            mask = df["id"] == movie_id
            df.loc[mask, "vote_average"] = round(
                float(summary.get("vote_average") or 0), 1
            )
            df.loc[mask, "vote_count"] = int(summary.get("vote_count") or 0)
            refreshed += 1
            continue

        title = (summary.get("title") or "").strip()
        if not title or title.lower() in known_titles:
            continue

        try:
            row = build_row(movie_id, summary)
        except requests.RequestException as error:
            print(f"  skipping {title}: {error}")
            continue

        if row is None:
            continue

        new_rows.append(row)
        known_titles.add(title.lower())
        time.sleep(0.05)

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df = df.drop_duplicates(subset="title").reset_index(drop=True)
    df.to_csv(DATA_PATH, index=False)

    stamp = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "movies": int(len(df)),
        "added": len(new_rows),
        "refreshed": refreshed,
    }
    STAMP_PATH.write_text(json.dumps(stamp, indent=2) + "\n")

    print(
        f"Added {len(new_rows)} new films, refreshed {refreshed}. "
        f"Dataset now has {len(df)} movies."
    )


if __name__ == "__main__":
    main()
