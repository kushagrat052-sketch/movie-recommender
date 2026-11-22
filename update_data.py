"""Refresh the catalogue with current titles and vote counts from TMDB.

The GitHub Actions workflow runs this every Monday, but you can also run it
by hand:

    python update_data.py

New releases come from TMDB's popular / now-playing lists. Existing films in
those lists get their vote counts refreshed, and any row still missing a
poster path is filled in. Set TMDB_API_KEY in .env (or in the environment);
without a key the script exits quietly so scheduled runs on a fresh clone
do not fail.

To only backfill poster paths for the existing CSV:

    python update_data.py --posters
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
POSTER_COLUMN = "poster_path"


def api_get(path, **params):
    params["api_key"] = API_KEY
    response = requests.get(f"{API_URL}{path}", params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def ensure_poster_column(df):
    if POSTER_COLUMN not in df:
        df[POSTER_COLUMN] = ""
    df[POSTER_COLUMN] = df[POSTER_COLUMN].fillna("").astype(str)
    return df


def fetch_poster_path(movie_id):
    """One poster path, with a little patience for TMDB rate limits."""
    for attempt in range(3):
        try:
            return api_get(f"/movie/{movie_id}").get("poster_path") or ""
        except requests.HTTPError as error:
            status = getattr(error.response, "status_code", None)
            if status == 404:  # film removed from TMDB
                return ""
            if status == 429:  # rate limited; back off and retry
                time.sleep(2 * (attempt + 1))
                continue
            return ""
        except requests.RequestException:
            time.sleep(1)
    return ""


def fill_missing_posters(df, workers=12):
    """Fetch poster paths for rows that do not have one yet."""
    ensure_poster_column(df)
    missing = df[df[POSTER_COLUMN].str.strip() == ""]
    if missing.empty:
        return df, 0

    print(f"Fetching {len(missing)} missing poster paths...")
    found = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_poster_path, int(movie_id)): int(movie_id)
            for movie_id in missing["id"]
        }
        for done, future in enumerate(as_completed(futures), 1):
            found[futures[future]] = future.result()
            if done % 500 == 0:
                print(f"  {done}/{len(futures)} checked")

    for movie_id, path in found.items():
        if path:
            df.loc[df["id"] == movie_id, POSTER_COLUMN] = path
    return df, sum(1 for path in found.values() if path)


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
        "poster_path": details.get("poster_path") or summary.get("poster_path") or "",
    }


def main():
    if not API_KEY:
        print("TMDB_API_KEY is not set - skipping data refresh.")
        return

    if not DATA_PATH.exists():
        raise SystemExit(f"{DATA_PATH} not found. Run prepare_data.py first.")

    df = ensure_poster_column(pd.read_csv(DATA_PATH))
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
            if summary.get("poster_path"):
                df.loc[mask, POSTER_COLUMN] = summary["poster_path"]
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
    df, posters_filled = fill_missing_posters(df)
    df.to_csv(DATA_PATH, index=False)

    stamp = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "movies": int(len(df)),
        "added": len(new_rows),
        "refreshed": refreshed,
        "posters_added": posters_filled,
    }
    STAMP_PATH.write_text(json.dumps(stamp, indent=2) + "\n")

    print(
        f"Added {len(new_rows)} new films, refreshed {refreshed}, "
        f"filled {posters_filled} posters. Dataset now has {len(df)} movies."
    )


def backfill_posters():
    """Fill poster paths for the current CSV without touching anything else."""
    if not API_KEY:
        print("TMDB_API_KEY is not set - skipping poster backfill.")
        return

    if not DATA_PATH.exists():
        raise SystemExit(f"{DATA_PATH} not found. Run prepare_data.py first.")

    df = ensure_poster_column(pd.read_csv(DATA_PATH))
    df, filled = fill_missing_posters(df)
    df.to_csv(DATA_PATH, index=False)
    print(f"Filled {filled} poster paths. Dataset now has {len(df)} movies.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refresh movie data from TMDB.")
    parser.add_argument(
        "--posters",
        action="store_true",
        help="only backfill missing poster paths, then exit",
    )
    args = parser.parse_args()
    if args.posters:
        backfill_posters()
    else:
        main()
