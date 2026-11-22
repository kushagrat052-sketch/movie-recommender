"""Content-based recommender: TF-IDF + cosine similarity.

Every movie is represented by a TF-IDF vector built from its metadata
(overview, genres, keywords, top cast, director). To recommend, we score
the selected movie against all the others and keep the closest ones.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, csv_path="data/movies_processed.csv"):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        for column in ["overview", "genres", "cast", "director"]:
            if column in self.df:
                self.df[column] = self.df[column].fillna("")

        missing = {"title", "tags"} - set(self.df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=20000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(self.df["tags"].fillna(""))

        self._by_title = {
            title.lower(): i for i, title in enumerate(self.df["title"])
        }
        self._genres = self.df["genres"].fillna("").apply(
            lambda value: {g.strip() for g in value.split(",") if g.strip()}
        )

    @property
    def titles(self):
        return sorted(self.df["title"].tolist())

    @property
    def genres(self):
        found = set()
        for movie_genres in self._genres:
            found |= movie_genres
        return sorted(found)

    def details(self, title):
        """Everything we know about one movie (without the tag blob)."""
        return self.df.loc[self._find(title)].drop(labels=["tags"])

    def recommend(self, title, n=5, min_votes=0, min_rating=0.0, genres=None):
        """Return the n movies most similar to `title`.

        min_votes / min_rating are quality cut-offs, and `genres` keeps only
        movies that share at least one of the given genres.
        """
        idx = self._find(title)
        scores = cosine_similarity(self.matrix[idx], self.matrix).ravel()
        scores[idx] = -1  # never recommend the movie itself

        allowed = self.df["vote_count"].fillna(0) >= min_votes
        allowed &= self.df["vote_average"].fillna(0) >= min_rating
        if genres:
            wanted = set(genres)
            allowed &= self._genres.apply(lambda movie: bool(movie & wanted))

        # Push filtered movies out of the ranking with -inf
        ranked = np.where(allowed.to_numpy(), scores, -np.inf)
        order = np.argsort(ranked)[::-1][:n]
        order = [i for i in order if np.isfinite(ranked[i])]

        result = self.df.iloc[order].copy()
        result["similarity"] = scores[order]
        return result.drop(columns=["tags"]).reset_index(drop=True)

    def _find(self, title):
        try:
            return self._by_title[title.lower()]
        except KeyError:
            raise KeyError(f"Unknown movie title: {title!r}") from None
