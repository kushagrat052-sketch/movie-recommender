import pytest

from recommender import MovieRecommender


@pytest.fixture(scope="module")
def model():
    return MovieRecommender("data/movies_processed.csv")


def test_dataset_loaded(model):
    assert len(model.df) > 4000
    assert "The Dark Knight" in model.titles


def test_details_hide_tags(model):
    movie = model.details("The Dark Knight")
    assert movie["director"] == "Christopher Nolan"
    assert "tags" not in movie.index


def test_poster_paths_are_shipped_with_the_data(model):
    assert "poster_path" in model.df.columns
    filled = model.df["poster_path"].fillna("").astype(str).str.strip().ne("")
    assert filled.mean() > 0.9


def test_recommendations_are_ranked_and_exclude_itself(model):
    recs = model.recommend("The Dark Knight", n=10)

    assert len(recs) == 10
    assert recs["similarity"].is_monotonic_decreasing
    assert "The Dark Knight" not in set(recs["title"])


def test_min_votes_filter(model):
    recs = model.recommend("Inception", n=8, min_votes=5000)
    assert (recs["vote_count"] >= 5000).all()


def test_genre_filter(model):
    recs = model.recommend("Toy Story", n=5, genres=["Animation"])
    assert recs["genres"].str.contains("Animation").all()


def test_genre_counts(model):
    counts = model.genre_counts
    assert counts["Drama"] > 1000
    assert counts["Animation"] > 0


def test_top_rated_genre(model):
    picks = model.top_rated("Animation", n=4)
    assert len(picks) == 4
    assert picks["genres"].str.contains("Animation").all()
    assert picks["vote_count"].min() >= 100


def test_unknown_title_raises(model):
    with pytest.raises(KeyError):
        model.recommend("This Movie Does Not Exist")
