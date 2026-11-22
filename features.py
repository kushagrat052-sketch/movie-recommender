"""Tag construction shared by the batch preparation and the data updater."""


def tokenise(names):
    """'Sam Worthington' -> 'samworthington' so a person counts as one token."""
    return [
        str(name).lower().replace(" ", "").replace("-", "")
        for name in names
        if name
    ]


def make_tags(overview, genres, keywords, cast, director):
    """Build the text blob that TF-IDF turns into a vector.

    Genres and director are repeated so they carry a little more weight
    than the overview, which is long and tends to dominate.
    """
    return " ".join(
        [
            (overview or "").lower(),
            " ".join(tokenise(genres) * 2),
            " ".join(tokenise(keywords)),
            " ".join(tokenise(cast)),
            " ".join(tokenise([director]) * 2),
        ]
    )
