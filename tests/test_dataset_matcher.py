from sklearn.feature_extraction.text import TfidfVectorizer

from app.tools import dataset_matcher


def _install_fake_index(monkeypatch, texts, labels, titles):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    monkeypatch.setattr(
        dataset_matcher,
        "_index",
        {"vectorizer": vectorizer, "matrix": matrix, "labels": labels, "titles": titles},
    )
    monkeypatch.setattr(dataset_matcher, "_load_attempted", True)


def test_match_against_dataset_finds_close_match(monkeypatch):
    _install_fake_index(
        monkeypatch,
        texts=[
            "Scientists confirm water boils at one hundred degrees celsius at sea level",
            "Aliens have secretly replaced the moon with a hologram according to leaked memo",
        ],
        labels=["REAL", "FAKE"],
        titles=["Boiling point fact", "Moon hologram hoax"],
    )

    match = dataset_matcher.match_against_dataset(
        "Aliens have secretly replaced the moon with a hologram according to a leaked memo"
    )

    assert match is not None
    assert match["label"] == "FAKE"
    assert match["similarity"] > 0.8


def test_match_against_dataset_low_similarity_for_unrelated_text(monkeypatch):
    _install_fake_index(
        monkeypatch,
        texts=["Scientists confirm water boils at one hundred degrees celsius at sea level"],
        labels=["REAL"],
        titles=["Boiling point fact"],
    )

    match = dataset_matcher.match_against_dataset("My cat knocked a plant off the windowsill today")

    assert match is not None
    assert match["similarity"] < 0.3


def test_match_against_dataset_returns_none_without_index(monkeypatch):
    monkeypatch.setattr(dataset_matcher, "_index", None)
    monkeypatch.setattr(dataset_matcher, "_load_attempted", True)

    assert dataset_matcher.match_against_dataset("Some article text") is None


def test_match_against_dataset_returns_none_for_empty_text(monkeypatch):
    _install_fake_index(
        monkeypatch,
        texts=["Some article text about a real event"],
        labels=["REAL"],
        titles=["Title"],
    )

    assert dataset_matcher.match_against_dataset("   ") is None
