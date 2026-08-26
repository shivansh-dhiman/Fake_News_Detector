"""Look up input text against the pre-built TF-IDF index of data/Fake.csv
and data/True.csv, so exact/near-duplicate known articles can be answered
from the labeled dataset instead of the LLM + web fact-check pipeline.
"""

import logging
import threading
from typing import TypedDict

import joblib

from app.config import DATASET_INDEX_PATH

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_index = None
_load_attempted = False


class DatasetMatch(TypedDict):
    label: str
    similarity: float
    title: str


def _load_index():
    global _index, _load_attempted
    if _load_attempted:
        return _index
    with _lock:
        if _load_attempted:
            return _index
        _load_attempted = True
        try:
            _index = joblib.load(DATASET_INDEX_PATH)
        except FileNotFoundError:
            logger.warning(
                "Dataset index not found at %s; dataset lookup is disabled until it's built "
                "(run scripts/build_dataset_index.py)",
                DATASET_INDEX_PATH,
            )
            _index = None
    return _index


def match_against_dataset(text: str) -> DatasetMatch | None:
    """Return the closest labeled dataset article by cosine similarity, or None
    if the index isn't built or the text is empty."""
    index = _load_index()
    if index is None or not text.strip():
        return None

    vector = index["vectorizer"].transform([text])
    # TF-IDF vectors from scikit-learn are L2-normalized, so the dot product
    # of the query against the (also normalized) document matrix is exactly
    # the cosine similarity.
    similarities = (index["matrix"] @ vector.T).toarray().ravel()
    best_idx = similarities.argmax()
    best_similarity = float(similarities[best_idx])

    return {
        "label": index["labels"][best_idx],
        "similarity": best_similarity,
        "title": index["titles"][best_idx],
    }
