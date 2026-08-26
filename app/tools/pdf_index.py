"""Searchable index over chunks extracted from user-uploaded PDFs.

Unlike app/tools/dataset_matcher.py (which reads a fixed, offline-built
index), this index starts empty and grows as PDFs are uploaded through the
app, refitting its TF-IDF vectorizer over all accumulated chunks each time.
"""

import logging
import threading

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import PDF_INDEX_PATH

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_state = None


def _empty_state() -> dict:
    return {"vectorizer": None, "matrix": None, "chunks": []}


def _load_state() -> dict:
    global _state
    if _state is not None:
        return _state
    with _lock:
        if _state is not None:
            return _state
        try:
            _state = joblib.load(PDF_INDEX_PATH)
        except FileNotFoundError:
            _state = _empty_state()
    return _state


def add_document(filename: str, chunks: list[dict]) -> int:
    """Add a PDF's extracted chunks to the index and persist it. Returns the number added."""
    state = _load_state()
    with _lock:
        for chunk in chunks:
            state["chunks"].append({"text": chunk["text"], "source": filename, "page": chunk["page"]})

        texts = [c["text"] for c in state["chunks"]]
        vectorizer = TfidfVectorizer(stop_words="english")
        state["matrix"] = vectorizer.fit_transform(texts)
        state["vectorizer"] = vectorizer

        joblib.dump(state, PDF_INDEX_PATH)
    return len(chunks)


def search(query: str, top_k: int = 3) -> list[dict]:
    """Return up to top_k {text, source, page, similarity} chunks, best first."""
    state = _load_state()
    if state["vectorizer"] is None or not query.strip():
        return []

    vector = state["vectorizer"].transform([query])
    similarities = (state["matrix"] @ vector.T).toarray().ravel()
    ranked = similarities.argsort()[::-1][:top_k]

    results = []
    for idx in ranked:
        similarity = float(similarities[idx])
        if similarity <= 0:
            continue
        chunk = state["chunks"][idx]
        results.append({**chunk, "similarity": similarity})
    return results
