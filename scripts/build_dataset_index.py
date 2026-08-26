"""Build a TF-IDF similarity index over data/Fake.csv and data/True.csv.

This is an offline preprocessing step: it reads the two labeled datasets,
fits a TF-IDF vectorizer over their article text, and persists the
vectorizer + document matrix + labels/titles to disk so the running app
can do fast nearest-neighbor lookups without re-reading the ~116MB CSVs
or refitting the vectorizer on every request.

Usage:
    .\\venv\\Scripts\\python.exe scripts\\build_dataset_index.py
"""

import csv
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MAX_DOC_CHARS = 20000

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def load_rows(path: Path, label: str) -> tuple[list[str], list[str], list[str]]:
    texts, titles, labels = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = (row.get("text") or "").strip()
            if not text:
                continue
            texts.append(text[:MAX_DOC_CHARS])
            titles.append((row.get("title") or "").strip())
            labels.append(label)
    return texts, titles, labels


def main():
    fake_texts, fake_titles, fake_labels = load_rows(DATA_DIR / "Fake.csv", "FAKE")
    true_texts, true_titles, true_labels = load_rows(DATA_DIR / "True.csv", "REAL")

    texts = fake_texts + true_texts
    titles = fake_titles + true_titles
    labels = fake_labels + true_labels
    print(f"Loaded {len(fake_texts)} FAKE + {len(true_texts)} REAL = {len(texts)} documents")

    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    print(f"Fitted TF-IDF matrix: {matrix.shape}")

    index_path = DATA_DIR / "dataset_index.joblib"
    joblib.dump(
        {"vectorizer": vectorizer, "matrix": matrix, "labels": labels, "titles": titles},
        index_path,
    )
    print(f"Wrote index to {index_path}")


if __name__ == "__main__":
    main()
