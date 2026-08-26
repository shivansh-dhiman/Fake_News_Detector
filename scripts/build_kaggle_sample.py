"""Build a stratified random sample from the Kaggle Fake.csv/True.csv split
into the text,label CSV format expected by scripts/evaluate.py.

Usage:
    .\\venv\\Scripts\\python.exe scripts\\build_kaggle_sample.py --per-class 20 --seed 42
"""

import argparse
import csv
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_rows(path: Path, label: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        text = (r.get("text") or "").strip()
        if len(text) >= 200:  # skip empty/near-empty rows
            out.append({"text": text, "label": label})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=20, help="How many rows to sample per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-chars", type=int, default=4000, help="Truncate article text to this length")
    parser.add_argument("--output", default=str(DATA_DIR / "kaggle_sample.csv"))
    args = parser.parse_args()

    random.seed(args.seed)

    fake_rows = load_rows(DATA_DIR / "Fake.csv", "FAKE")
    true_rows = load_rows(DATA_DIR / "True.csv", "REAL")

    fake_sample = random.sample(fake_rows, args.per_class)
    true_sample = random.sample(true_rows, args.per_class)

    combined = fake_sample + true_sample
    random.shuffle(combined)

    for row in combined:
        if len(row["text"]) > args.max_chars:
            row["text"] = row["text"][: args.max_chars]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(combined)

    print(f"Wrote {len(combined)} rows ({args.per_class} FAKE + {args.per_class} REAL) to {args.output}")


if __name__ == "__main__":
    main()
