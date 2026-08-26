"""Benchmark the fake-news detection pipeline against a labeled CSV dataset.

Usage:
    .\\venv\\Scripts\\python.exe scripts\\evaluate.py --dataset data\\sample_labeled_news.csv
    .\\venv\\Scripts\\python.exe scripts\\evaluate.py --dataset data\\my_big_dataset.csv --limit 50

The CSV must have columns: text,label   where label is REAL or FAKE.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.graph import compiled_graph  # noqa: E402

INITIAL_STATE = {
    "claims": [],
    "evidence": [],
    "web_context": [],
    "verdict": "",
    "confidence": 0.0,
    "explanation": "",
    "sources": [],
    "dataset_match": False,
    "dataset_label": "",
    "dataset_similarity": 0.0,
}


def load_dataset(path: Path, limit: int | None) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    return rows


def predicted_label(verdict: str) -> str:
    verdict_lower = verdict.lower()
    if "fake" in verdict_lower:
        return "FAKE"
    if "real" in verdict_lower:
        return "REAL"
    return "UNCERTAIN"


def compute_metrics(rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(1 for r in rows if r["predicted"] == r["label"])
    uncertain = sum(1 for r in rows if r["predicted"] == "UNCERTAIN")

    def prf(positive_class: str) -> dict:
        tp = sum(1 for r in rows if r["predicted"] == positive_class and r["label"] == positive_class)
        fp = sum(1 for r in rows if r["predicted"] == positive_class and r["label"] != positive_class)
        fn = sum(1 for r in rows if r["predicted"] != positive_class and r["label"] == positive_class)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3)}

    return {
        "total_examples": total,
        "accuracy": round(correct / total, 3) if total else 0.0,
        "uncertain_rate": round(uncertain / total, 3) if total else 0.0,
        "fake_class_metrics": prf("FAKE"),
        "real_class_metrics": prf("REAL"),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark the pipeline against a labeled dataset.")
    parser.add_argument("--dataset", required=True, help="Path to a CSV with columns: text,label")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N rows")
    parser.add_argument("--output", default="evaluation_report.json", help="Where to write the JSON report")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    rows = load_dataset(dataset_path, args.limit)
    print(f"Evaluating {len(rows)} examples from {dataset_path}...\n")

    results = []
    for i, row in enumerate(rows, start=1):
        text, label = row["text"], row["label"].strip().upper()
        start = time.perf_counter()
        try:
            state = compiled_graph.invoke({"text": text, **INITIAL_STATE})
            verdict = state["verdict"]
        except Exception as exc:
            verdict = f"ERROR: {exc}"
        elapsed = time.perf_counter() - start

        pred = predicted_label(verdict)
        match = "OK" if pred == label else "MISS"
        print(f"[{i}/{len(rows)}] {match:4} truth={label:9} pred={pred:9} ({elapsed:.1f}s)  {text[:70]}")

        results.append({"text": text, "label": label, "verdict": verdict, "predicted": pred})

    metrics = compute_metrics(results)

    print("\n--- Summary ---")
    print(f"Accuracy:        {metrics['accuracy'] * 100:.1f}%")
    print(f"Uncertain rate:  {metrics['uncertain_rate'] * 100:.1f}%")
    print(f"FAKE class:      precision={metrics['fake_class_metrics']['precision']} "
          f"recall={metrics['fake_class_metrics']['recall']} f1={metrics['fake_class_metrics']['f1']}")
    print(f"REAL class:      precision={metrics['real_class_metrics']['precision']} "
          f"recall={metrics['real_class_metrics']['recall']} f1={metrics['real_class_metrics']['f1']}")

    report = {"metrics": metrics, "results": results}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFull report written to {args.output}")


if __name__ == "__main__":
    main()
