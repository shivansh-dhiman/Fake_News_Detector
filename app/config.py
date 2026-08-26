import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable '{name}'. Set it in your .env file."
        )
    return value


GROQ_API_KEY = _require("GROQ_API_KEY")
GOOGLE_FACT_CHECK_API_KEY = _require("GOOGLE_FACT_CHECK_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

RATE_LIMIT = os.environ.get("RATE_LIMIT", "10/minute")

DATASET_INDEX_PATH = os.environ.get("DATASET_INDEX_PATH", str(BASE_DIR / "data" / "dataset_index.joblib"))
DATASET_MATCH_THRESHOLD = float(os.environ.get("DATASET_MATCH_THRESHOLD", "0.6"))
# Short claims are unreliable to match via TF-IDF: a single rare shared word (e.g. a name)
# can push cosine similarity above threshold against a topically-related but substantively
# different article. Only attempt a dataset short-circuit for long, article-length input.
DATASET_MATCH_MIN_CHARS = int(os.environ.get("DATASET_MATCH_MIN_CHARS", "300"))

# Q&A mode: PDF-first retrieval with a live web search fallback.
PDF_INDEX_PATH = os.environ.get("PDF_INDEX_PATH", str(BASE_DIR / "data" / "pdf_index.joblib"))
PDF_MATCH_THRESHOLD = float(os.environ.get("PDF_MATCH_THRESHOLD", "0.15"))
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
