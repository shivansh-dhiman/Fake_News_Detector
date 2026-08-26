import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FakeNewsDetector/1.0)"}
MAX_CHARS = 8000


def fetch_article_text(url: str) -> str:
    """Download a news article URL and extract its main readable text."""
    response = requests.get(url, headers=HEADERS, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = "\n".join(p for p in paragraphs if len(p) > 40)

    if not text:
        raise ValueError("Could not extract readable article text from this URL")

    return text[:MAX_CHARS]
