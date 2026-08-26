from unittest.mock import MagicMock, patch

import pytest

from app.tools.article_fetcher import fetch_article_text

SAMPLE_HTML = """
<html>
  <head><script>trackUser()</script></head>
  <body>
    <nav>Home | About</nav>
    <article>
      <p>This is the first real paragraph of the article and it is long enough to count.</p>
      <p>This is the second real paragraph of the article and it is also long enough.</p>
    </article>
    <footer>Copyright 2026</footer>
  </body>
</html>
"""


@patch("app.tools.article_fetcher.requests.get")
def test_fetch_article_text_extracts_paragraphs(mock_get):
    mock_response = MagicMock()
    mock_response.text = SAMPLE_HTML
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    text = fetch_article_text("https://example.com/article")

    assert "first real paragraph" in text
    assert "second real paragraph" in text
    assert "trackUser" not in text
    assert "Home | About" not in text


@patch("app.tools.article_fetcher.requests.get")
def test_fetch_article_text_raises_when_no_paragraphs(mock_get):
    mock_response = MagicMock()
    mock_response.text = "<html><body><div>short</div></body></html>"
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    with pytest.raises(ValueError):
        fetch_article_text("https://example.com/empty")
