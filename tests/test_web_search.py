from unittest.mock import MagicMock, patch

import pytest
import requests

from app.tools import web_search


def test_search_web_returns_empty_without_api_key(monkeypatch):
    monkeypatch.setattr(web_search, "TAVILY_API_KEY", None)

    assert web_search.search_web("some query") == []


@patch("app.tools.web_search.TAVILY_API_KEY", "fake-key")
@patch("app.tools.web_search.requests.post")
def test_search_web_parses_results(mock_post):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "results": [{"title": "Example title", "url": "https://example.com", "content": "Some snippet"}]
    }
    mock_post.return_value = mock_response

    results = web_search.search_web("some query")

    assert results == [{"title": "Example title", "url": "https://example.com", "content": "Some snippet"}]


@patch("app.tools.web_search.TAVILY_API_KEY", "fake-key")
@patch("app.tools.web_search.requests.post")
def test_search_web_raises_on_http_error(mock_post):
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("500 error")
    mock_post.return_value = mock_response

    with pytest.raises(requests.HTTPError):
        web_search.search_web("some query")
