import pytest
from pydantic import ValidationError

from app.schemas import AnalyzeRequest


def test_analyze_request_accepts_valid_text():
    request = AnalyzeRequest(text="This is a valid claim to analyze.")
    assert request.text == "This is a valid claim to analyze."


def test_analyze_request_rejects_too_short_text():
    with pytest.raises(ValidationError):
        AnalyzeRequest(text="short")


def test_analyze_request_accepts_url_without_text():
    request = AnalyzeRequest(url="https://example.com/article")
    assert request.url == "https://example.com/article"


def test_analyze_request_rejects_neither_text_nor_url():
    with pytest.raises(ValidationError):
        AnalyzeRequest()
