from unittest.mock import patch

import requests

from app.qa import NO_INFO_ANSWER, answer_question


@patch("app.qa._synthesize_answer")
@patch("app.qa.search_pdfs")
def test_answer_question_uses_pdf_when_match_is_strong(mock_search_pdfs, mock_synthesize):
    mock_search_pdfs.return_value = [
        {"text": "chunk text", "source": "manual.pdf", "page": 3, "similarity": 0.4}
    ]
    mock_synthesize.return_value = "The answer is from the manual."

    result = answer_question("What does the manual say?")

    assert result["source"] == "pdf"
    assert result["citations"] == ["manual.pdf (p.3)"]
    assert result["answer"] == "The answer is from the manual."
    assert result["confidence"] == 0.4


@patch("app.qa._synthesize_answer")
@patch("app.qa.search_web")
@patch("app.qa.search_pdfs")
def test_answer_question_falls_back_to_web_when_pdf_match_is_weak(
    mock_search_pdfs, mock_search_web, mock_synthesize
):
    mock_search_pdfs.return_value = [{"text": "x", "source": "a.pdf", "page": 1, "similarity": 0.01}]
    mock_search_web.return_value = [
        {"title": "News", "url": "https://news.example.com", "content": "Latest info"}
    ]
    mock_synthesize.return_value = "The latest info is X."

    result = answer_question("What's the latest on X?")

    assert result["source"] == "web"
    assert result["citations"] == ["https://news.example.com"]
    assert result["answer"] == "The latest info is X."


@patch("app.qa.search_web")
@patch("app.qa.search_pdfs")
def test_answer_question_returns_no_info_when_nothing_found(mock_search_pdfs, mock_search_web):
    mock_search_pdfs.return_value = []
    mock_search_web.return_value = []

    result = answer_question("Some obscure question")

    assert result["source"] == "none"
    assert result["answer"] == NO_INFO_ANSWER
    assert result["citations"] == []


@patch("app.qa.search_web")
@patch("app.qa.search_pdfs")
def test_answer_question_survives_web_search_failure(mock_search_pdfs, mock_search_web):
    mock_search_pdfs.return_value = []
    mock_search_web.side_effect = requests.RequestException("timeout")

    result = answer_question("Some question")

    assert result["source"] == "none"
