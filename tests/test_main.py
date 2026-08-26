from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

MOCK_GRAPH_RESULT = {
    "verdict": "Likely Real",
    "confidence": 0.8,
    "explanation": "Looks consistent with known facts.",
    "claims": ["Some claim"],
    "evidence": [{"claim": "Some claim", "fact_checks": []}],
    "sources": [],
    "style_flags": [],
    "style_score": 0.1,
}


def test_analyze_rejects_short_text():
    response = client.post("/analyze", json={"text": "short"})
    assert response.status_code == 422


def test_analyze_rejects_missing_text_and_url():
    response = client.post("/analyze", json={})
    assert response.status_code == 422


@patch("app.main.history.save_analysis")
@patch("app.main.compiled_graph.invoke")
def test_analyze_returns_verdict_for_valid_text(mock_invoke, mock_save):
    mock_invoke.return_value = MOCK_GRAPH_RESULT

    response = client.post("/analyze", json={"text": "This is a long enough claim to analyze."})

    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "Likely Real"
    assert body["confidence"] == 0.8
    assert body["style_score"] == 0.1
    mock_save.assert_called_once()


@patch("app.main.history.save_analysis")
@patch("app.main.fetch_article_text")
@patch("app.main.compiled_graph.invoke")
def test_analyze_fetches_text_from_url(mock_invoke, mock_fetch, mock_save):
    mock_fetch.return_value = "Extracted article text from the page."
    mock_invoke.return_value = MOCK_GRAPH_RESULT

    response = client.post("/analyze", json={"url": "https://example.com/article"})

    assert response.status_code == 200
    mock_fetch.assert_called_once_with("https://example.com/article")
    invoked_state = mock_invoke.call_args[0][0]
    assert invoked_state["text"] == "Extracted article text from the page."


@patch("app.main.fetch_article_text")
def test_analyze_returns_400_when_url_fetch_fails(mock_fetch):
    mock_fetch.side_effect = ValueError("no readable text")

    response = client.post("/analyze", json={"url": "https://example.com/article"})

    assert response.status_code == 400


@patch("app.main.compiled_graph.invoke")
def test_analyze_returns_502_on_pipeline_failure(mock_invoke):
    mock_invoke.side_effect = RuntimeError("boom")

    response = client.post("/analyze", json={"text": "This is a long enough claim to analyze."})

    assert response.status_code == 502


@patch("app.main.add_pdf_document")
@patch("app.main.extract_pdf_chunks")
def test_upload_pdf_returns_chunk_count(mock_extract, mock_add):
    mock_extract.return_value = [{"text": "chunk", "page": 1}]
    mock_add.return_value = 1

    response = client.post(
        "/qa/pdf",
        files={"file": ("manual.pdf", b"%PDF-1.4 fake bytes", "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "manual.pdf"
    assert body["chunks_indexed"] == 1


def test_upload_pdf_rejects_non_pdf():
    response = client.post(
        "/qa/pdf",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


@patch("app.main.extract_pdf_chunks")
def test_upload_pdf_returns_400_on_extraction_failure(mock_extract):
    mock_extract.side_effect = ValueError("no text found")

    response = client.post(
        "/qa/pdf",
        files={"file": ("scanned.pdf", b"%PDF-1.4", "application/pdf")},
    )

    assert response.status_code == 400


@patch("app.main.qa.answer_question")
def test_ask_question_returns_answer(mock_answer):
    mock_answer.return_value = {
        "answer": "It's 42.",
        "source": "pdf",
        "citations": ["doc.pdf (p.1)"],
        "confidence": 0.9,
    }

    response = client.post("/qa/ask", json={"question": "What is the answer?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "It's 42."
    assert body["source"] == "pdf"


def test_ask_question_rejects_short_question():
    response = client.post("/qa/ask", json={"question": "hi"})
    assert response.status_code == 422


@patch("app.main.qa.answer_question")
def test_ask_question_returns_502_on_failure(mock_answer):
    mock_answer.side_effect = RuntimeError("boom")

    response = client.post("/qa/ask", json={"question": "What is the answer?"})

    assert response.status_code == 502


@patch("app.main.history.get_recent")
def test_history_endpoint_returns_entries(mock_get_recent):
    mock_get_recent.return_value = [
        {
            "id": 1,
            "created_at": "2026-08-23T00:00:00+00:00",
            "input_preview": "Some article text",
            "verdict": "Likely Real",
            "confidence": 0.8,
            "result": MOCK_GRAPH_RESULT,
        }
    ]

    response = client.get("/history")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["verdict"] == "Likely Real"
