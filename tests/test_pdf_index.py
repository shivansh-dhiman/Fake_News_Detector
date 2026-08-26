import pytest

from app.tools import pdf_index


@pytest.fixture(autouse=True)
def reset_pdf_index(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_index, "PDF_INDEX_PATH", str(tmp_path / "pdf_index.joblib"))
    monkeypatch.setattr(pdf_index, "_state", None)
    yield


def test_search_returns_empty_before_any_document_added():
    assert pdf_index.search("anything") == []


def test_add_document_then_search_finds_match():
    pdf_index.add_document(
        "manual.pdf",
        [
            {"text": "The refund policy allows returns within 30 days of purchase.", "page": 1},
            {"text": "Shipping typically takes five to seven business days.", "page": 2},
        ],
    )

    results = pdf_index.search("What is the refund policy?")

    assert results
    assert results[0]["source"] == "manual.pdf"
    assert results[0]["page"] == 1


def test_add_document_accumulates_across_calls():
    pdf_index.add_document("a.pdf", [{"text": "Alpha document content about cats.", "page": 1}])
    pdf_index.add_document("b.pdf", [{"text": "Beta document content about dogs.", "page": 1}])

    results = pdf_index.search("Tell me about dogs")

    assert results[0]["source"] == "b.pdf"


def test_search_returns_no_match_for_unrelated_query():
    pdf_index.add_document("a.pdf", [{"text": "Quarterly revenue grew by ten percent.", "page": 1}])

    results = pdf_index.search("banana smoothie recipe")

    assert results == []


def test_add_document_returns_number_of_chunks_added():
    count = pdf_index.add_document(
        "a.pdf",
        [
            {"text": "First chunk of content.", "page": 1},
            {"text": "Second chunk of content.", "page": 1},
        ],
    )

    assert count == 2
