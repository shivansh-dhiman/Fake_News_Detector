from unittest.mock import MagicMock, patch

import pytest

from app.tools.pdf_extractor import extract_pdf_chunks


@patch("app.tools.pdf_extractor.PdfReader")
def test_extract_pdf_chunks_splits_by_page(mock_reader_cls):
    page1 = MagicMock()
    page1.extract_text.return_value = "Page one text here."
    page2 = MagicMock()
    page2.extract_text.return_value = "Page two text here."
    mock_reader_cls.return_value.pages = [page1, page2]

    chunks = extract_pdf_chunks(b"fake-pdf-bytes")

    assert chunks == [
        {"text": "Page one text here.", "page": 1},
        {"text": "Page two text here.", "page": 2},
    ]


@patch("app.tools.pdf_extractor.PdfReader")
def test_extract_pdf_chunks_splits_long_page_into_multiple_chunks(mock_reader_cls):
    page = MagicMock()
    page.extract_text.return_value = "A" * 3000
    mock_reader_cls.return_value.pages = [page]

    chunks = extract_pdf_chunks(b"fake-pdf-bytes", chunk_chars=1000)

    assert len(chunks) == 3
    assert all(c["page"] == 1 for c in chunks)


@patch("app.tools.pdf_extractor.PdfReader")
def test_extract_pdf_chunks_skips_blank_pages(mock_reader_cls):
    blank_page = MagicMock()
    blank_page.extract_text.return_value = "   "
    text_page = MagicMock()
    text_page.extract_text.return_value = "Real content here."
    mock_reader_cls.return_value.pages = [blank_page, text_page]

    chunks = extract_pdf_chunks(b"fake-pdf-bytes")

    assert chunks == [{"text": "Real content here.", "page": 2}]


@patch("app.tools.pdf_extractor.PdfReader")
def test_extract_pdf_chunks_raises_when_no_text(mock_reader_cls):
    page = MagicMock()
    page.extract_text.return_value = ""
    mock_reader_cls.return_value.pages = [page]

    with pytest.raises(ValueError):
        extract_pdf_chunks(b"fake-pdf-bytes")
