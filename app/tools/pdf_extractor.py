"""Extract text chunks (with page numbers) from an uploaded PDF file."""

from io import BytesIO

from pypdf import PdfReader

CHUNK_CHARS = 1500


def extract_pdf_chunks(file_bytes: bytes, chunk_chars: int = CHUNK_CHARS) -> list[dict]:
    """Return a list of {"text": str, "page": int} chunks, one or more per page."""
    reader = PdfReader(BytesIO(file_bytes))
    chunks = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        for i in range(0, len(text), chunk_chars):
            piece = text[i : i + chunk_chars].strip()
            if piece:
                chunks.append({"text": piece, "page": page_num})

    if not chunks:
        raise ValueError("Could not extract any text from this PDF (it may be scanned/image-only)")
    return chunks
