import logging
import time
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app import history, qa
from app.agents.graph import compiled_graph
from app.config import RATE_LIMIT
from app.logging_config import configure_logging
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    HistoryEntry,
    PDFUploadResponse,
    QAResponse,
    QuestionRequest,
)
from app.tools.article_fetcher import fetch_article_text
from app.tools.pdf_extractor import extract_pdf_chunks
from app.tools.pdf_index import add_document as add_pdf_document

configure_logging()
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Fake News Detection System")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s -> %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/analyze", response_model=AnalyzeResponse)
@limiter.limit(RATE_LIMIT)
def analyze(request: Request, body: AnalyzeRequest):
    if body.url:
        try:
            text = fetch_article_text(body.url)
        except (requests.RequestException, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Could not fetch article: {exc}") from exc
    else:
        text = body.text

    try:
        result = compiled_graph.invoke(
            {
                "text": text,
                "claims": [],
                "evidence": [],
                "web_context": [],
                "style_flags": [],
                "style_score": 0.0,
                "verdict": "",
                "confidence": 0.0,
                "explanation": "",
                "sources": [],
                "dataset_match": False,
                "dataset_label": "",
                "dataset_similarity": 0.0,
            }
        )
    except Exception as exc:
        logger.exception("Analysis pipeline failed")
        raise HTTPException(status_code=502, detail=f"Analysis pipeline failed: {exc}") from exc

    response = AnalyzeResponse(
        verdict=result["verdict"],
        confidence=result["confidence"],
        explanation=result["explanation"],
        claims=result["claims"],
        evidence=result["evidence"],
        sources=result["sources"],
        style_flags=result["style_flags"],
        style_score=result["style_score"],
        dataset_match=result.get("dataset_match", False),
        dataset_label=result.get("dataset_label") or None,
        dataset_similarity=result.get("dataset_similarity", 0.0),
    )

    history.save_analysis(body.url or text, response.model_dump())

    return response


@app.get("/history", response_model=list[HistoryEntry])
def get_history(limit: int = 20):
    return history.get_recent(limit=min(limit, 100))


@app.post("/qa/pdf", response_model=PDFUploadResponse)
@limiter.limit(RATE_LIMIT)
async def upload_pdf(request: Request, file: UploadFile):
    if file.content_type != "application/pdf" and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await file.read()
    try:
        chunks = extract_pdf_chunks(file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    count = add_pdf_document(file.filename, chunks)
    return PDFUploadResponse(filename=file.filename, chunks_indexed=count)


@app.post("/qa/ask", response_model=QAResponse)
@limiter.limit(RATE_LIMIT)
def ask_question(request: Request, body: QuestionRequest):
    try:
        result = qa.answer_question(body.question)
    except Exception as exc:
        logger.exception("Q&A pipeline failed")
        raise HTTPException(status_code=502, detail=f"Q&A pipeline failed: {exc}") from exc

    return QAResponse(**result)
