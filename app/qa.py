"""Q&A mode: answer a question from uploaded PDFs first, falling back to a
live web search when the PDFs don't have relevant content."""

import logging

import requests

from app.config import PDF_MATCH_THRESHOLD
from app.llm import call_groq_json
from app.tools.pdf_index import search as search_pdfs
from app.tools.web_search import search_web

logger = logging.getLogger(__name__)

NO_INFO_ANSWER = "I couldn't find relevant information in the uploaded PDFs or on the web."


def _synthesize_answer(question: str, context: str) -> str:
    result = call_groq_json(
        system=(
            "You answer questions using ONLY the provided context. If the context doesn't "
            "contain enough information to answer, say so plainly instead of guessing. "
            "Respond with ONLY a JSON object with key: answer (a concise, direct answer, "
            "2-5 sentences)."
        ),
        user=f"Question: {question}\n\nContext:\n{context}",
    )
    return result["answer"]


def answer_question(question: str) -> dict:
    pdf_matches = search_pdfs(question)
    if pdf_matches and pdf_matches[0]["similarity"] >= PDF_MATCH_THRESHOLD:
        context = "\n\n".join(
            f"[{m['source']}, page {m['page']}]\n{m['text']}" for m in pdf_matches
        )
        answer = _synthesize_answer(question, context)
        return {
            "answer": answer,
            "source": "pdf",
            "citations": [f"{m['source']} (p.{m['page']})" for m in pdf_matches],
            "confidence": round(pdf_matches[0]["similarity"], 3),
        }

    try:
        web_results = search_web(question)
    except requests.RequestException as exc:
        logger.warning("Web search failed for question %r: %s", question, exc)
        web_results = []

    if not web_results:
        return {"answer": NO_INFO_ANSWER, "source": "none", "citations": [], "confidence": 0.0}

    context = "\n\n".join(f"[{r['title']}]({r['url']})\n{r['content']}" for r in web_results)
    answer = _synthesize_answer(question, context)
    return {
        "answer": answer,
        "source": "web",
        "citations": [r["url"] for r in web_results],
        "confidence": 0.0,
    }
