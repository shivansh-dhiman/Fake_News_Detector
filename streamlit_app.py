"""Streamlit deployment entrypoint for Veritas — Fake News Detector.

Streamlit Community Cloud runs a single script (`streamlit run streamlit_app.py`),
so this is a self-contained UI that calls the same LangGraph pipeline and Q&A
module the FastAPI app (app/main.py) uses — no HTTP layer in between.

Secrets on Streamlit Cloud are configured under Settings -> Secrets as
GROQ_API_KEY = "..." etc. They're copied into os.environ below so that
app.config (which reads plain env vars via python-dotenv) picks them up
the same way it does for local .env-based development.
"""

import os

import streamlit as st

st.set_page_config(page_title="Veritas — Fake News Detector", page_icon="🛡️", layout="centered")

_secrets_paths = [
    os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml"),
    os.path.expanduser(os.path.join("~", ".streamlit", "secrets.toml")),
]
if any(os.path.exists(_p) for _p in _secrets_paths):
    try:
        for _key, _value in st.secrets.items():
            if isinstance(_value, str) and _key not in os.environ:
                os.environ[_key] = _value
    except Exception:
        pass

from app import history, qa  # noqa: E402
from app.agents.graph import compiled_graph  # noqa: E402
from app.tools.article_fetcher import fetch_article_text  # noqa: E402
from app.tools.pdf_extractor import extract_pdf_chunks  # noqa: E402
from app.tools.pdf_index import add_document as add_pdf_document  # noqa: E402

VERDICT_COLORS = {"likely real": "#12946f", "likely fake": "#e0335f", "uncertain": "#b8860b"}
VERDICT_ICONS = {"likely real": "🟢", "likely fake": "🔴", "uncertain": "🟡"}


def verdict_badge(verdict: str) -> str:
    color = VERDICT_COLORS.get(verdict.lower(), "#b8860b")
    return (
        f"<span style='background:{color}22;color:{color};padding:5px 14px;"
        f"border-radius:999px;font-weight:700;font-size:0.9rem;'>{verdict}</span>"
    )


def build_initial_state(text: str) -> dict:
    return {
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


def render_result(result: dict) -> None:
    st.markdown(verdict_badge(result["verdict"]), unsafe_allow_html=True)
    st.progress(min(max(result["confidence"], 0.0), 1.0), text=f"{round(result['confidence'] * 100)}% confidence")
    st.write(result["explanation"])

    if result.get("dataset_match"):
        st.caption(
            f"Matched directly against a known **{result.get('dataset_label')}** article in the "
            f"reference dataset (similarity {result.get('dataset_similarity', 0):.2f}) — the full "
            "pipeline below was skipped."
        )
        return

    st.markdown("**Writing style signals**")
    st.progress(min(max(result.get("style_score", 0.0), 0.0), 1.0), text=f"{round(result.get('style_score', 0) * 100)}% sensationalized")
    flags = result.get("style_flags") or []
    st.write(", ".join(flags) if flags else "No manipulative style signals detected.")

    if result.get("claims"):
        st.markdown("**Extracted claims**")
        for claim in result["claims"]:
            st.markdown(f"- {claim}")

    if result.get("sources"):
        st.markdown("**Sources**")
        for url in result["sources"]:
            st.markdown(f"- [{url}]({url})")


def render_qa_result(result: dict) -> None:
    labels = {"pdf": "From your PDFs", "web": "From the web", "none": "No source found"}
    st.markdown(f"**{labels.get(result['source'], result['source'])}**")
    if result["source"] == "pdf":
        st.caption(f"{round(result['confidence'] * 100)}% match")
    st.write(result["answer"])
    if result.get("citations"):
        st.markdown("**Sources**")
        for citation in result["citations"]:
            st.markdown(f"- {citation}")


st.title("🛡️ Veritas")
st.caption("Multi-agent AI verification pipeline — paste an article or claim, get a verdict.")

fact_tab, qa_tab = st.tabs(["Fact Checker", "Ask a Question"])

with fact_tab:
    mode = st.radio("Input type", ["Text", "URL"], horizontal=True, label_visibility="collapsed")
    text_value, url_value = "", ""
    if mode == "Text":
        text_value = st.text_area("Article text", height=180, placeholder="Paste article text here...", label_visibility="collapsed")
    else:
        url_value = st.text_input("Article URL", placeholder="https://example.com/news-article", label_visibility="collapsed")

    if st.button("Check News", type="primary"):
        if mode == "Text" and len(text_value.strip()) < 10:
            st.warning("Please paste at least 10 characters of text.")
        elif mode == "URL" and not url_value.strip().lower().startswith(("http://", "https://")):
            st.warning("Please enter a valid URL starting with http:// or https://")
        else:
            with st.spinner("Analyzing claims and checking sources…"):
                try:
                    text = fetch_article_text(url_value) if mode == "URL" else text_value
                    result = compiled_graph.invoke(build_initial_state(text))
                    history.save_analysis(url_value or text, result)
                    st.session_state["last_result"] = result
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")

    if st.session_state.get("last_result"):
        st.divider()
        render_result(st.session_state["last_result"])

    st.divider()
    st.markdown("**Recent checks**")
    entries = history.get_recent(limit=8)
    if not entries:
        st.caption("No checks yet — analyze something above to see it here.")
    for entry in entries:
        icon = VERDICT_ICONS.get(entry["verdict"].lower(), "🟡")
        st.markdown(f"{icon} {entry['input_preview']}")

with qa_tab:
    st.markdown("**Upload reference PDFs**")
    uploaded_files = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    for uploaded in uploaded_files or []:
        indexed_key = f"pdf_indexed::{uploaded.name}::{uploaded.size}"
        if indexed_key not in st.session_state:
            try:
                chunks = extract_pdf_chunks(uploaded.getvalue())
                count = add_pdf_document(uploaded.name, chunks)
                st.session_state[indexed_key] = True
                st.success(f"{uploaded.name} — {count} chunks indexed")
            except ValueError as exc:
                st.error(f"{uploaded.name} — {exc}")

    st.markdown("**Ask a question**")
    question = st.text_input(
        "Question",
        placeholder="Ask something about your uploaded PDFs, or any current-events question...",
        label_visibility="collapsed",
    )
    if st.button("Ask"):
        if len(question.strip()) < 5:
            st.warning("Please enter a question (at least 5 characters).")
        else:
            with st.spinner("Searching PDFs and the web…"):
                try:
                    st.session_state["qa_result"] = qa.answer_question(question)
                except Exception as exc:
                    st.error(f"Q&A failed: {exc}")

    if st.session_state.get("qa_result"):
        st.divider()
        render_qa_result(st.session_state["qa_result"])

st.divider()
st.caption("Verdicts are generated by AI and third-party fact-check sources — always use your own judgement.")
