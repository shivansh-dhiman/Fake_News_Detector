import json
import logging
import re

import requests
from groq import Groq

from app.agents.state import GraphState
from app.config import DATASET_MATCH_MIN_CHARS, DATASET_MATCH_THRESHOLD, GROQ_API_KEY, GROQ_MODEL
from app.tools.dataset_matcher import match_against_dataset
from app.tools.fact_check_api import search_fact_checks
from app.tools.web_search import search_web

logger = logging.getLogger(__name__)

client = Groq(api_key=GROQ_API_KEY)

MAX_LLM_ATTEMPTS = 2


def check_dataset(state: GraphState) -> GraphState:
    """RAG step: look up the article against the labeled dataset before
    spending LLM/API calls. Only short-circuits the rest of the pipeline
    when the match is close enough to trust (DATASET_MATCH_THRESHOLD).

    Short inputs are skipped entirely: with few terms, TF-IDF cosine
    similarity can be dominated by a single rare shared word (e.g. a name),
    producing a high score against an article that's topically related but
    makes an entirely different claim.
    """
    if len(state["text"].strip()) < DATASET_MATCH_MIN_CHARS:
        return {**state, "dataset_match": False, "dataset_label": "", "dataset_similarity": 0.0}

    match = match_against_dataset(state["text"])
    if match is None or match["similarity"] < DATASET_MATCH_THRESHOLD:
        return {
            **state,
            "dataset_match": False,
            "dataset_label": "",
            "dataset_similarity": match["similarity"] if match else 0.0,
        }

    verdict = "Likely Fake" if match["label"] == "FAKE" else "Likely Real"
    return {
        **state,
        "dataset_match": True,
        "dataset_label": match["label"],
        "dataset_similarity": match["similarity"],
        "verdict": verdict,
        "confidence": round(match["similarity"], 2),
        "explanation": (
            f"This text closely matches a known {match['label']} article in our reference "
            f"dataset (\"{match['title']}\", similarity {match['similarity']:.2f}), so it was "
            "answered directly from the dataset without needing the web fact-check pipeline."
        ),
        "sources": [],
    }


def _call_groq_json(system: str, user: str) -> dict | list:
    last_error = None
    for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=2000,
            temperature=0.2,
            reasoning_effort="low",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        raw = completion.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            last_error = exc
            logger.warning(
                "LLM returned invalid JSON on attempt %d/%d: %s",
                attempt,
                MAX_LLM_ATTEMPTS,
                raw[:200],
            )
    raise RuntimeError(f"LLM did not return valid JSON after {MAX_LLM_ATTEMPTS} attempts") from last_error


def extract_claims(state: GraphState) -> GraphState:
    """Agent 1: pull out the discrete, checkable factual claims from the article."""
    claims = _call_groq_json(
        system=(
            "You extract discrete, checkable factual claims from news text. "
            "Respond with ONLY a JSON array of up to 5 short claim strings, no prose."
        ),
        user=state["text"],
    )
    return {**state, "claims": claims}


def gather_evidence(state: GraphState) -> GraphState:
    """Agent 2: search Google Fact Check Tools and the live web for evidence on each claim.

    Fact-check databases only cover claims someone has explicitly fact-checked (mostly
    hoaxes), so they're blind to real, recent events that simply haven't been "fact-checked"
    yet, and the LLM's own training data is blind to anything after its cutoff. A live web
    search fills that gap: independent, current news coverage confirming a claim is real
    evidence too, not just an absence of debunking.
    """
    evidence = []
    web_context = []
    for claim in state["claims"]:
        try:
            fact_checks = search_fact_checks(claim)
        except requests.RequestException as exc:
            logger.warning("Fact-check lookup failed for claim %r: %s", claim, exc)
            fact_checks = []
        evidence.append({"claim": claim, "fact_checks": fact_checks})

        try:
            web_results = search_web(claim, max_results=3)
        except requests.RequestException as exc:
            logger.warning("Web search failed for claim %r: %s", claim, exc)
            web_results = []
        for result in web_results:
            web_context.append({"claim": claim, **result})

    return {**state, "evidence": evidence, "web_context": web_context}


def analyze_style(state: GraphState) -> GraphState:
    """Agent 3: flag manipulative writing patterns independent of fact-checking."""
    result = _call_groq_json(
        system=(
            "You are a media literacy analyst. Examine the WRITING STYLE of this text for "
            "signals commonly associated with misinformation: sensational or emotionally "
            "manipulative language, clickbait phrasing, vague/anonymous sourcing "
            "('sources say', 'many believe'), excessive urgency or fear appeals, ALL-CAPS "
            "shouting, and missing attribution for factual claims. This is about tone and "
            "sourcing quality, not whether the claims are true. Respond with ONLY a JSON "
            "object with keys: style_flags (array of up to 5 short strings naming specific "
            "issues found, empty array if the writing looks like normal, sober reporting), "
            "style_score (0-1 float, where 0 = neutral/professional tone and 1 = highly "
            "sensationalized/manipulative tone)."
        ),
        user=state["text"],
    )
    return {**state, "style_flags": result["style_flags"], "style_score": result["style_score"]}


def synthesize_verdict(state: GraphState) -> GraphState:
    """Agent 4: weigh the claims, fact-check evidence, live web context, and style
    signals into a final verdict."""
    result = _call_groq_json(
        system=(
            "You are a fact-checking editor. Given an article, its extracted claims, "
            "third-party fact-check evidence for each claim, live general web search results "
            "for each claim, and an independent writing-style analysis, decide whether the "
            "article is likely real, likely fake, or uncertain. "
            "Each fact-check's claim_text is what THAT fact-check actually rated — it may only "
            "loosely match the claim it's attached to (search results are keyword-matched, not "
            "guaranteed relevant). Before treating a fact-check as evidence, check whether its "
            "claim_text really asserts or denies the same thing as the claim under review. A "
            "'False' rating on a claim_text about an unrelated photo/video/quote that merely "
            "references the same event or person as context (e.g. 'this video is not from after "
            "X's resignation') does NOT mean the event itself (X's resignation) is false — it "
            "only debunks that specific side-piece of misinformation, and often actually "
            "presupposes the event happened. Only let a fact-check push you toward 'Likely Fake' "
            "when its claim_text directly contradicts the claim being evaluated. "
            "The web_context entries are live search results, independent of your own training "
            "data cutoff — use them to confirm or refute claims about recent or current events "
            "you might otherwise consider implausible just because they're unfamiliar or dated "
            "after your training. Credible news coverage in web_context that corroborates a "
            "claim is real evidence toward 'Likely Real', even with no formal fact-check and "
            "even if the date is one you don't otherwise recognize. Never dismiss a claim as "
            "implausible solely because it postdates your training or because you don't "
            "personally recall the event — check web_context first. "
            "Base your verdict primarily on fact-check and web evidence that's actually "
            "on-topic; only when both are sparse or irrelevant should you fall back to reasoning "
            "from claim plausibility, and let a high style_score push you toward more skepticism "
            "in that case, and say so explicitly. Respond with ONLY a JSON object "
            'with keys: verdict (one of "Likely Real", "Likely Fake", "Uncertain"), '
            "confidence (0-1 float), explanation (2-4 sentences), "
            "sources (array of URLs pulled from the evidence or web_context, empty array if none)."
        ),
        user=json.dumps(
            {
                "article_text": state["text"],
                "claims": state["claims"],
                "evidence": state["evidence"],
                "web_context": state["web_context"],
                "style_flags": state["style_flags"],
                "style_score": state["style_score"],
            }
        ),
    )
    return {
        **state,
        "verdict": result["verdict"],
        "confidence": result["confidence"],
        "explanation": result["explanation"],
        "sources": result["sources"],
    }
