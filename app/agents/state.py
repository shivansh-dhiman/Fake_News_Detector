from typing import TypedDict


class ClaimEvidenceEntry(TypedDict):
    claim: str
    fact_checks: list[dict]


class GraphState(TypedDict):
    text: str
    claims: list[str]
    evidence: list[ClaimEvidenceEntry]
    web_context: list[dict]
    style_flags: list[str]
    style_score: float
    verdict: str
    confidence: float
    explanation: str
    sources: list[str]
    dataset_match: bool
    dataset_label: str
    dataset_similarity: float
