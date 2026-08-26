from unittest.mock import MagicMock, patch

import pytest
import requests

from app.agents.nodes import analyze_style, check_dataset, extract_claims, gather_evidence

BASE_STATE = {
    "text": "Some article text.",
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


def _mock_completion(content: str):
    completion = MagicMock()
    completion.choices = [MagicMock(message=MagicMock(content=content))]
    return completion


@patch("app.agents.nodes.client.chat.completions.create")
def test_extract_claims_parses_json_array(mock_create):
    mock_create.return_value = _mock_completion('["Claim one", "Claim two"]')

    result = extract_claims({**BASE_STATE})

    assert result["claims"] == ["Claim one", "Claim two"]


@patch("app.agents.nodes.client.chat.completions.create")
def test_extract_claims_retries_on_invalid_json_then_succeeds(mock_create):
    mock_create.side_effect = [
        _mock_completion("not json"),
        _mock_completion('["Claim one"]'),
    ]

    result = extract_claims({**BASE_STATE})

    assert result["claims"] == ["Claim one"]
    assert mock_create.call_count == 2


@patch("app.agents.nodes.client.chat.completions.create")
def test_extract_claims_raises_after_exhausting_retries(mock_create):
    mock_create.return_value = _mock_completion("not json")

    with pytest.raises(RuntimeError):
        extract_claims({**BASE_STATE})


@patch("app.agents.nodes.search_web")
@patch("app.agents.nodes.search_fact_checks")
def test_gather_evidence_survives_one_failing_claim(mock_search, mock_web):
    mock_search.side_effect = [
        [{"claim_text": "x", "claimant": "y", "publisher": "z", "rating": "False", "url": "u"}],
        requests.RequestException("network error"),
    ]
    mock_web.return_value = []

    state = {**BASE_STATE, "claims": ["claim a", "claim b"]}
    result = gather_evidence(state)

    assert result["evidence"][0]["fact_checks"] != []
    assert result["evidence"][1]["fact_checks"] == []


@patch("app.agents.nodes.search_web")
@patch("app.agents.nodes.search_fact_checks")
def test_gather_evidence_collects_web_context_per_claim(mock_search, mock_web):
    mock_search.return_value = []
    mock_web.side_effect = [
        [{"title": "Real News Outlet", "url": "https://news.example.com/a", "content": "confirms it"}],
        requests.RequestException("timeout"),
    ]

    state = {**BASE_STATE, "claims": ["claim a", "claim b"]}
    result = gather_evidence(state)

    assert result["web_context"] == [
        {
            "claim": "claim a",
            "title": "Real News Outlet",
            "url": "https://news.example.com/a",
            "content": "confirms it",
        }
    ]


LONG_TEXT = "A" * 350  # clears DATASET_MATCH_MIN_CHARS so check_dataset actually looks it up


@patch("app.agents.nodes.match_against_dataset")
def test_check_dataset_short_circuits_on_strong_match(mock_match):
    mock_match.return_value = {"label": "FAKE", "similarity": 0.83, "title": "Some hoax headline"}

    result = check_dataset({**BASE_STATE, "text": LONG_TEXT})

    assert result["dataset_match"] is True
    assert result["dataset_label"] == "FAKE"
    assert result["verdict"] == "Likely Fake"
    assert result["confidence"] == 0.83
    assert result["sources"] == []


@patch("app.agents.nodes.match_against_dataset")
def test_check_dataset_falls_through_on_weak_match(mock_match):
    mock_match.return_value = {"label": "REAL", "similarity": 0.1, "title": "Unrelated article"}

    result = check_dataset({**BASE_STATE, "text": LONG_TEXT})

    assert result["dataset_match"] is False
    assert result["verdict"] == ""
    assert result["dataset_similarity"] == 0.1


@patch("app.agents.nodes.match_against_dataset")
def test_check_dataset_falls_through_when_no_index(mock_match):
    mock_match.return_value = None

    result = check_dataset({**BASE_STATE, "text": LONG_TEXT})

    assert result["dataset_match"] is False
    assert result["dataset_similarity"] == 0.0


@patch("app.agents.nodes.match_against_dataset")
def test_check_dataset_skips_lookup_for_short_text(mock_match):
    mock_match.return_value = {"label": "REAL", "similarity": 0.99, "title": "Some unrelated article"}

    result = check_dataset({**BASE_STATE, "text": "modi is dead now"})

    mock_match.assert_not_called()
    assert result["dataset_match"] is False
    assert result["verdict"] == ""
    assert result["dataset_similarity"] == 0.0


@patch("app.agents.nodes.client.chat.completions.create")
def test_analyze_style_parses_flags_and_score(mock_create):
    mock_create.return_value = _mock_completion(
        '{"style_flags": ["Excessive urgency language"], "style_score": 0.7}'
    )

    result = analyze_style({**BASE_STATE})

    assert result["style_flags"] == ["Excessive urgency language"]
    assert result["style_score"] == 0.7
