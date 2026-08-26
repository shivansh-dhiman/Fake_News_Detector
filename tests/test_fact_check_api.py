from unittest.mock import MagicMock, patch

from app.tools.fact_check_api import search_fact_checks


@patch("app.tools.fact_check_api.requests.get")
def test_search_fact_checks_parses_claim_reviews(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "claims": [
            {
                "text": "Vaccines contain microchips",
                "claimant": "Viral post",
                "claimReview": [
                    {
                        "publisher": {"name": "FactCheck.org"},
                        "textualRating": "False",
                        "url": "https://example.com/factcheck",
                    }
                ],
            }
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    results = search_fact_checks("vaccines microchips")

    assert results == [
        {
            "claim_text": "Vaccines contain microchips",
            "claimant": "Viral post",
            "publisher": "FactCheck.org",
            "rating": "False",
            "url": "https://example.com/factcheck",
        }
    ]


@patch("app.tools.fact_check_api.requests.get")
def test_search_fact_checks_returns_empty_when_no_claims(mock_get):
    mock_response = MagicMock()
    mock_response.json.return_value = {}
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    assert search_fact_checks("some obscure claim") == []
