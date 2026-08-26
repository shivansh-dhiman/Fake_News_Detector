import requests

from app.config import GOOGLE_FACT_CHECK_API_KEY

SEARCH_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


def search_fact_checks(query: str, max_results: int = 5) -> list[dict]:
    """Query Google's Fact Check Tools API for published fact-checks matching a claim."""
    response = requests.get(
        SEARCH_URL,
        params={
            "query": query,
            "key": GOOGLE_FACT_CHECK_API_KEY,
            "pageSize": max_results,
        },
        timeout=10,
    )
    response.raise_for_status()
    claims = response.json().get("claims", [])

    results = []
    for claim in claims:
        for review in claim.get("claimReview", []):
            results.append(
                {
                    "claim_text": claim.get("text", ""),
                    "claimant": claim.get("claimant", "unknown"),
                    "publisher": review.get("publisher", {}).get("name", "unknown"),
                    "rating": review.get("textualRating", "unrated"),
                    "url": review.get("url", ""),
                }
            )
    return results
