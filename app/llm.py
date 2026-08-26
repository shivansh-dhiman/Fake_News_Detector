"""Shared Groq JSON-completion helper for the Q&A pipeline (app/qa.py).

Kept separate from app/agents/nodes.py's own client/helper so existing tests
that patch app.agents.nodes.client continue to target the same object.
"""

import json
import logging
import re

from groq import Groq

from app.config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

client = Groq(api_key=GROQ_API_KEY)

MAX_LLM_ATTEMPTS = 2


def call_groq_json(system: str, user: str) -> dict | list:
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
