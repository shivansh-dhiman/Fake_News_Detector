from app import history


def test_save_and_get_recent_roundtrip():
    result = {
        "verdict": "Likely Fake",
        "confidence": 0.9,
        "explanation": "Test explanation",
        "claims": ["Test claim"],
        "evidence": [],
        "sources": [],
        "style_flags": [],
        "style_score": 0.5,
    }

    history.save_analysis("A unique test preview for roundtrip", result)
    recent = history.get_recent(limit=5)

    assert len(recent) > 0
    latest = recent[0]
    assert latest["input_preview"] == "A unique test preview for roundtrip"
    assert latest["verdict"] == "Likely Fake"
    assert latest["result"]["style_score"] == 0.5

    history._connection.execute("DELETE FROM history WHERE id = ?", (latest["id"],))
    history._connection.commit()
