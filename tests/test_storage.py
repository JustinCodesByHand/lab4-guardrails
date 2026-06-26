import storage


def _sample(content_id="c1"):
    return {
        "content_id": content_id,
        "creator_id": "u1",
        "text": "hello world",
        "llm_score": 0.8,
        "stylometry_score": 0.7,
        "confidence": 0.75,
        "attribution": "likely_ai",
        "status": "classified",
    }


def test_save_and_get(db_path):
    storage.init_db(db_path)
    storage.save_submission(_sample(), db_path)
    row = storage.get_content("c1", db_path)
    assert row["creator_id"] == "u1"
    assert row["attribution"] == "likely_ai"
    assert row["status"] == "classified"


def test_get_missing_returns_none(db_path):
    storage.init_db(db_path)
    assert storage.get_content("nope", db_path) is None


def test_update_status(db_path):
    storage.init_db(db_path)
    storage.save_submission(_sample(), db_path)
    ok = storage.update_status("c1", "under_review", "I wrote this myself", db_path)
    assert ok is True
    row = storage.get_content("c1", db_path)
    assert row["status"] == "under_review"


def test_update_status_missing_returns_false(db_path):
    storage.init_db(db_path)
    assert storage.update_status("nope", "under_review", "x", db_path) is False


def test_audit_append_and_recent(db_path):
    storage.init_db(db_path)
    storage.append_audit({**_sample(), "appeal_reasoning": None}, db_path)
    storage.append_audit({**_sample("c2"), "appeal_reasoning": "appeal text"}, db_path)
    entries = storage.recent_log(10, db_path)
    assert len(entries) == 2
    assert entries[0]["content_id"] == "c2"
    assert entries[0]["appeal_reasoning"] == "appeal text"
