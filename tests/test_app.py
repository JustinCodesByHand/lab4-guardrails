import pytest
import app as app_module
import storage


@pytest.fixture
def client(tmp_path, monkeypatch):
    db = str(tmp_path / "app.db")
    monkeypatch.setattr(app_module, "DB_PATH", db)
    monkeypatch.setattr(storage, "DB_PATH", db)
    # deterministic signals — no network
    monkeypatch.setattr(app_module, "llm_score", lambda t: 0.9)
    monkeypatch.setattr(app_module, "stylometry_score", lambda t: 0.9)
    storage.init_db(db)
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_submit_returns_full_shape(client):
    r = client.post("/submit", json={"text": "some text here", "creator_id": "u1"})
    assert r.status_code == 200
    data = r.get_json()
    for key in ("content_id", "attribution", "confidence", "label",
                "llm_score", "stylometry_score"):
        assert key in data
    assert data["attribution"] == "likely_ai"   # 0.9/0.9 -> 0.9
    assert "AI-generated" in data["label"]


def test_submit_missing_text_is_400(client):
    r = client.post("/submit", json={"creator_id": "u1"})
    assert r.status_code == 400


def test_submit_writes_audit_entry(client):
    client.post("/submit", json={"text": "hello", "creator_id": "u1"})
    r = client.get("/log")
    entries = r.get_json()["entries"]
    assert len(entries) == 1
    assert entries[0]["attribution"] == "likely_ai"
