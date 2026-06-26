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


def test_submit_runs_real_stylometry(tmp_path, monkeypatch):
    # Only the LLM (network) signal is stubbed; the real stylometry signal must
    # actually flow through /submit and produce a genuine score in [0,1].
    db = str(tmp_path / "real_stylo.db")
    monkeypatch.setattr(app_module, "DB_PATH", db)
    monkeypatch.setattr(storage, "DB_PATH", db)
    monkeypatch.setattr(app_module, "llm_score", lambda t: 0.5)
    storage.init_db(db)
    app_module.app.config["TESTING"] = True
    client = app_module.app.test_client()

    text = ("The quarterly report indicates steady growth. Revenue increased. "
            "Margins held firm across every operating segment this period.")
    r = client.post("/submit", json={"text": text, "creator_id": "u1"})
    assert r.status_code == 200
    stylo = r.get_json()["stylometry_score"]
    assert 0.0 <= stylo <= 1.0
    assert stylo != 0.5  # proves the real signal ran, not a constant stub
