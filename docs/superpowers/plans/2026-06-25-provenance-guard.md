# Provenance Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Flask backend that classifies text as human- or AI-written using two independent signals, scores confidence on an asymmetric scale, returns a reader-facing transparency label, persists to SQLite, and supports appeals + rate limiting + an audit log.

**Architecture:** Pure-function detection signals (`signals/llm.py`, `signals/stylometry.py`) each return `P(AI) ∈ [0,1]`. `scoring.py` combines them (50/50), classifies into three bands (≥0.75 / 0.40 / <0.40), and maps to label text. `storage.py` wraps SQLite (submissions + audit). `app.py` wires four Flask routes with Flask-Limiter. Tunables live in `config.py` so M4 calibration touches one file.

**Tech Stack:** Python 3, Flask, Flask-Limiter, Groq SDK (`llama-3.3-70b-versatile`), SQLite (stdlib), pytest (dev).

**Build dir:** `C:\Users\jcmac\RAG4\ai201-project4-provenance-guard` (all paths below are relative to it).

**Milestone mapping:** Tasks 1–4 = brief Milestone 3 · Tasks 5–7 = Milestone 4 · Tasks 8–12 = Milestone 5. README/walkthrough = Milestone 6.

**Testing convention:** Storage functions take an optional `db_path` arg so tests use a `tmp_path` DB. LLM network calls are isolated behind `_call_groq(text)`; tests monkeypatch it. No test hits the real Groq API.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `config.py` | Env load + all tunables (model, weights, thresholds, rate limits, DB path) |
| `signals/__init__.py` | Package marker |
| `signals/llm.py` | `llm_score(text)`; `_parse_score`, `_call_groq` helpers |
| `signals/stylometry.py` | `stylometry_score(text)` + three sub-score functions |
| `scoring.py` | `combine`, `classify`, `label_for`, `LABELS` |
| `storage.py` | SQLite: `init_db`, `save_submission`, `get_content`, `update_status`, `append_audit`, `recent_log` |
| `app.py` | Flask routes `/submit`, `/appeal`, `/log`, `/content/<id>` + limiter |
| `scripts/calibrate.py` | Manual M4 calibration over the 4 reference inputs (real Groq) |
| `tests/conftest.py` | Shared fixtures (temp DB, sample texts) |
| `tests/test_*.py` | Unit tests per module |

---

## Task 0: Dev environment + package skeleton

**Files:**
- Create: `signals/__init__.py` (empty), `tests/__init__.py` (empty)
- Create: `config.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create venv + install deps**

Run (Git Bash):
```bash
cd /c/Users/jcmac/RAG4/ai201-project4-provenance-guard
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt pytest
```
Expected: installs flask, flask-limiter, groq, python-dotenv, pytest.

- [ ] **Step 2: Create `config.py`**

```python
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"

DB_PATH = "provenance.db"

# Scoring tunables (calibrated in Milestone 4)
LLM_WEIGHT = 0.5
STYLO_WEIGHT = 0.5
AI_THRESHOLD = 0.75       # confidence >= this  -> likely_ai
HUMAN_THRESHOLD = 0.40    # confidence <  this  -> likely_human

# Stylometry sub-score weights (TTR weighted lightly; see planning.md §1)
STYLO_VAR_WEIGHT = 0.45
STYLO_PUNCT_WEIGHT = 0.35
STYLO_TTR_WEIGHT = 0.20

RATE_LIMITS = "10 per minute;100 per day"
```

- [ ] **Step 3: Create empty package markers**

Create `signals/__init__.py` and `tests/__init__.py` as empty files.

- [ ] **Step 4: Create `tests/conftest.py`**

```python
import pytest

# Reference inputs from the brief (used in stylometry + calibration tests)
CLEAR_AI = (
    "Artificial intelligence represents a transformative paradigm shift in modern "
    "society. It is important to note that while the benefits of AI are numerous, it "
    "is equally essential to consider the ethical implications. Furthermore, "
    "stakeholders across various sectors must collaborate to ensure responsible "
    "deployment."
)

CLEAR_HUMAN = (
    "ok so i finally tried that new ramen place downtown and honestly? underwhelming. "
    "the broth was fine but they put WAY too much sodium in it and i was thirsty for "
    "like three hours after. my friend got the spicy version and said it was better. "
    "probably won't go back unless someone drags me there"
)

@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")
```

- [ ] **Step 5: Commit**

```bash
git add config.py signals/__init__.py tests/__init__.py tests/conftest.py
git commit -m "chore: dev env, config tunables, test scaffolding"
```

---

## Task 1: Stylometry sub-scores (M4 logic, built first — pure + easiest to TDD)

> Built before the LLM signal because it is deterministic. Milestone-wise this is M4 work; building it early de-risks calibration.

**Files:**
- Create: `signals/stylometry.py`
- Test: `tests/test_stylometry.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_stylometry.py
from signals.stylometry import (
    sentence_length_subscore,
    type_token_ratio_subscore,
    punctuation_subscore,
    stylometry_score,
)
from tests.conftest import CLEAR_AI, CLEAR_HUMAN


def test_uniform_sentences_score_ai_ward():
    # Five sentences all the same length -> std 0 -> max AI-likeness
    text = "the cat sat on mat. the dog ran in park. the bird flew over tree. " \
           "the fish swam in pond. the fox hid in den."
    assert sentence_length_subscore(text) > 0.9


def test_variable_sentences_score_human_ward():
    text = "Hi. I went to the enormous, sprawling market downtown yesterday afternoon " \
           "and wandered for hours. Loved it. The vendors there, especially the older " \
           "gentleman selling figs, made the whole rainy miserable trip genuinely " \
           "worthwhile somehow. Wow."
    assert sentence_length_subscore(text) < 0.5


def test_short_text_returns_neutral():
    # Edge case B: too few sentences to judge -> ~0.5
    assert sentence_length_subscore("Just one short line.") == 0.5


def test_ttr_subscore_in_range():
    assert 0.0 <= type_token_ratio_subscore(CLEAR_HUMAN) <= 1.0


def test_punctuation_variety_low_is_ai_ward():
    plain = "the cat sat on the mat the dog ran the bird flew the fish swam"
    varied = 'Wait—really? Yes; I think so... "maybe," she said (laughing).'
    assert punctuation_subscore(plain) > punctuation_subscore(varied)


def test_stylometry_score_separates_clear_cases():
    # The uniform AI sample should out-score the casual human sample
    assert stylometry_score(CLEAR_AI) > stylometry_score(CLEAR_HUMAN)


def test_stylometry_score_bounded():
    for t in (CLEAR_AI, CLEAR_HUMAN, "tiny."):
        assert 0.0 <= stylometry_score(t) <= 1.0
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/test_stylometry.py -v`
Expected: FAIL (ImportError: cannot import name 'sentence_length_subscore').

- [ ] **Step 3: Implement `signals/stylometry.py`**

```python
import re
import statistics

from config import STYLO_VAR_WEIGHT, STYLO_PUNCT_WEIGHT, STYLO_TTR_WEIGHT

_PUNCT_CHARS = [",", ";", ":", ".", "!", "?", "-", "—", "(", ")", '"', "'", "..."]


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _sentences(text: str) -> list[str]:
    parts = re.split(r"[.!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def _words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def sentence_length_subscore(text: str) -> float:
    """Uniform sentence lengths read AI-ward (1.0); high variance human-ward (0.0)."""
    sentences = _sentences(text)
    if len(sentences) < 2:
        return 0.5  # edge case B: not enough data to judge
    lengths = [len(_words(s)) for s in sentences]
    std = statistics.pstdev(lengths)
    # std >= 8 -> 0.0 (very human), std <= 2 -> 1.0 (very AI), linear between
    return _clamp((8.0 - std) / (8.0 - 2.0))


def type_token_ratio_subscore(text: str) -> float:
    """Lower vocabulary diversity nudges AI-ward. Lightly weighted (length-sensitive)."""
    words = _words(text)
    if not words:
        return 0.5
    ttr = len(set(words)) / len(words)
    # ttr 0.3 -> ~AI, ttr 0.8 -> ~human; map inversely and clamp
    return _clamp((0.8 - ttr) / (0.8 - 0.3))


def punctuation_subscore(text: str) -> float:
    """Few distinct punctuation types reads AI-ward (regular/sparse)."""
    present = {p for p in _PUNCT_CHARS if p in text}
    variety = len(present)
    # 0 distinct -> 1.0 (AI), >=5 distinct -> 0.0 (human)
    return _clamp((5 - variety) / 5)


def stylometry_score(text: str) -> float:
    """Weighted average of three sub-scores. Returns P(AI) in [0,1]."""
    var = sentence_length_subscore(text)
    ttr = type_token_ratio_subscore(text)
    punct = punctuation_subscore(text)
    score = (
        STYLO_VAR_WEIGHT * var
        + STYLO_PUNCT_WEIGHT * punct
        + STYLO_TTR_WEIGHT * ttr
    )
    return _clamp(score)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_stylometry.py -v`
Expected: PASS (7 passed). If `test_stylometry_score_separates_clear_cases` fails, print the three sub-scores for each input and adjust the threshold constants — that is expected calibration, do it now.

- [ ] **Step 5: Commit**

```bash
git add signals/stylometry.py tests/test_stylometry.py
git commit -m "feat: stylometric detection signal (variance, TTR, punctuation)"
```

---

## Task 2: SQLite storage layer

**Files:**
- Create: `storage.py`
- Test: `tests/test_storage.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_storage.py
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
    # most recent first
    assert entries[0]["content_id"] == "c2"
    assert entries[0]["appeal_reasoning"] == "appeal text"
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/test_storage.py -v`
Expected: FAIL (ModuleNotFoundError / AttributeError: no attribute 'init_db').

- [ ] **Step 3: Implement `storage.py`**

```python
import sqlite3
from datetime import datetime, timezone

from config import DB_PATH


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _conn(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    with _conn(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS submissions (
                content_id TEXT PRIMARY KEY,
                creator_id TEXT,
                text TEXT,
                llm_score REAL,
                stylometry_score REAL,
                confidence REAL,
                attribution TEXT,
                status TEXT,
                created_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_id TEXT,
                creator_id TEXT,
                timestamp TEXT,
                attribution TEXT,
                confidence REAL,
                llm_score REAL,
                stylometry_score REAL,
                status TEXT,
                appeal_reasoning TEXT
            )
            """
        )


def save_submission(record: dict, db_path: str = DB_PATH) -> None:
    with _conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO submissions
                (content_id, creator_id, text, llm_score, stylometry_score,
                 confidence, attribution, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["content_id"], record["creator_id"], record["text"],
                record["llm_score"], record["stylometry_score"], record["confidence"],
                record["attribution"], record["status"], _now(),
            ),
        )


def get_content(content_id: str, db_path: str = DB_PATH) -> dict | None:
    with _conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM submissions WHERE content_id = ?", (content_id,)
        ).fetchone()
    return dict(row) if row else None


def update_status(content_id: str, status: str, reasoning: str,
                  db_path: str = DB_PATH) -> bool:
    with _conn(db_path) as conn:
        cur = conn.execute(
            "UPDATE submissions SET status = ? WHERE content_id = ?",
            (status, content_id),
        )
    return cur.rowcount > 0


def append_audit(entry: dict, db_path: str = DB_PATH) -> None:
    with _conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO audit
                (content_id, creator_id, timestamp, attribution, confidence,
                 llm_score, stylometry_score, status, appeal_reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry["content_id"], entry["creator_id"], _now(),
                entry["attribution"], entry["confidence"], entry["llm_score"],
                entry["stylometry_score"], entry["status"],
                entry.get("appeal_reasoning"),
            ),
        )


def recent_log(n: int = 20, db_path: str = DB_PATH) -> list[dict]:
    with _conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM audit ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
    return [dict(r) for r in rows]
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_storage.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add storage.py tests/test_storage.py
git commit -m "feat: SQLite storage for submissions and audit log"
```

---

## Task 3: LLM detection signal

**Files:**
- Create: `signals/llm.py`
- Test: `tests/test_llm.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_llm.py
import signals.llm as llm


def test_parse_valid_score():
    assert llm._parse_score("Verdict: AI\nScore: 0.82") == 0.82


def test_parse_clamps_high():
    assert llm._parse_score("Score: 1.7") == 1.0


def test_parse_unparseable_returns_none():
    assert llm._parse_score("I think it is probably human, hard to say") is None


def test_llm_score_uses_parsed_value(monkeypatch):
    monkeypatch.setattr(llm, "_call_groq", lambda text: "Verdict: AI\nScore: 0.9")
    assert llm.llm_score("whatever") == 0.9


def test_llm_score_falls_back_to_neutral_on_garbage(monkeypatch):
    monkeypatch.setattr(llm, "_call_groq", lambda text: "no score here")
    assert llm.llm_score("whatever") == 0.5


def test_llm_score_falls_back_on_exception(monkeypatch):
    def boom(text):
        raise RuntimeError("api down")
    monkeypatch.setattr(llm, "_call_groq", boom)
    assert llm.llm_score("whatever") == 0.5
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/test_llm.py -v`
Expected: FAIL (AttributeError: module 'signals.llm' has no attribute '_parse_score').

- [ ] **Step 3: Implement `signals/llm.py`**

```python
import re

from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL

_client = Groq(api_key=GROQ_API_KEY)

_SYSTEM_PROMPT = (
    "You are a forensic text analyst. Judge whether the text was written by a human "
    "or generated by an AI language model. Consider voice, idiosyncrasy, coherence, "
    "and hedging. Respond in exactly two lines:\n"
    "Verdict: <AI or HUMAN>\n"
    "Score: <a number from 0.0 to 1.0 = probability the text is AI-generated>"
)

_NEUTRAL = 0.5


def _call_groq(text: str) -> str:
    resp = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return resp.choices[0].message.content


def _parse_score(raw: str) -> float | None:
    for line in raw.splitlines():
        if line.strip().lower().startswith("score:"):
            m = re.search(r"[-+]?\d*\.?\d+", line.split(":", 1)[1])
            if m:
                return max(0.0, min(1.0, float(m.group())))
    return None


def llm_score(text: str) -> float:
    """Return P(AI) in [0,1] from the LLM signal. Fail toward neutral 0.5."""
    try:
        raw = _call_groq(text)
    except Exception:
        return _NEUTRAL
    parsed = _parse_score(raw)
    return parsed if parsed is not None else _NEUTRAL
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_llm.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Manual smoke test against real Groq (needs `.env`)**

Run:
```bash
python -c "from signals.llm import llm_score; print(llm_score('It is important to note that stakeholders must collaborate.'))"
```
Expected: a float; the formal sample should print something above ~0.6. (Requires `GROQ_API_KEY` in `.env`.)

- [ ] **Step 6: Commit**

```bash
git add signals/llm.py tests/test_llm.py
git commit -m "feat: Groq LLM detection signal with neutral fallback"
```

---

## Task 4: Scoring — combine, classify, labels

**Files:**
- Create: `scoring.py`
- Test: `tests/test_scoring.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_scoring.py
from scoring import combine, classify, label_for, LABELS


def test_combine_is_even_average():
    assert combine(1.0, 0.0) == 0.5
    assert combine(0.8, 0.6) == 0.7


def test_classify_bands():
    assert classify(0.90) == "likely_ai"
    assert classify(0.75) == "likely_ai"      # boundary inclusive
    assert classify(0.74) == "uncertain"
    assert classify(0.40) == "uncertain"       # boundary
    assert classify(0.39) == "likely_human"
    assert classify(0.10) == "likely_human"


def test_confident_llm_plus_neutral_stylo_is_uncertain():
    # Documented consequence of 50/50: 0.9 + 0.5 -> 0.7 -> uncertain
    assert classify(combine(0.9, 0.5)) == "uncertain"


def test_label_for_each_attribution():
    for attr in ("likely_ai", "uncertain", "likely_human"):
        assert label_for(attr) == LABELS[attr]
    assert "AI-generated" in LABELS["likely_ai"]
    assert "Uncertain" in LABELS["uncertain"]
    assert "human-written" in LABELS["likely_human"]
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/test_scoring.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'scoring').

- [ ] **Step 3: Implement `scoring.py`**

> Label text is the verbatim wording from planning.md §3, minus markdown emphasis (these strings are returned through the API as plain text). Keep README labels identical to these.

```python
from config import LLM_WEIGHT, STYLO_WEIGHT, AI_THRESHOLD, HUMAN_THRESHOLD

LABELS = {
    "likely_ai": (
        "🤖 Likely AI-generated. Our automated analysis suggests this text was "
        "probably produced with significant help from an AI tool. Automated detection "
        "is imperfect and can be wrong — if you're the creator and disagree, you can "
        "appeal this assessment for human review."
    ),
    "uncertain": (
        "❓ Uncertain origin. Our analysis couldn't confidently tell whether a person "
        "or an AI wrote this; the signals were mixed. This is not a judgment against "
        "the creator — treat the authorship as simply unverified. Creators can request "
        "a review."
    ),
    "likely_human": (
        "✍️ Likely human-written. Our analysis found the natural variation typical of "
        "human writing. This is an automated estimate, not a guarantee — and like every "
        "assessment, it can be appealed."
    ),
}


def combine(llm_score: float, stylometry_score: float) -> float:
    return LLM_WEIGHT * llm_score + STYLO_WEIGHT * stylometry_score


def classify(confidence: float) -> str:
    if confidence >= AI_THRESHOLD:
        return "likely_ai"
    if confidence < HUMAN_THRESHOLD:
        return "likely_human"
    return "uncertain"


def label_for(attribution: str) -> str:
    return LABELS[attribution]
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_scoring.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add scoring.py tests/test_scoring.py
git commit -m "feat: confidence scoring, asymmetric bands, transparency labels"
```

---

## Task 5: Flask app — `/submit` + `/log` (full pipeline wired)

**Files:**
- Create: `app.py`
- Test: `tests/test_app.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_app.py
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
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pytest tests/test_app.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'app').

- [ ] **Step 3: Implement `app.py`**

```python
import uuid

from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import storage
from config import DB_PATH, RATE_LIMITS
from signals.llm import llm_score
from signals.stylometry import stylometry_score
from scoring import combine, classify, label_for

app = Flask(__name__)
limiter = Limiter(get_remote_address, app=app, default_limits=[], storage_uri="memory://")

storage.init_db(DB_PATH)


@app.route("/submit", methods=["POST"])
@limiter.limit(RATE_LIMITS)
def submit():
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    creator_id = (body.get("creator_id") or "").strip()
    if not text or not creator_id:
        return jsonify({"error": "text and creator_id are required"}), 400

    llm = llm_score(text)
    stylo = stylometry_score(text)
    confidence = combine(llm, stylo)
    attribution = classify(confidence)
    label = label_for(attribution)
    content_id = str(uuid.uuid4())

    record = {
        "content_id": content_id, "creator_id": creator_id, "text": text,
        "llm_score": llm, "stylometry_score": stylo, "confidence": confidence,
        "attribution": attribution, "status": "classified",
    }
    storage.save_submission(record, DB_PATH)
    storage.append_audit({**record, "appeal_reasoning": None}, DB_PATH)

    return jsonify({
        "content_id": content_id, "attribution": attribution,
        "confidence": round(confidence, 4), "label": label,
        "llm_score": round(llm, 4), "stylometry_score": round(stylo, 4),
    })


@app.route("/log", methods=["GET"])
def log():
    return jsonify({"entries": storage.recent_log(50, DB_PATH)})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `pytest tests/test_app.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Manual end-to-end curl (real Groq)**

Run server in one terminal: `python app.py`. In another:
```bash
curl -s -X POST http://localhost:5000/submit \
  -H "Content-Type: application/json" \
  -d '{"text": "The sun dipped below the horizon, painting the sky in hues of amber and rose. I sat on the porch, coffee in hand, watching the neighborhood slowly go quiet.", "creator_id": "test-user-1"}' | python -m json.tool
```
Expected: JSON with `content_id`, `attribution`, `confidence`, `label`, both scores. Save the `content_id` for Task 7.

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: /submit pipeline (both signals -> score -> label) and /log"
```

---

## Task 6: Calibration script + reference-input check (M4 verification)

**Files:**
- Create: `scripts/calibrate.py`
- Test: `tests/test_calibration.py`

- [ ] **Step 1: Write the stylometry ordering test (deterministic, no network)**

```python
# tests/test_calibration.py
from signals.stylometry import stylometry_score
from tests.conftest import CLEAR_AI, CLEAR_HUMAN

BORDERLINE_FORMAL_HUMAN = (
    "The relationship between monetary policy and asset price inflation has been "
    "extensively studied in the literature. Central banks face a fundamental tension "
    "between their mandate for price stability and the unintended consequences of "
    "prolonged low interest rates on equity and real estate valuations."
)


def test_stylometry_orders_clear_cases():
    assert stylometry_score(CLEAR_AI) > stylometry_score(CLEAR_HUMAN)


def test_formal_human_scores_above_casual_human():
    # Demonstrates the known false-positive pressure (edge case C-style)
    assert stylometry_score(BORDERLINE_FORMAL_HUMAN) > stylometry_score(CLEAR_HUMAN)
```

- [ ] **Step 2: Run, verify pass (adjust constants if ordering is wrong)**

Run: `pytest tests/test_calibration.py -v`
Expected: PASS (2 passed). If not, tune the threshold constants in `signals/stylometry.py` until the ordering holds, then re-run.

- [ ] **Step 3: Create `scripts/calibrate.py` (manual, real Groq)**

```python
"""Run the four reference inputs through the full pipeline and print scores.
Manual M4 calibration aid: python scripts/calibrate.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signals.llm import llm_score
from signals.stylometry import stylometry_score
from scoring import combine, classify

CASES = {
    "clear_ai": "Artificial intelligence represents a transformative paradigm shift in "
        "modern society. It is important to note that while the benefits of AI are "
        "numerous, it is equally essential to consider the ethical implications.",
    "clear_human": "ok so i finally tried that new ramen place downtown and honestly? "
        "underwhelming. the broth was fine but they put WAY too much sodium in it and i "
        "was thirsty for like three hours after.",
    "formal_human": "The relationship between monetary policy and asset price inflation "
        "has been extensively studied in the literature. Central banks face a "
        "fundamental tension between price stability and prolonged low interest rates.",
    "edited_ai": "I've been thinking a lot about remote work lately. There are genuine "
        "tradeoffs — flexibility and no commute on one side, isolation and blurred "
        "work-life boundaries on the other.",
}

if __name__ == "__main__":
    print(f"{'case':<14}{'llm':>6}{'stylo':>8}{'conf':>7}  attribution")
    for name, text in CASES.items():
        llm = llm_score(text)
        stylo = stylometry_score(text)
        conf = combine(llm, stylo)
        print(f"{name:<14}{llm:>6.2f}{stylo:>8.2f}{conf:>7.2f}  {classify(conf)}")
```

- [ ] **Step 4: Run it manually and eyeball results**

Run: `python scripts/calibrate.py`
Expected: `clear_ai` lands `likely_ai` or high `uncertain`; `clear_human` lands `likely_human`; borderlines land `uncertain`. Record the printed table — it becomes the README "two example submissions" evidence. If results contradict intuition, adjust `config.py` thresholds and document the change in README spec-reflection.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate.py tests/test_calibration.py
git commit -m "feat: calibration script and reference-input ordering tests"
```

---

## Task 7: `/appeal` endpoint

**Files:**
- Modify: `app.py` (add route)
- Test: `tests/test_app.py` (add cases)

- [ ] **Step 1: Add failing tests to `tests/test_app.py`**

```python
def test_appeal_updates_status_and_logs(client):
    sub = client.post("/submit", json={"text": "my poem", "creator_id": "u1"}).get_json()
    cid = sub["content_id"]
    r = client.post("/appeal", json={"content_id": cid,
                                     "creator_reasoning": "I wrote this myself."})
    assert r.status_code == 200
    assert r.get_json()["status"] == "under_review"

    # status persisted
    rec = client.get(f"/content/{cid}").get_json()
    assert rec["status"] == "under_review"

    # appeal shows in the log with reasoning
    entries = client.get("/log").get_json()["entries"]
    appeal_entries = [e for e in entries if e["appeal_reasoning"]]
    assert any(e["content_id"] == cid for e in appeal_entries)


def test_appeal_unknown_id_is_404(client):
    r = client.post("/appeal", json={"content_id": "ghost",
                                     "creator_reasoning": "x"})
    assert r.status_code == 404
```

> Note: `test_appeal_updates_status_and_logs` also exercises `/content/<id>` (Task 8). If running tests before Task 8, expect the `/content` assertion to fail until Task 8 lands — implement Task 7 route first, then Task 8, then both pass.

- [ ] **Step 2: Run, verify the new appeal tests fail**

Run: `pytest tests/test_app.py -k appeal -v`
Expected: FAIL (404 route not found for `/appeal`).

- [ ] **Step 3: Add `/appeal` route to `app.py`** (insert before the `if __name__` block)

```python
@app.route("/appeal", methods=["POST"])
def appeal():
    body = request.get_json(silent=True) or {}
    content_id = (body.get("content_id") or "").strip()
    reasoning = (body.get("creator_reasoning") or "").strip()
    if not content_id or not reasoning:
        return jsonify({"error": "content_id and creator_reasoning are required"}), 400

    original = storage.get_content(content_id, DB_PATH)
    if original is None:
        return jsonify({"error": "content_id not found"}), 404

    storage.update_status(content_id, "under_review", reasoning, DB_PATH)
    storage.append_audit({
        "content_id": content_id, "creator_id": original["creator_id"],
        "attribution": original["attribution"], "confidence": original["confidence"],
        "llm_score": original["llm_score"],
        "stylometry_score": original["stylometry_score"],
        "status": "under_review", "appeal_reasoning": reasoning,
    }, DB_PATH)

    return jsonify({
        "content_id": content_id, "status": "under_review",
        "message": "Appeal received. This content is now under human review.",
    })
```

- [ ] **Step 4: Run appeal tests**

Run: `pytest tests/test_app.py -k appeal -v`
Expected: `test_appeal_unknown_id_is_404` PASS; `test_appeal_updates_status_and_logs` PASS after Task 8 (the `/content` call). Proceed to Task 8.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: /appeal endpoint updates status and logs alongside decision"
```

---

## Task 8: `GET /content/<id>` endpoint

**Files:**
- Modify: `app.py` (add route)
- Test: `tests/test_app.py` (add case)

- [ ] **Step 1: Add failing test**

```python
def test_content_lookup_and_404(client):
    sub = client.post("/submit", json={"text": "hi there", "creator_id": "u1"}).get_json()
    cid = sub["content_id"]
    ok = client.get(f"/content/{cid}")
    assert ok.status_code == 200
    assert ok.get_json()["content_id"] == cid
    assert client.get("/content/ghost").status_code == 404
```

- [ ] **Step 2: Run, verify fail**

Run: `pytest tests/test_app.py -k content -v`
Expected: FAIL (404 for the valid id — route not defined).

- [ ] **Step 3: Add `/content/<content_id>` route to `app.py`**

```python
@app.route("/content/<content_id>", methods=["GET"])
def content(content_id):
    rec = storage.get_content(content_id, DB_PATH)
    if rec is None:
        return jsonify({"error": "content_id not found"}), 404
    return jsonify(rec)
```

- [ ] **Step 4: Run full app suite**

Run: `pytest tests/test_app.py -v`
Expected: PASS (all app tests, including the appeal+content test from Task 7).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: GET /content/<id> inspection endpoint"
```

---

## Task 9: Rate-limit evidence + full suite green

**Files:**
- Test: full suite

- [ ] **Step 1: Run the whole test suite**

Run: `pytest -v`
Expected: PASS (all tests across all modules).

- [ ] **Step 2: Manual rate-limit test (real server)**

Start `python app.py`, then in another terminal:
```bash
for i in $(seq 1 12); do
  curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:5000/submit \
    -H "Content-Type: application/json" \
    -d '{"text": "This is a test submission for rate limit testing purposes only.", "creator_id": "ratelimit-test"}'
done
```
Expected: first 10 → `200`, last 2 → `429`. Copy this output block into the README (rate-limit evidence).

- [ ] **Step 3: Capture 3+ audit entries for README**

Run `curl -s http://localhost:5000/log | python -m json.tool` after at least 3 submissions and 1 appeal. Save ≥3 entries (including the appeal) for the README audit-log section.

- [ ] **Step 4: Commit any captured evidence (e.g. a `docs/evidence.md`)**

```bash
git add docs/evidence.md
git commit -m "docs: rate-limit and audit-log evidence captures"
```

---

## Task 10: README (Milestone 6)

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write the README** covering every required section. Pull content from `planning.md` and the captured evidence:
  - **Architecture overview** — the submission→label path (lift the narrative + diagram from planning.md).
  - **Detection signals** — what each measures, why chosen, blind spots (planning.md §1).
  - **Confidence scoring** — combination, asymmetric thresholds, validation, **two example submissions with actual differing scores** from the `scripts/calibrate.py` table.
  - **Transparency label** — all three variants verbatim (must match `scoring.py LABELS`).
  - **Rate limiting** — `10/min;100/day` + reasoning + the 429 output block.
  - **Known limitations** — edge cases A, B, D from planning.md §5, tied to signal properties.
  - **Spec reflection** — one way the spec helped; one divergence (e.g. stylometry threshold tuning during calibration).
  - **AI usage** — ≥2 concrete instances with what was directed / revised.
  - **Audit log** — ≥3 structured entries (incl. an appeal).

- [ ] **Step 2: Verify label text matches code**

Run: `python -c "from scoring import LABELS; [print(v) for v in LABELS.values()]"`
Confirm each printed string appears verbatim in `README.md`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: complete README with evidence, signals, scoring, limitations"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- Detection signals (§1) → Tasks 1, 3 ✓
- Uncertainty/scoring (§2) → Task 4 + config thresholds ✓
- Label variants (§3) → Task 4 (`LABELS`) + Task 10 ✓
- Appeals (§4) → Task 7 + storage `update_status` (Task 2) ✓
- Edge cases (§5) → stylometry neutral-on-short (Task 1), calibration (Task 6), README (Task 10) ✓
- Error handling → 400/404 (Tasks 5,7,8), LLM fallback (Task 3) ✓
- Rate limiting → Task 5 (decorator) + Task 9 (evidence) ✓
- Audit log → Task 2 + wired in Tasks 5,7; ≥3 entries Task 9 ✓
- API contract (4 endpoints) → /submit,/log (Task 5), /appeal (Task 7), /content (Task 8) ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. ✓

**Type consistency:** `db_path` arg present on all storage fns and used consistently in tests; `llm_score`/`stylometry_score`/`combine`/`classify`/`label_for` signatures match across scoring, app, and tests; record dict keys identical across storage, app, and audit. ✓

**Known cross-task dependency:** Task 7's combined test also asserts `/content` (Task 8). Flagged inline in Task 7 Step 1.
