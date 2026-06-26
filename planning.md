# Provenance Guard — Planning & Design Spec

**Project 4 (AI201).** Backend that classifies whether submitted text is human- or
AI-written, scores confidence honestly, surfaces a transparency label, and lets
creators appeal. Written **before** implementation; this document is the spec that
drives all code generation in Milestones 3–5.

**Stack:** Flask · Groq (`llama-3.3-70b-versatile`) · pure-Python stylometry ·
Flask-Limiter · SQLite (built-in).

---

## Architecture

### Narrative

A single piece of text enters through `POST /submit` with a `creator_id`. The request
first passes the **rate limiter** (10/min; 100/day per IP). The text is then scored by
**two independent detection signals**: an **LLM classifier** (Groq — judges semantic and
stylistic coherence holistically) and a **stylometric analyzer** (pure Python — measures
statistical structure). Each emits a probability the text is AI-generated, `P(AI) ∈ [0,1]`.
The **scoring module** combines them into one calibrated `confidence` score, maps that
score to one of three **attributions** (`likely_ai` / `uncertain` / `likely_human`), and
selects the matching reader-facing **transparency label**. The submission — id, both
signal scores, combined confidence, attribution, and status — is written to **SQLite** and
an **audit entry** is appended. The response returns `content_id`, `attribution`,
`confidence`, the `label` text, and both individual signal scores.

The **appeal flow** is separate: `POST /appeal` takes a `content_id` and the creator's
`creator_reasoning`, looks the submission up in SQLite, flips its `status` to
`under_review`, logs the appeal *alongside* the original decision in the audit log, and
returns a confirmation. No automated re-classification — a human reviewer would read the
appeal queue (surfaced via `GET /log`).

### Diagram

```
SUBMISSION FLOW
POST /submit {text, creator_id}
      │  rate limit: 10/min; 100/day per IP  ── exceeded ──► 429
      ▼
 ┌─ Signal 1: llm_score(text) ───────► llm    (0–1 P_AI)   [raw text → score]
 ├─ Signal 2: stylometry_score(text) ► stylo  (0–1 P_AI)   [raw text → score]
      ▼
 combine:  confidence = 0.5*llm + 0.5*stylo                [two scores → one score]
      ▼
 classify: ≥0.75 likely_ai │ 0.40–0.75 uncertain │ <0.40 likely_human
      ▼
 label_for(attribution) → reader-facing label text         [attribution → label]
      ▼
 SQLite: save_submission(id, creator_id, llm, stylo, confidence, attribution,
                         status='classified')  +  append_audit(...)
      ▼
 RESPONSE {content_id, attribution, confidence, label, llm_score, stylometry_score}

APPEAL FLOW
POST /appeal {content_id, creator_reasoning}
      ▼
 SQLite lookup by content_id  ── missing ──► 404
      ▼
 update_status(content_id, 'under_review', creator_reasoning)
      ▼
 append_audit(appeal entry, beside original decision)       [reasoning → log]
      ▼
 RESPONSE {content_id, status:'under_review', message}

GET /log         → recent audit entries (JSON)
GET /content/<id>→ one submission's current record (JSON)
```

### Components

| Module | Responsibility | Key interface |
|--------|----------------|---------------|
| `signals/llm.py` | LLM detection signal | `llm_score(text) -> float [0,1]` |
| `signals/stylometry.py` | Statistical detection signal | `stylometry_score(text) -> float [0,1]` |
| `scoring.py` | Combine + classify + label | `combine(llm, stylo)`, `classify(conf)`, `label_for(attr)` |
| `storage.py` | SQLite persistence + audit | `init_db`, `save_submission`, `get_content`, `update_status`, `append_audit`, `recent_log` |
| `app.py` | Flask routes + rate limiter | `/submit`, `/appeal`, `/log`, `/content/<id>` |

Each module is independently testable: signals take a string and return a float; scoring is
pure functions over floats; storage is CRUD over SQLite; `app.py` only wires them together.

---

## 1. Detection signals

Two signals that capture **genuinely different properties** — one semantic, one structural —
so that agreement is meaningful and disagreement signals uncertainty.

### Signal 1 — LLM classifier (Groq `llama-3.3-70b-versatile`)

- **Measures:** holistic semantic and stylistic coherence — whether the text *reads* as
  human or AI-generated, the way a careful human reader would judge.
- **Why it differs human↔AI:** AI prose tends toward even, hedged, "essay-shaped"
  coherence; human writing carries idiosyncratic voice, abrupt shifts, and intent the model
  can recognize.
- **Output:** a single `P(AI) ∈ [0,1]`. The model is prompted to return a structured
  verdict (label + a 0–1 score) which we parse to the float.
- **Blind spot:** it is a black box that can be *confidently wrong*, has no ground truth,
  and is weakest on lightly-edited AI text (the case most adversaries actually produce).

### Signal 2 — Stylometric analyzer (pure Python)

Combines **three** measurable metrics, each mapped to a 0–1 "AI-likeness" sub-score; the
signal output is their average.

- **Sentence-length variance** (std-dev of words per sentence): humans vary sentence length
  widely; AI is uniform. *Mapping (initial): std ≥ 8 → 0.0 (human), std ≤ 2 → 1.0 (AI),
  linear between.*
- **Type-token ratio** (unique words ÷ total words): vocabulary diversity. Weighted lightly
  because TTR mechanically drops as text gets longer; treated as a nudge, not a verdict.
- **Punctuation variety/density** (distinct punctuation types and marks per sentence):
  humans punctuate idiosyncratically; very regular, sparse punctuation reads AI-ward.
- **Why it differs human↔AI:** AI text is statistically *more uniform*; human writing is
  *more variable*. This is structural — independent of meaning.
- **Output:** a single `P(AI) ∈ [0,1]` (average of the three sub-scores).
- **Blind spot:** pure math, no understanding of meaning. A repetitive, simple-vocabulary
  *human* poem looks "AI-uniform"; very short text gives it too little data to be reliable.

**Why these two together:** one is semantic, one is structural — genuinely independent, so
the combination is more informative than either alone.

---

## 2. Uncertainty representation

- **What the score means:** one number, `confidence = P(text is AI-generated)`.
  `0.0` = certainly human, `1.0` = certainly AI. A single axis yields all three labels.
- **What 0.6 means:** "leaning AI but not confidently — mixed evidence." It deliberately
  lands in the **uncertain** band, not an AI verdict. The system refuses to brand text as AI
  on a coin-flip.
- **Combination:** `confidence = 0.5 * llm_score + 0.5 * stylometry_score`. Equal weight —
  neither signal is trusted over the other. A useful consequence: when the two signals
  **strongly disagree** (e.g. LLM 0.9, stylometry 0.2) the average (~0.55) falls into
  *uncertain* automatically — disagreement becomes uncertainty for free.
- **Thresholds (asymmetric, on purpose):**

  | `confidence` (P-AI) | attribution | label shown |
  |---|---|---|
  | **≥ 0.75** | `likely_ai` | "Likely AI-generated" |
  | **0.40 – 0.75** | `uncertain` | "Uncertain origin" |
  | **< 0.40** | `likely_human` | "Likely human-written" |

  **Why asymmetric (0.75 vs 0.40):** a false positive — labeling a real human's work as AI —
  is worse than a false negative on a writing platform; it publicly accuses a creator. So
  "Likely AI" is made *harder to reach* (needs ≥0.75) than "Likely human" (<0.40), widening
  the uncertain band on the AI side. The asymmetry **is** the false-positive-aversion,
  encoded numerically.
- **Calibration / validation:** thresholds and the stylometry metric mappings are tuned in
  Milestone 4 against four deliberately chosen inputs (clear-AI, clear-human, formal-human
  borderline, lightly-edited-AI borderline). Validation = each input lands in its intended
  band and the two clear cases are well-separated in score.

---

## 3. Transparency label design

Reader-facing, plain language, **words only** (no raw number shown — a "0.82" implies more
precision than detection has; the score still lives in the API response and audit log). The
AI label is **hedged, never accusatory**, because a false positive harms a real person.

**Variant A — `likely_ai` (confidence ≥ 0.75):**

> 🤖 **Likely AI-generated.** Our automated analysis suggests this text was probably produced
> with significant help from an AI tool. Automated detection is imperfect and *can be wrong* —
> if you're the creator and disagree, you can appeal this assessment for human review.

**Variant B — `uncertain` (0.40 ≤ confidence < 0.75):**

> ❓ **Uncertain origin.** Our analysis couldn't confidently tell whether a person or an AI
> wrote this; the signals were mixed. This is *not* a judgment against the creator — treat the
> authorship as simply unverified. Creators can request a review.

**Variant C — `likely_human` (confidence < 0.40):**

> ✍️ **Likely human-written.** Our analysis found the natural variation typical of human
> writing. This is an automated estimate, not a guarantee — and like every assessment, it can
> be appealed.

---

## 4. Appeals workflow

- **Who can appeal:** the creator of a submission, identified by holding its `content_id`
  (no auth layer in this scope — noted as a known simplification).
- **What they provide:** `content_id` + `creator_reasoning` (free text explaining why they
  believe the classification is wrong).
- **What the system does on appeal:**
  1. Look up the submission by `content_id` (404 if it does not exist).
  2. Update its `status` from `classified` → `under_review` and store the reasoning.
  3. Append an audit entry capturing the appeal *alongside* the original decision
     (original scores + attribution + the new reasoning), so a reviewer sees both.
  4. Return a confirmation `{content_id, status: "under_review", message}`.
- **No automated re-classification** (per brief).
- **What a human reviewer sees:** the appeal queue via `GET /log` — each appeal entry shows
  the original attribution/confidence/signal scores next to the creator's reasoning, enough
  to make a manual judgment.

---

## 5. Anticipated edge cases (known limitations)

Specific scenarios our exact signals handle poorly:

- **A. Repetitive, simple-vocabulary human poetry/verse.** Low type-token ratio + uniform
  sentence length is *precisely* the structural fingerprint our stylometry reads as "AI." A
  genuine human poet is pushed AI-ward → **false positive**. This is the direct blind spot of
  the stylometric signal, and the strongest reason the AI threshold is set high (0.75).
- **B. Very short submissions (1–3 sentences).** Stylometry needs several sentences to
  compute a meaningful length variance; on haiku-length input the variance is unstable and
  TTR ≈ 1.0. Both metrics fire on near-zero data, so the stylometry signal is unreliable and
  the system should lean toward `uncertain`.
- **D. Lightly-edited AI text.** A human passing AI output through light edits injects just
  enough variance to dodge the stylometry signal and enough natural phrasing to soften the
  LLM signal → **false negative**. This is the adversarial case detectors fail at in general,
  and we acknowledge it rather than pretend otherwise.

---

## Error handling & safety defaults

- **LLM call fails or response unparseable:** fall back toward **`uncertain`** (neutral
  ~0.5), never silently "human" or "AI." Failing toward an accusation (or a false clearance)
  is the dangerous direction; uncertain is the honest one.
- **Missing/empty `text` or `creator_id`:** `400 Bad Request`.
- **`/appeal` with unknown `content_id`:** `404 Not Found`.
- **Rate limit exceeded:** `429 Too Many Requests`.
- **Very short text:** stylometry returns a low-confidence neutral score (feeds edge case B).

## Rate limiting

`10 per minute; 100 per day` per IP on `/submit`. **Reasoning:** a real writer submits their
own work occasionally and almost never exceeds 10 pieces in a minute — that pace is a script,
not a person. The 100/day cap stops sustained abuse, detector reverse-engineering, and DoS
flooding, while a prolific legitimate writer stays comfortably under it. In-memory storage
(`storage_uri="memory://"`) for local dev.

## Audit log

Every decision and every appeal is a structured SQLite/JSON entry:

```json
{
  "content_id": "3f7a2b1e-...",
  "creator_id": "test-user-1",
  "timestamp": "2026-06-25T14:32:10.123Z",
  "attribution": "likely_ai",
  "confidence": 0.78,
  "llm_score": 0.81,
  "stylometry_score": 0.75,
  "status": "classified",
  "appeal_reasoning": null
}
```

Surfaced via `GET /log`. At least 3 entries (incl. ≥1 appeal) documented in the README.

---

## API surface (contract)

| Method | Path | Accepts | Returns |
|--------|------|---------|---------|
| POST | `/submit` | `{text, creator_id}` | `{content_id, attribution, confidence, label, llm_score, stylometry_score}` |
| POST | `/appeal` | `{content_id, creator_reasoning}` | `{content_id, status, message}` |
| GET | `/log` | — | `{entries: [...]}` |
| GET | `/content/<id>` | — | one submission record, or 404 |

---

## AI Tool Plan

How `planning.md` + the Architecture diagram drive code generation each milestone.

### M3 — submission endpoint + first signal
- **Spec provided to AI:** §1 (Detection signals, Signal 1) + Architecture diagram + API contract.
- **Ask for:** Flask `app.py` skeleton with the `POST /submit` route stub (hardcoded response
  first), and `signals/llm.py` (`llm_score`).
- **Verify:** call `llm_score` directly on the 4 test inputs and inspect; confirm the route
  shape matches the API contract and the function returns a float in [0,1] before wiring in.
  Add SQLite + `GET /log`.

### M4 — second signal + confidence scoring
- **Spec provided to AI:** §1 (Signal 2) + §2 (Uncertainty representation) + diagram.
- **Ask for:** `signals/stylometry.py` (`stylometry_score`) and `scoring.py`
  (`combine` / `classify`).
- **Verify:** confirm the generated thresholds match §2 *exactly* (AI tools drift here); run
  all 4 test inputs and check each lands in its intended band; print both signal scores
  separately when one misbehaves; tune metric mappings. Extend audit log to store both
  individual scores.

### M5 — production layer
- **Spec provided to AI:** §3 (Label variants) + §4 (Appeals workflow) + diagram.
- **Ask for:** `label_for(attribution)` mapping confidence→exact label text, and the
  `POST /appeal` endpoint.
- **Verify:** ask the AI to print all three label variants and diff against §3 verbatim;
  confirm `/appeal` flips status to `under_review` and writes an audit row; add Flask-Limiter
  (10/min;100/day) and `GET /content/<id>`; run the 12-rapid-request test → expect 429s.

---

## Build order

- **M3:** `app.py` skeleton + `/submit` (hardcoded) → `signals/llm.py` → wire in → SQLite + `/log`.
- **M4:** `signals/stylometry.py` → `scoring.py` → calibrate on 4 inputs → audit both scores.
- **M5:** `label_for` → `/appeal` → rate limiter → `/content/<id>` → README evidence.
- **M6:** README (all required sections) + short walkthrough video.

## Stretch (only after required features pass; update this doc before starting)

- Ensemble 3rd signal (burstiness / banned-AI-phrase) with documented weighting.
- "Verified human" provenance certificate.
- Analytics view (detection mix, appeal rate, + one metric).
- Multi-modal (image-description / metadata) second content type.
