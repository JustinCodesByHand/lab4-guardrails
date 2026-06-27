# Provenance Guard

A backend any creative-sharing platform can plug into to classify whether submitted
text was written by a human or an AI, score how confident that judgment is, surface a
plain-language transparency label, and let creators appeal a classification — all while
recording every decision to a structured audit log.

Built on the principle that **a false positive (calling a human's work AI) is worse than
a false negative**, so the system is deliberately reluctant to brand text as AI and gives
creators an appeal path when it gets things wrong.

**Stack:** Flask · Groq (`llama-3.3-70b-versatile`) · pure-Python stylometry ·
Flask-Limiter · SQLite. Full design rationale in [`planning.md`](planning.md).

---

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate          # Windows (Git Bash);  use .venv/bin/activate on Mac/Linux
pip install -r requirements.txt
cp .env.example .env                    # then add your GROQ_API_KEY
python app.py                           # serves on http://localhost:5000
```

Run the tests (no API key needed — the network signal is mocked):

```bash
.venv/Scripts/python -m pytest -q       # 31 passing
```

---

## Architecture overview — the path a submission takes

```
POST /submit {text, creator_id}
      │  rate limit: 10/min; 100/day per IP  ── exceeded ──► 429
      ▼
 ┌─ Signal 1: llm_score(text) ───────► P(AI)  (Groq, semantic)
 ├─ Signal 2: stylometry_score(text) ► P(AI)  (pure Python, structural)
      ▼
 combine:  confidence = 0.5*llm + 0.5*stylo
      ▼
 classify: ≥0.75 likely_ai │ 0.40–0.75 uncertain │ <0.40 likely_human
      ▼
 label_for(attribution) → reader-facing transparency label
      ▼
 SQLite: save submission (status=classified) + append audit entry
      ▼
 response {content_id, attribution, confidence, label, llm_score, stylometry_score}
```

A submission enters `POST /submit`, passes the rate limiter, and is scored by two
independent detection signals. Their scores are combined into one confidence value,
mapped to one of three attributions, and turned into a transparency label. The result is
persisted to SQLite and written to the audit log before the response returns. A separate
`POST /appeal` flow lets a creator contest a result: it flips the stored status to
`under_review` and logs the appeal next to the original decision for a human reviewer.

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/submit` | Classify a text submission (rate-limited) |
| POST | `/appeal` | Contest a classification; sets status `under_review` |
| GET | `/log` | Recent audit-log entries |
| GET | `/content/<id>` | One submission's current record |

---

## Detection signals

The pipeline uses **two signals that capture genuinely different properties** — one
semantic, one structural — so that agreement is meaningful and disagreement signals
uncertainty.

### Signal 1 — LLM classifier (Groq `llama-3.3-70b-versatile`)
- **Measures:** holistic semantic/stylistic coherence — whether the text *reads* as human
  or AI, the way a careful human reader would judge it.
- **Why it chosen:** captures intent, voice, and idiosyncrasy that no simple statistic can.
- **What it misses:** it is a black box that can be *confidently wrong*, has no ground
  truth, and is weakest on lightly-edited AI text — the case adversaries actually produce.
  It also tends to rate *formal* human writing as AI-like (demonstrated below).

### Signal 2 — Stylometric analyzer (pure Python, no network)
Combines three structural metrics, each mapped to a 0–1 "AI-likeness" sub-score and
averaged with weights (sentence-length 0.45, punctuation 0.35, type-token ratio 0.20 —
TTR is weighted lightly because it drifts with text length):
- **Sentence-length variance** — humans vary sentence length; AI is uniform.
- **Type-token ratio** — vocabulary diversity; low diversity nudges AI-ward.
- **Punctuation variety** — sparse, regular punctuation reads AI-ward.
- **Why chosen:** statistical structure is independent of meaning, so it is a true second
  opinion rather than an echo of the LLM.
- **What it misses:** pure math, no understanding. A repetitive, simple-vocabulary *human*
  poem looks "AI-uniform"; very short text gives it too little data to be reliable.

---

## Confidence scoring

- **The score is a single number, `confidence = P(text is AI-generated)`** (0 = certainly
  human, 1 = certainly AI). One axis yields all three labels.
- **Combination:** `confidence = 0.5 * llm_score + 0.5 * stylometry_score`. Equal weight —
  neither signal is trusted over the other. A deliberate consequence: when the two signals
  **strongly disagree**, the average lands near the middle and the result is `uncertain` —
  disagreement becomes honest uncertainty for free.
- **Asymmetric thresholds (the false-positive defense):** `likely_ai` requires
  `≥ 0.75`, but `likely_human` only requires `< 0.40`. Calling a human's work AI is the
  more harmful error, so AI is made *harder to reach*. The asymmetry **is** the
  false-positive aversion, encoded numerically.

### Validating that the score is meaningful

Scores were checked against deliberately chosen inputs via `scripts/calibrate.py` (real
Groq + real stylometry). Clearly different inputs produce clearly different scores that
land in different bands:

| Input | llm | stylo | **confidence** | label |
|-------|-----|-------|----------------|-------|
| Casual human restaurant review | 0.20 | 0.21 | **0.21** | ✍️ likely_human |
| Polished AI marketing prose | 0.80 | 0.36 | **0.58** | ❓ uncertain |
| Formal academic human prose | 0.80 | 0.73 | **0.77** | 🤖 likely_ai |
| Lightly-edited AI text | 0.20 | 0.23 | **0.21** | ✍️ likely_human |

**Two example submissions with noticeably different confidence:**
- **Lower-confidence / human:** the casual review scored **0.21** → confidently
  `likely_human`.
- **Higher-confidence / AI-leaning:** the formal academic prose scored **0.77** →
  `likely_ai`.

The 0.56-point spread between them, and the fact that all three bands are reachable,
shows the score produces real variation rather than a constant. The results also surface
the system's honest failure modes (see Known Limitations) — clearly-AI marketing copy
only reached `uncertain`, because the 50/50 blend plus the high AI bar deliberately
resists AI verdicts.

---

## Transparency label

Reader-facing, plain-language, **words only** (no raw number — a "0.82" implies more
precision than detection has; the score still lives in the API response and audit log).
The AI label is **hedged, never accusatory**, because a false positive harms a real
person. The three variants, verbatim:

**Likely AI-generated** (confidence ≥ 0.75):
> 🤖 Likely AI-generated. Our automated analysis suggests this text was probably produced with significant help from an AI tool. Automated detection is imperfect and can be wrong — if you're the creator and disagree, you can appeal this assessment for human review.

**Uncertain** (0.40 ≤ confidence < 0.75):
> ❓ Uncertain origin. Our analysis couldn't confidently tell whether a person or an AI wrote this; the signals were mixed. This is not a judgment against the creator — treat the authorship as simply unverified. Creators can request a review.

**Likely human-written** (confidence < 0.40):
> ✍️ Likely human-written. Our analysis found the natural variation typical of human writing. This is an automated estimate, not a guarantee — and like every assessment, it can be appealed.

---

## Appeals workflow

`POST /appeal` accepts a `content_id` and `creator_reasoning`. It looks the submission up
(404 if unknown), updates its status `classified → under_review`, and appends an audit
entry that carries the original decision's scores **plus** the creator's reasoning, so a
human reviewer sees the appeal beside the original verdict. No automated re-classification.

```json
POST /appeal → {
  "content_id": "d8494d06-…",
  "status": "under_review",
  "message": "Appeal received. This content is now under human review."
}
```

---

## Rate limiting

**`10 per minute; 100 per day` per IP** on `/submit` (in-memory store for local dev).

**Reasoning:** a real writer submits their own work occasionally and almost never exceeds
10 pieces in a minute — that pace is a script, not a person. The 100/day cap stops
sustained abuse, detector reverse-engineering, and flooding, while a prolific legitimate
writer stays comfortably under it.

**Evidence** — 12 rapid requests from one IP (first 10 accepted, rest rejected):

```
request 1 -> 200      request 7  -> 200
request 2 -> 200      request 8  -> 200
request 3 -> 200      request 9  -> 200
request 4 -> 200      request 10 -> 200
request 5 -> 200      request 11 -> 429
request 6 -> 200      request 12 -> 429
```

---

## Audit log

Every decision and every appeal is a structured SQLite row, surfaced via `GET /log`
(newest first). Entry `id:11` is an appeal (`status: under_review`, `appeal_reasoning`
populated); `id:10` is the original classification for the same `content_id` — the
reviewer sees both.

```json
[
  {
    "id": 11, "content_id": "d8494d06-…", "creator_id": "ratelimit-test",
    "timestamp": "2026-06-26T16:38:03.219593Z", "attribution": "uncertain",
    "confidence": 0.6525, "llm_score": 0.8, "stylometry_score": 0.505,
    "status": "under_review",
    "appeal_reasoning": "I am a non-native English speaker; my formal style is my own, not AI."
  },
  {
    "id": 10, "content_id": "d8494d06-…", "creator_id": "ratelimit-test",
    "timestamp": "2026-06-26T16:36:27.309122Z", "attribution": "uncertain",
    "confidence": 0.6525, "llm_score": 0.8, "stylometry_score": 0.505,
    "status": "classified", "appeal_reasoning": null
  },
  {
    "id": 9, "content_id": "edeeaabf-…", "creator_id": "ratelimit-test",
    "timestamp": "2026-06-26T16:36:26.683367Z", "attribution": "uncertain",
    "confidence": 0.7025, "llm_score": 0.9, "stylometry_score": 0.505,
    "status": "classified", "appeal_reasoning": null
  }
]
```

(Identical submission text scored `llm 0.8` on one call and `0.9` on another — a real
demonstration of LLM nondeterminism, and part of why no single signal is trusted alone.)
Full captures in [`docs/evidence.md`](docs/evidence.md).

---

## Known limitations

These are specific content types the system gets wrong, tied to properties of the signals —
several were demonstrated live during calibration:

- **Formal / non-native human writing → false positive (demonstrated).** Formal academic
  prose scored 0.77 → `likely_ai`. The LLM *and* stylometry independently read its uniform,
  careful structure as AI. This is the most harmful error type (accusing a human), and **no
  reweighting fixes it** — both signals genuinely misjudge this text. It is exactly why the
  AI threshold is set high and why appeals exist; the appeal evidence above is this case.
- **Repetitive, simple-vocabulary human poetry.** Low type-token ratio + uniform sentence
  length is precisely the structural fingerprint the stylometric signal reads as "AI," so a
  genuine human poet is pushed AI-ward.
- **Very short submissions (1–3 sentences).** Stylometry needs several sentences for a
  meaningful length variance; on haiku-length input it returns a neutral 0.5, so the verdict
  rests on the LLM alone.
- **Lightly-edited AI text → false negative (demonstrated).** Scored 0.21 → `likely_human`.
  Light human editing injects enough variance to dodge stylometry and enough natural phrasing
  to soften the LLM. This is the adversarial case detectors fail at in general; we acknowledge
  it rather than pretend otherwise.

Perfect AI detection is unsolved. The engineering goal here is to **communicate uncertainty
honestly and give creators a path to appeal**, not to pretend to a precision the signals
don't have.

---

## Spec reflection

- **How the spec helped:** writing `planning.md` *before* code meant the contested
  decisions were already settled — the exact thresholds (0.75/0.40), the 50/50 weighting,
  and the verbatim label text. Implementation and its tests were unambiguous: `classify()`
  and `LABELS` were transcribed straight from the spec, and the asymmetric-threshold
  rationale directly justified the boundary values rather than being reverse-rationalized.
- **How implementation diverged:** the spec described the stylometric signal as the
  "average of three sub-scores," but implementation used a *weighted* average
  (0.45/0.35/0.20) to honor the spec's own note that TTR should be "weighted lightly" — the
  intent was in the spec, the precise mechanism was decided at build time. A second, smaller
  divergence: `storage.update_status()` originally took a `reasoning` argument, dropped after
  review because the reasoning belongs in the audit log, not the submissions row.

---

## AI usage

This project was built spec-first: `planning.md` and the architecture diagram drove an AI
coding tool to generate each module, which was then reviewed and revised. Two concrete
instances:

1. **Stylometry signal — directed generation, then a review caught a real bug.** I directed
   the AI to implement the three stylometric sub-scores from the spec. A code-review pass
   found that the punctuation set listed both `"."` and `"..."`, so any text containing an
   ellipsis double-counted its punctuation variety — skewing the score for exactly the
   informal/human writing the system must not misjudge. I had it removed (`"."` already
   covers ellipses) and added a neutral-on-empty-input guard.
2. **Flask pipeline — overrode weak test coverage the AI produced.** The AI's `/submit`
   tests mocked *both* signals, so the real (deterministic) stylometry never actually flowed
   through the endpoint — the tests proved only that constants plumbed through. I overrode
   this by adding a test that stubs only the network LLM signal and asserts the genuine
   stylometry score reaches the response, catching real wiring regressions.

In both cases the AI produced a working first draft quickly, but reviewing rather than
trusting it caught a correctness bug and a testing blind spot before they shipped.
