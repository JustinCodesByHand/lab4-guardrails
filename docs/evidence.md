# Captured Evidence

Real runs against the running Flask app (`python app.py`) with a live Groq key.
Used to populate the README. Regenerate any time with the commands shown.

---

## Rate limiting (`10 per minute; 100 per day` per IP)

12 rapid `POST /submit` requests from one IP — first 10 accepted, rest rejected:

```
request 1 -> 200
request 2 -> 200
request 3 -> 200
request 4 -> 200
request 5 -> 200
request 6 -> 200
request 7 -> 200
request 8 -> 200
request 9 -> 200
request 10 -> 200
request 11 -> 429
request 12 -> 429
```

Command:
```bash
for i in $(seq 1 12); do
  curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:5000/submit \
    -H "Content-Type: application/json" \
    -d '{"text": "This is a test submission for rate limit testing purposes only.", "creator_id": "ratelimit-test"}'
done
```

---

## Confidence-score variation (full pipeline, real Groq)

From `scripts/calibrate.py` — clearly different inputs produce clearly different
scores and reach different bands:

```
case             llm   stylo   conf  attribution
clear_ai        0.80    0.36   0.58  uncertain
clear_human     0.20    0.21   0.21  likely_human
formal_human    0.80    0.73   0.77  likely_ai
edited_ai       0.20    0.23   0.21  likely_human
```

- **High-confidence-human example:** casual review text → `conf 0.21` → **likely_human**.
- **Higher-AI example:** formal academic prose → `conf 0.77` → **likely_ai**.
- Notable honest results: clearly-AI marketing prose only reached `uncertain` (0.58)
  because the 50/50 blend + asymmetric 0.75 bar deliberately resists AI verdicts;
  formal *human* writing was a false positive (both signals independently read it as
  AI) — a documented known limitation.

---

## Appeals workflow

`POST /appeal` flips status and logs the appeal beside the original decision:

```json
{
    "content_id": "d8494d06-8d67-4b3f-8d0f-4814fd07acea",
    "message": "Appeal received. This content is now under human review.",
    "status": "under_review"
}
```

`GET /content/<id>` after the appeal confirms the persisted status change:

```json
{
    "attribution": "uncertain",
    "confidence": 0.6525,
    "content_id": "d8494d06-8d67-4b3f-8d0f-4814fd07acea",
    "creator_id": "ratelimit-test",
    "llm_score": 0.8,
    "status": "under_review",
    "stylometry_score": 0.505,
    "text": "This is a test submission for rate limit testing purposes only."
}
```

---

## Audit log (`GET /log`) — appeal logged alongside the original decision

Newest first. Entry `id:11` is the appeal (`status: under_review`, `appeal_reasoning`
populated); entry `id:10` is the original classification (`status: classified`) for the
same `content_id` — the reviewer sees both side by side.

```json
[
  {
    "id": 11,
    "content_id": "d8494d06-8d67-4b3f-8d0f-4814fd07acea",
    "creator_id": "ratelimit-test",
    "timestamp": "2026-06-26T16:38:03.219593Z",
    "attribution": "uncertain",
    "confidence": 0.6525,
    "llm_score": 0.8,
    "stylometry_score": 0.505,
    "status": "under_review",
    "appeal_reasoning": "I am a non-native English speaker; my formal style is my own, not AI."
  },
  {
    "id": 10,
    "content_id": "d8494d06-8d67-4b3f-8d0f-4814fd07acea",
    "creator_id": "ratelimit-test",
    "timestamp": "2026-06-26T16:36:27.309122Z",
    "attribution": "uncertain",
    "confidence": 0.6525,
    "llm_score": 0.8,
    "stylometry_score": 0.505,
    "status": "classified",
    "appeal_reasoning": null
  },
  {
    "id": 9,
    "content_id": "edeeaabf-d179-4fff-952b-46bda21a325a",
    "creator_id": "ratelimit-test",
    "timestamp": "2026-06-26T16:36:26.683367Z",
    "attribution": "uncertain",
    "confidence": 0.7025,
    "llm_score": 0.9,
    "stylometry_score": 0.505,
    "status": "classified",
    "appeal_reasoning": null
  }
]
```

(Identical submission text scored `llm 0.8` on one call and `0.9` on another — a real
demonstration of LLM nondeterminism, and part of why a single signal isn't trusted alone.)
