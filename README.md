# Provenance Guard

Backend that classifies whether submitted text is human- or AI-written, scores confidence
honestly, surfaces a transparency label, and lets creators appeal a classification.

**Status:** design complete — see [`planning.md`](planning.md). Implementation in progress
(Milestones 3–5). This README will carry the required evidence (architecture overview,
detection signals, confidence scoring with two example scores, the three label variants,
rate-limit reasoning + 429 evidence, known limitations, spec reflection, AI usage).

## Stack

Flask · Groq (`llama-3.3-70b-versatile`) · pure-Python stylometry · Flask-Limiter · SQLite.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows (Git Bash)
pip install -r requirements.txt
cp .env.example .env                # then add your GROQ_API_KEY
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/submit` | Classify a text submission |
| POST | `/appeal` | Contest a classification |
| GET | `/log` | Recent audit-log entries |
| GET | `/content/<id>` | One submission's record |
