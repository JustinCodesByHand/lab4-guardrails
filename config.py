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
