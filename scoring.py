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
