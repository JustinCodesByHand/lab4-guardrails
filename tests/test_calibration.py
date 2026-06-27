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
    assert stylometry_score(BORDERLINE_FORMAL_HUMAN) > stylometry_score(CLEAR_HUMAN)
