from signals.stylometry import (
    sentence_length_subscore,
    type_token_ratio_subscore,
    punctuation_subscore,
    stylometry_score,
)
from tests.conftest import CLEAR_AI, CLEAR_HUMAN


def test_uniform_sentences_score_ai_ward():
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
    assert sentence_length_subscore("Just one short line.") == 0.5


def test_ttr_subscore_in_range():
    assert 0.0 <= type_token_ratio_subscore(CLEAR_HUMAN) <= 1.0


def test_punctuation_variety_low_is_ai_ward():
    plain = "the cat sat on the mat the dog ran the bird flew the fish swam"
    varied = 'Wait—really? Yes; I think so... "maybe," she said (laughing).'
    assert punctuation_subscore(plain) > punctuation_subscore(varied)


def test_stylometry_score_separates_clear_cases():
    assert stylometry_score(CLEAR_AI) > stylometry_score(CLEAR_HUMAN)


def test_stylometry_score_bounded():
    for t in (CLEAR_AI, CLEAR_HUMAN, "tiny."):
        assert 0.0 <= stylometry_score(t) <= 1.0
