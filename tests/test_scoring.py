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
    assert classify(combine(0.9, 0.5)) == "uncertain"


def test_label_for_each_attribution():
    for attr in ("likely_ai", "uncertain", "likely_human"):
        assert label_for(attr) == LABELS[attr]
    assert "AI-generated" in LABELS["likely_ai"]
    assert "Uncertain" in LABELS["uncertain"]
    assert "human-written" in LABELS["likely_human"]
