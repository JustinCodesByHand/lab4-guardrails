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
