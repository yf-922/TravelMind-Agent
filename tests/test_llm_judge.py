from app.evaluation import llm_judge


def test_judge_uses_versioned_prompt_and_structured_schema(monkeypatch):
    captured = {}

    class FakeJudge:
        pass

    expected = llm_judge.JudgeScore(
        groundedness=5, constraint_satisfaction=4, route_completeness=5,
        reviewer_consistency=5, clarity=4, pass_case=True, reason="行程景点均来自候选池。",
    )
    monkeypatch.setattr(llm_judge, "build_structured_llm", lambda *args, **kwargs: FakeJudge())
    def fake_invoke(_llm, messages):
        captured["messages"] = messages
        return expected
    monkeypatch.setattr(llm_judge, "invoke_structured", fake_invoke)
    score = llm_judge.judge_itinerary(
        user_request="北京一日游", destination="北京", candidates=[{"name": "故宫"}],
        itinerary=[{"name": "故宫"}], review={"approved": True},
    )
    assert score.pass_case is True
    assert "候选池外景点" in captured["messages"][0][1]
    assert llm_judge.JUDGE_PROMPT_VERSION == "travelmind-judge-v1"
