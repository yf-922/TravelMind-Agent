from app.planning.graph import _stage_summary


def test_attraction_summary_reports_candidate_count():
    summary = _stage_summary(
        "attraction_search",
        {"destination": "重庆"},
        {"pois": [{"name": "洪崖洞"}, {"name": "解放碑"}]},
    )
    assert "2" in summary
    assert "候选景点池" in summary


def test_planner_summary_uses_structured_route_not_reasoning():
    summary = _stage_summary(
        "planner",
        {"days": 2, "review_round": 0},
        {
            "review_round": 1,
            "route": [{"spots": [{"name": "A"}, {"name": "B"}]}],
            "reasoning": "this private model reasoning must never be displayed",
        },
    )
    assert "第 1 轮" in summary
    assert "2 个景点" in summary
    assert "private model reasoning" not in summary


def test_reviewer_summary_distinguishes_approval_and_rework():
    assert "通过" in _stage_summary("reviewer", {}, {"approved": True})
    summary = _stage_summary("reviewer", {}, {"approved": False, "reviewer_issues": [1, 2, 3]})
    assert "3" in summary
