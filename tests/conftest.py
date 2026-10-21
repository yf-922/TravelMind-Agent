"""Explicit isolation for tests that construct offline Supervisor nodes."""

import os

import pytest


@pytest.fixture
def offline_supervisor_models(monkeypatch, tmp_path):
    from app.core import env
    from app.multi_agent_core import modification
    from app.planning import nodes

    # Reproduce a clean CI checkout, even on a developer machine with keys.
    monkeypatch.setattr(env, "LOCAL_ENV_FILE", tmp_path / "missing.env")
    for name in list(os.environ):
        if name.endswith("_API_KEY"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")

    def fake_model(*args, **kwargs):
        return object()

    # Patch where each factory is looked up, including modification analysis.
    monkeypatch.setattr(nodes, "build_structured_llm", fake_model)
    monkeypatch.setattr(modification, "build_structured_llm", fake_model)
