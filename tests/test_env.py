from pathlib import Path


def test_local_env_duplicate_keys_use_last_file_value(monkeypatch, tmp_path):
    from app.core import env

    env_file = tmp_path / ".env.local"
    env_file.write_text("LLM_PROVIDER=grok\nLLM_PROVIDER=openai\n", encoding="utf-8")
    monkeypatch.setattr(env, "LOCAL_ENV_FILE", Path(env_file))
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    env.load_local_env()
    assert env.os.getenv("LLM_PROVIDER") == "openai"
