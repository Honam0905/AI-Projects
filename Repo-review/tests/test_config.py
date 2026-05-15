"""Configuration tests."""

from pathlib import Path

from agent_swarms.config import DEFAULT_SANDBOX_DATA_DIR, get_settings


def test_default_sandbox_data_dir_lives_outside_repo(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_SWARMS_SANDBOX_DATA_DIR", raising=False)
    get_settings.cache_clear()

    settings = get_settings()

    assert Path(settings.sandbox_data_dir) == DEFAULT_SANDBOX_DATA_DIR
    assert Path(settings.sandbox_data_dir).is_absolute()
    assert settings.sandbox_data_dir.endswith(".agent_swarms_data")
