"""Unit tests for the application settings."""

from app.core.config import Settings


def test_defaults():
    settings = Settings(_env_file=None)
    assert settings.qdrant_collection == "sepa_rulebook"
    assert settings.qdrant_url == "http://localhost:6333"


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("QDRANT_COLLECTION", "other")
    assert Settings(_env_file=None).qdrant_collection == "other"
