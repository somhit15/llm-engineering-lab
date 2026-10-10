"""Tests for configuration that do not make live provider calls."""

import pytest

from langchain_learning.config import get_model_settings


def test_defaults_to_gemini_when_environment_is_unset(monkeypatch):
    monkeypatch.delenv("LC_PROVIDER", raising=False)
    monkeypatch.delenv("LC_MODEL", raising=False)

    settings = get_model_settings()

    assert settings.provider == "google_genai"
    assert settings.model == "gemini-2.5-flash"


def test_reads_nvidia_configuration(monkeypatch):
    monkeypatch.setenv("LC_PROVIDER", "nvidia")
    monkeypatch.setenv("LC_MODEL", "example-nvidia-model")

    settings = get_model_settings()

    assert settings.provider == "nvidia"
    assert settings.model == "example-nvidia-model"


def test_provider_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("LC_PROVIDER", "NVIDIA")
    monkeypatch.setenv("LC_MODEL", "example-model")

    assert get_model_settings().provider == "nvidia"


def test_rejects_unknown_provider(monkeypatch):
    monkeypatch.setenv("LC_PROVIDER", "unknown-provider")

    with pytest.raises(ValueError, match="Unsupported LC_PROVIDER"):
        get_model_settings()


def test_rejects_empty_model(monkeypatch):
    monkeypatch.setenv("LC_PROVIDER", "nvidia")
    monkeypatch.setenv("LC_MODEL", "   ")

    with pytest.raises(ValueError, match="LC_MODEL must not be empty"):
        get_model_settings()
