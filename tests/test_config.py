import pytest
from pydantic import ValidationError

from src.config import Settings


def test_settings_accept_local_ollama() -> None:
    configured = Settings(ollama_host="http://127.0.0.1:11434/")
    assert configured.ollama_host == "http://127.0.0.1:11434"


def test_settings_reject_remote_llm_host() -> None:
    with pytest.raises(ValidationError, match="本机"):
        Settings(ollama_host="https://cloud.example.com")


def test_settings_validate_search_limit() -> None:
    with pytest.raises(ValidationError):
        Settings(search_max_results=20)

