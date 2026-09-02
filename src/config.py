"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Settings(BaseModel):
    """Runtime settings for the local model and search provider."""

    model_config = ConfigDict(frozen=True)

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    search_max_results: int = Field(default=5, ge=1, le=10)

    @field_validator("ollama_host")
    @classmethod
    def require_local_ollama(cls, value: str) -> str:
        host = value.strip().rstrip("/")
        parsed = urlparse(host)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise ValueError("OLLAMA_HOST 必须指向本机 localhost 或回环地址")
        return host

    @field_validator("ollama_model")
    @classmethod
    def require_model_name(cls, value: str) -> str:
        model = value.strip()
        if not model:
            raise ValueError("OLLAMA_MODEL 不能为空")
        return model

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
            search_max_results=os.getenv("SEARCH_MAX_RESULTS", "5"),
        )
