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
    max_pages: int = Field(default=3, ge=1, le=10)
    http_timeout: float = Field(default=10.0, gt=0, le=60)
    max_page_bytes: int = Field(default=2_000_000, ge=100_000, le=10_000_000)
    min_content_length: int = Field(default=200, ge=1, le=10_000)
    max_chars_per_document: int = Field(default=6_000, ge=100, le=50_000)
    max_total_context_chars: int = Field(default=15_000, ge=500, le=100_000)

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
            max_pages=os.getenv("MAX_PAGES", "3"),
            http_timeout=os.getenv("HTTP_TIMEOUT", "10"),
            max_page_bytes=os.getenv("MAX_PAGE_BYTES", "2000000"),
            min_content_length=os.getenv("MIN_CONTENT_LENGTH", "200"),
            max_chars_per_document=os.getenv("MAX_CHARS_PER_DOCUMENT", "6000"),
            max_total_context_chars=os.getenv("MAX_TOTAL_CONTEXT_CHARS", "15000"),
        )
