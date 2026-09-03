"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Settings(BaseModel):
    """Runtime settings for local inference, broad search, and web reading."""

    model_config = ConfigDict(frozen=True)

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    max_search_queries: int = Field(default=3, ge=1, le=10)
    results_per_query_per_provider: int = Field(default=5, ge=1, le=10)
    max_combined_search_results: int = Field(default=20, ge=1, le=100)
    max_pages_to_read: int = Field(default=5, ge=1, le=10)
    max_search_concurrency: int = Field(default=5, ge=1, le=20)
    search_timeout: float = Field(default=10.0, gt=0, le=60)
    allowed_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()
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

    @field_validator("allowed_domains", "blocked_domains", mode="before")
    @classmethod
    def normalize_domains(cls, value: object) -> tuple[str, ...]:
        if isinstance(value, str):
            items = value.split(",")
        elif isinstance(value, (list, tuple, set)):
            items = value
        else:
            raise ValueError("域名配置必须是列表或逗号分隔字符串")

        domains: list[str] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, str):
                raise ValueError("域名必须是字符串")
            domain = item.strip().lower().strip(".")
            if not domain:
                continue
            if "://" in domain or "/" in domain or any(
                char.isspace() for char in domain
            ):
                raise ValueError(f"域名规则格式无效：{item}")
            if domain not in seen:
                seen.add(domain)
                domains.append(domain)
        return tuple(domains)

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
            max_search_queries=os.getenv("MAX_SEARCH_QUERIES", "3"),
            results_per_query_per_provider=os.getenv(
                "RESULTS_PER_QUERY_PER_PROVIDER",
                os.getenv("SEARCH_MAX_RESULTS", "5"),
            ),
            max_combined_search_results=os.getenv(
                "MAX_COMBINED_SEARCH_RESULTS", "20"
            ),
            max_pages_to_read=os.getenv(
                "MAX_PAGES_TO_READ", os.getenv("MAX_PAGES", "5")
            ),
            max_search_concurrency=os.getenv("MAX_SEARCH_CONCURRENCY", "5"),
            search_timeout=os.getenv("SEARCH_TIMEOUT", "10"),
            allowed_domains=os.getenv("ALLOWED_DOMAINS", ""),
            blocked_domains=os.getenv("BLOCKED_DOMAINS", ""),
            http_timeout=os.getenv("HTTP_TIMEOUT", "10"),
            max_page_bytes=os.getenv("MAX_PAGE_BYTES", "2000000"),
            min_content_length=os.getenv("MIN_CONTENT_LENGTH", "200"),
            max_chars_per_document=os.getenv("MAX_CHARS_PER_DOCUMENT", "6000"),
            max_total_context_chars=os.getenv("MAX_TOTAL_CONTEXT_CHARS", "15000"),
        )
