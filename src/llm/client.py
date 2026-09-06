"""Small, local-only Ollama client wrapper."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

from src.runtime.context import effective_runtime_timeout, run_sync_operation

# The ollama package creates a module-level default client while importing.  That
# client reads proxy variables before our own ``trust_env=False`` client exists,
# and HTTPX rejects common desktop values such as ``socks://127.0.0.1:7890``.
# Temporarily hide proxies only for the import, then restore them for DDGS.
_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
_saved_proxy_environment = {
    key: os.environ.pop(key) for key in _PROXY_ENV_KEYS if key in os.environ
}
try:
    from ollama import Client
finally:
    os.environ.update(_saved_proxy_environment)
    del _saved_proxy_environment


class LLMError(RuntimeError):
    """Raised when the local Ollama model cannot serve a request."""


class LLMClient:
    """The only entry point for local Qwen calls in the application."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen3:8b",
        keep_alive: str | float | None = "0",
        timeout: float = 120.0,
    ) -> None:
        parsed = urlparse(host)
        if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("为避免使用云端 LLM，Ollama host 必须是本机地址")
        self.host = host.rstrip("/")
        self.model = model
        self.keep_alive = keep_alive
        self.timeout = timeout
        # Local requests must never be routed through HTTP_PROXY/HTTPS_PROXY.
        self._client = Client(host=self.host, trust_env=False, timeout=timeout)

    def check_health(self) -> None:
        """Verify that Ollama responds and the configured model is local."""
        try:
            response = self._client.list()
        except Exception as exc:
            raise LLMError(
                "无法连接本地 Ollama 服务。请检查 `systemctl status ollama`，"
                f"并确认 {self.host} 可访问。"
            ) from exc

        available = self._extract_model_names(response)
        if not self._model_is_available(available):
            raise LLMError(
                f"本地没有找到模型 {self.model}。请执行 `ollama pull {self.model}`。"
            )

    def chat(self, user_prompt: str, system_prompt: str | None = None) -> str:
        """Send one chat request to the configured local model."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        return run_sync_operation(
            "llm",
            f"ollama.chat:{self.model}",
            lambda: self._chat(messages),
        )

    def _chat(self, messages: list[dict[str, str]]) -> str:
        effective_timeout = effective_runtime_timeout(self.timeout)
        client = self._client
        temporary_client = None
        if effective_timeout < self.timeout:
            temporary_client = Client(
                host=self.host,
                trust_env=False,
                timeout=effective_timeout,
            )
            client = temporary_client
        try:
            response = client.chat(
                model=self.model,
                messages=messages,
                keep_alive=self.keep_alive,
            )
            content = self._extract_content(response).strip()
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(
                f"调用本地模型 {self.model} 失败。"
                "请确认 Ollama 正在运行且模型已下载。"
            ) from exc
        finally:
            if temporary_client is not None:
                temporary_client.close()

        if not content:
            raise LLMError(f"本地模型 {self.model} 返回了空响应。")
        return content

    @staticmethod
    def _extract_content(response: Any) -> str:
        message = getattr(response, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content
        if isinstance(response, Mapping):
            mapped_message = response.get("message")
            if isinstance(mapped_message, Mapping):
                content = mapped_message.get("content")
                if isinstance(content, str):
                    return content
        raise LLMError("Ollama 返回了无法识别的响应格式。")

    @staticmethod
    def _extract_model_names(response: Any) -> set[str]:
        models = getattr(response, "models", None)
        if models is None and isinstance(response, Mapping):
            models = response.get("models", [])
        if not isinstance(models, Sequence):
            return set()

        names: set[str] = set()
        for item in models:
            name = getattr(item, "model", None) or getattr(item, "name", None)
            if name is None and isinstance(item, Mapping):
                name = item.get("model") or item.get("name")
            if isinstance(name, str):
                names.add(name)
        return names

    def _model_is_available(self, names: set[str]) -> bool:
        requested = self.model.removesuffix(":latest")
        return any(name.removesuffix(":latest") == requested for name in names)
