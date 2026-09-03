import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.llm.client import LLMClient, LLMError


class FakeOllama:
    def __init__(self, *, models=None, response=None, error=None) -> None:
        self.models = models if models is not None else []
        self.response = response
        self.error = error
        self.chat_arguments = None

    def list(self):
        if self.error:
            raise self.error
        return SimpleNamespace(models=self.models)

    def chat(self, **kwargs):
        self.chat_arguments = kwargs
        if self.error:
            raise self.error
        return self.response


def make_client(fake: FakeOllama) -> LLMClient:
    client = LLMClient("http://localhost:11434", "qwen3:8b")
    client._client = fake
    return client


def test_llm_client_rejects_remote_host() -> None:
    with pytest.raises(ValueError, match="本机"):
        LLMClient("https://llm.example.com", "qwen3:8b")


def test_llm_client_defaults_to_local_qwen3() -> None:
    client = LLMClient()
    assert client.host == "http://localhost:11434"
    assert client.model == "qwen3:8b"


def test_import_ignores_invalid_socks_proxy_and_restores_environment() -> None:
    proxy = "socks://127.0.0.1:7890"
    environment = os.environ.copy()
    environment["ALL_PROXY"] = proxy
    environment["all_proxy"] = proxy
    project_root = Path(__file__).resolve().parents[1]
    code = (
        "import os; "
        "from src.llm.client import LLMClient; "
        "client = LLMClient(); "
        "print(client.host, client.model, os.environ['ALL_PROXY'])"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == (
        "http://localhost:11434 qwen3:8b socks://127.0.0.1:7890"
    )


def test_health_accepts_configured_local_model() -> None:
    fake = FakeOllama(models=[SimpleNamespace(model="qwen3:8b")])
    make_client(fake).check_health()


def test_health_accepts_latest_alias() -> None:
    fake = FakeOllama(models=[{"name": "qwen3:8b"}])
    client = LLMClient("http://127.0.0.1:11434", "qwen3:8b:latest")
    client._client = fake
    client.check_health()


def test_health_reports_missing_model() -> None:
    fake = FakeOllama(models=[SimpleNamespace(model="other:latest")])
    with pytest.raises(LLMError, match="ollama pull qwen3:8b"):
        make_client(fake).check_health()


def test_health_wraps_connection_error() -> None:
    fake = FakeOllama(error=ConnectionError("refused"))
    with pytest.raises(LLMError, match="无法连接本地 Ollama"):
        make_client(fake).check_health()


def test_chat_sends_system_and_user_messages() -> None:
    fake = FakeOllama(
        response=SimpleNamespace(message=SimpleNamespace(content=" local answer "))
    )
    client = make_client(fake)

    answer = client.chat("question", "system")

    assert answer == "local answer"
    assert fake.chat_arguments == {
        "model": "qwen3:8b",
        "keep_alive": "0",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ],
    }


def test_chat_supports_mapping_response() -> None:
    fake = FakeOllama(response={"message": {"content": "answer"}})
    assert make_client(fake).chat("question") == "answer"


def test_chat_rejects_empty_response() -> None:
    fake = FakeOllama(response={"message": {"content": "  "}})
    with pytest.raises(LLMError, match="空响应"):
        make_client(fake).chat("question")


def test_chat_wraps_ollama_failure() -> None:
    fake = FakeOllama(error=RuntimeError("inference failed"))
    with pytest.raises(LLMError, match="调用本地模型"):
        make_client(fake).chat("question")
