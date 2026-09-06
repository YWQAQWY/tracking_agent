"""Small JSON checkpoints with deterministic paths and atomic replacement."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from src.runtime.errors import CheckpointCorruptedError, CheckpointError


class CheckpointStore:
    """Persist one idempotent state.json and trace.json per run."""

    def __init__(self, base_dir: str | Path = ".runtime/runs") -> None:
        self.base_dir = Path(base_dir)

    def run_dir(self, run_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
            raise CheckpointError("run_id 包含不安全字符")
        return self.base_dir / run_id

    def state_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "state.json"

    def save_state(self, run_id: str, payload: dict[str, Any]) -> Path:
        return self._atomic_write(self.state_path(run_id), payload)

    def save_trace(self, run_id: str, payload: dict[str, Any]) -> Path:
        return self._atomic_write(self.run_dir(run_id) / "trace.json", payload)

    def load_state(self, run_id: str) -> dict[str, Any]:
        path = self.state_path(run_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise CheckpointError(f"找不到 run checkpoint: {run_id}") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointCorruptedError(
                f"checkpoint 已损坏: {path}"
            ) from exc
        if not isinstance(value, dict):
            raise CheckpointCorruptedError(f"checkpoint 顶层必须是 JSON object: {path}")
        return value

    @staticmethod
    def _atomic_write(path: Path, payload: dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except (OSError, TypeError, ValueError) as exc:
            raise CheckpointError(f"无法写入 checkpoint: {path}") from exc
        return path
