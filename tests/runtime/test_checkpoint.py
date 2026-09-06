import os

import pytest

from src.runtime.checkpoint import CheckpointStore
from src.runtime.errors import CheckpointCorruptedError


def test_checkpoint_save_load_and_idempotent_path(tmp_path) -> None:
    store = CheckpointStore(tmp_path)
    first = store.save_state("run_1", {"round": 1})
    second = store.save_state("run_1", {"round": 2})
    assert first == second
    assert store.load_state("run_1") == {"round": 2}
    assert list((tmp_path / "run_1").glob("state*.json")) == [first]


def test_checkpoint_write_uses_atomic_replace(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    real_replace = os.replace

    def recorded_replace(source, target) -> None:
        calls.append((str(source), str(target)))
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", recorded_replace)
    path = CheckpointStore(tmp_path).save_state("run_1", {"ok": True})
    assert calls == [(str(path) + ".tmp", str(path))]
    assert not path.with_suffix(".json.tmp").exists()


def test_corrupted_checkpoint_has_clear_error(tmp_path) -> None:
    path = tmp_path / "run_1" / "state.json"
    path.parent.mkdir(parents=True)
    path.write_text("{bad", encoding="utf-8")
    with pytest.raises(CheckpointCorruptedError, match="已损坏"):
        CheckpointStore(tmp_path).load_state("run_1")


def test_checkpoint_rejects_unsafe_run_id(tmp_path) -> None:
    with pytest.raises(Exception, match="不安全"):
        CheckpointStore(tmp_path).save_state("../escape", {"bad": True})


def test_missing_checkpoint_has_clear_error(tmp_path) -> None:
    with pytest.raises(Exception, match="找不到"):
        CheckpointStore(tmp_path).load_state("run_missing")


def test_state_and_trace_use_separate_deterministic_files(tmp_path) -> None:
    store = CheckpointStore(tmp_path)
    state = store.save_state("run_1", {"status": "running"})
    trace = store.save_trace("run_1", {"status": "running"})
    assert state.name == "state.json"
    assert trace.name == "trace.json"
    assert state.parent == trace.parent
