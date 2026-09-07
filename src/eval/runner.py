"""Sequential evaluation with per-case persistence and failure isolation."""

import asyncio
import json
import os
import subprocess
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

from src.eval.dataset import EvaluationDatasetLoader, dataset_fingerprint
from src.eval.extractor import extract_runtime, json_value
from src.eval.models import EvalCaseResult, EvalRunConfig, EvaluationCase
from src.runtime.models import RunStatus, RuntimeResult


def write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def git_metadata() -> dict:
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", *args], stderr=subprocess.DEVNULL, text=True, timeout=5
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain"))}


class EvaluationRunner:
    def __init__(
        self,
        execute_runtime: Callable[[str], Awaitable[RuntimeResult]],
        component_runner=None,
        judge=None,
    ) -> None:
        # Production passes main.execute_runtime, the existing composition root
        # which calls AgentRuntime.run. Tests inject a fake runtime boundary.
        self.execute_runtime = execute_runtime
        self.component_runner = component_runner
        self.judge = judge

    async def run(self, config: EvalRunConfig, system_metadata: dict | None = None) -> Path:
        from src.eval.report import EvaluationReportBuilder, build_metrics

        all_cases = EvaluationDatasetLoader().load(config.dataset_path)
        cases = all_cases[:config.limit] if config.limit else all_cases
        if config.judge and self.judge is None:
            raise ValueError("judge requested but no judge was supplied")
        destination = Path(config.output_dir) / config.run_name
        destination.mkdir(parents=True, exist_ok=False)
        metadata = {
            **config.model_dump(mode="json"), **git_metadata(),
            "schema_version": 1,
            "timestamp": datetime.now(UTC).isoformat(),
            "dataset_fingerprint": dataset_fingerprint(cases),
            "full_dataset_fingerprint": dataset_fingerprint(all_cases),
            "case_ids": [case.id for case in cases], "dataset_size": len(all_cases),
            "selected_count": len(cases), "completed": False,
            "system": system_metadata or {},
        }
        write_json(destination / "config.json", metadata)
        with (destination / "dataset.jsonl").open("w", encoding="utf-8") as handle:
            for case in cases:
                handle.write(case.model_dump_json() + "\n")
        rows: list[EvalCaseResult] = []
        try:
            with (destination / "cases.jsonl").open("w", encoding="utf-8") as handle:
                for index, case in enumerate(cases, 1):
                    print(f"[{index}/{len(cases)}] {case.id} ...", flush=True)
                    row = await self._execute(case)
                    rows.append(row)
                    handle.write(row.model_dump_json() + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    # Runtime deliberately returns CANCELLED. Restore benchmark
                    # cancellation here, after saving the interrupted case.
                    if row.status == RunStatus.CANCELLED.value:
                        raise asyncio.CancelledError
                    if config.judge and case.task == "end_to_end" and row.status == "succeeded":
                        judge_started = time.perf_counter()
                        try:
                            row.llm_judged = json_value(self.judge.evaluate(case, row.answer or ""))
                        except Exception as exc:
                            row.llm_judged = {"error": type(exc).__name__ + ": " + str(exc)}
                        row.llm_judged["latency_seconds"] = time.perf_counter() - judge_started
                        # Save judge separately until final JSONL rewrite, so an
                        # interruption cannot discard the raw runtime result.
                    print(f"  {row.status} {row.latency_seconds:.2f}s", flush=True)
            metadata["completed"] = True
        finally:
            metadata["finished_at"] = datetime.now(UTC).isoformat()
            metadata["completed_count"] = len(rows)
            write_json(destination / "config.json", metadata)
            # Only completed in-memory rows are rewritten; no older run is opened.
            with (destination / "cases.jsonl.tmp").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(row.model_dump_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            (destination / "cases.jsonl.tmp").replace(destination / "cases.jsonl")
            metrics = build_metrics(rows, metadata, config.retrieval_k)
            write_json(destination / "metrics.json", metrics)
            (destination / "report.md").write_text(
                EvaluationReportBuilder().build(rows, metadata, metrics), encoding="utf-8"
            )
        return destination

    async def _execute(self, case: EvaluationCase) -> EvalCaseResult:
        started = time.perf_counter()
        fields = dict(case_id=case.id, question=case.question, category=case.category,
                      difficulty=case.difficulty, task=case.task)
        labels = case.model_dump(mode="json", include={
            "relevant_urls", "relevant_chunk_ids", "expected_sufficient", "label",
            "expected_aspects", "expected_missing_aspects",
        })
        try:
            if case.task != "end_to_end":
                if self.component_runner is None:
                    raise ValueError("component runner is required for this dataset")
                raw = self.component_runner.run(case)
                return EvalCaseResult(**fields, status="succeeded", raw=raw, labels=labels,
                                      latency_seconds=time.perf_counter() - started)
            result = await self.execute_runtime(case.question)
            return EvalCaseResult(
                **fields, status=result.status.value, answer=result.answer,
                termination_reason=result.termination_reason,
                error=json_value(result.error), raw=extract_runtime(result), labels=labels,
                latency_seconds=result.trace.total_duration,
            )
        except Exception as exc:
            return EvalCaseResult(
                **fields, status="failed", termination_reason="evaluation_case_error",
                error={"error_type": type(exc).__name__, "message": str(exc)}, labels=labels,
                latency_seconds=time.perf_counter() - started,
            )
