"""Strict JSONL loading and canonical, order-sensitive dataset fingerprints."""

import argparse
import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from src.eval.models import (
    CriticEvalCase, EndToEndEvalCase, EvaluationCase, PlannerEvalCase,
    RetrievalEvalCase, VerifierEvalCase,
)

CASE_TYPES = {
    "end_to_end": EndToEndEvalCase, "planner": PlannerEvalCase,
    "critic": CriticEvalCase, "verifier": VerifierEvalCase,
    "retrieval": RetrievalEvalCase,
}


class DatasetError(ValueError):
    """A dataset error including the offending file and line."""


class EvaluationDatasetLoader:
    def load(self, path: str | Path) -> list[EvaluationCase]:
        cases = []
        ids: set[str] = set()
        questions: set[tuple[str, str]] = set()
        fixtures: set[str] = set()
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise ValueError("case must be a JSON object")
                    task = payload.get("task", "end_to_end")
                    if task not in CASE_TYPES:
                        raise ValueError(f"unknown task: {task}")
                    case = CASE_TYPES[task].model_validate(payload)
                    if case.id in ids:
                        raise ValueError(f"duplicate ID: {case.id}")
                    # A verifier/critic may intentionally pair one question with
                    # different fixed evidence. Other tasks forbid duplicates.
                    key = (task, " ".join(case.question.casefold().split()))
                    if task not in {"verifier", "critic"} and key in questions:
                        raise ValueError(f"duplicate question: {case.question}")
                    fixture_key = json.dumps(
                        {**case.model_dump(mode="json", exclude={"id", "category", "difficulty", "notes", "metadata"}),
                         "question": key[1]}, sort_keys=True, ensure_ascii=False,
                    )
                    if fixture_key in fixtures:
                        raise ValueError("duplicate question/evidence fixture")
                    ids.add(case.id)
                    questions.add(key)
                    fixtures.add(fixture_key)
                    cases.append(case)
                except (ValueError, TypeError, ValidationError) as exc:
                    raise DatasetError(f"{path}:{line_number}: {exc}") from exc
        if not cases:
            raise DatasetError(f"{path}: dataset is empty")
        return cases


def dataset_fingerprint(cases: list[EvaluationCase]) -> str:
    normalized = "\n".join(
        json.dumps(case.model_dump(mode="json"), sort_keys=True,
                   ensure_ascii=False, separators=(",", ":"))
        for case in cases
    )
    return hashlib.sha256(normalized.encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", required=True)
    args = parser.parse_args()
    try:
        cases = EvaluationDatasetLoader().load(args.validate)
    except (OSError, DatasetError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Valid: {len(cases)} cases; SHA256 {dataset_fingerprint(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
