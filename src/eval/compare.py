"""Compare matching evaluation runs and apply explicit local regression gates."""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from src.eval.report import display


@dataclass(frozen=True)
class RegressionThreshold:
    tolerance: float
    severity: str = "fail"
    relative: bool = False

    def __post_init__(self):
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        if self.severity not in {"warn", "fail"}:
            raise ValueError("severity must be warn or fail")


DEFAULT_THRESHOLDS = {
    "runtime.runtime_success_rate": RegressionThreshold(0.05),
    "grounding.citation_coverage": RegressionThreshold(0.05),
    "retrieval.retrieval_recall_at_10": RegressionThreshold(0.05),
    "resource.avg_latency_seconds": RegressionThreshold(0.20, "warn", relative=True),
}


def metric_direction(name: str) -> str:
    """Quality and resource directions; counts/rounds have no universal optimum."""
    leaf = name.rsplit(".", 1)[-1]
    if name.startswith("llm_judged."):
        return "neutral"  # judge opinions are deliberately outside automatic gates
    lower = {"runtime_failure_rate", "runtime_timeout_rate", "runtime_budget_exceeded_rate",
             "runtime_cancellation_rate", "retry_rate", "unsupported_claim_rate",
             "claim_drop_rate", "false_positive_rate", "false_negative_rate",
             "empty_query_rate", "exact_duplicate_query_rate", "normalized_duplicate_query_rate",
             "avg_missing_aspect_count", "avg_search_requests", "avg_crawl_requests",
             "avg_llm_calls", "avg_retries", "avg_timeouts"}
    higher = {"runtime_success_rate", "planner_success_rate", "query_count_valid_rate",
              "component_success_rate", "accuracy", "precision", "recall", "f1", "verifier_f1",
              "grounding_pass_rate", "coverage_adequate_rate", "citation_coverage",
              "verifier_passed_citation_rate", "useful_follow_up_rate", "rewrite_recovery_rate"}
    if leaf in lower or "latency_seconds" in leaf or name.startswith("resource.avg_stage_seconds."):
        return "lower"
    if leaf in higher or any(word in leaf for word in ("recall_at_", "precision_at_", "mrr_at_")):
        return "higher"
    return "neutral"


def flatten(values: dict, prefix: str = "") -> dict:
    result = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def compare_eval_runs(baseline: dict, candidate: dict, thresholds=None) -> dict:
    if not baseline.get("dataset_fingerprint") or baseline["dataset_fingerprint"] != candidate.get("dataset_fingerprint"):
        raise ValueError("dataset fingerprint mismatch; compare the same selected cases")
    for run in (baseline, candidate):
        if not run.get("completed") or run["case_count"] != run["selected_count"]:
            raise ValueError("incomplete evaluation cannot be used for regression comparison")
    if baseline["case_ids"] != candidate["case_ids"]:
        raise ValueError("case IDs/order mismatch")
    if baseline.get("schema_version") != candidate.get("schema_version") or baseline["retrieval_k"] != candidate["retrieval_k"]:
        raise ValueError("metric schema or retrieval K mismatch")
    gates = dict(DEFAULT_THRESHOLDS) if thresholds is None else thresholds
    if thresholds is None and baseline["retrieval_k"] != 10:
        gate = gates.pop("retrieval.retrieval_recall_at_10")
        gates[f"retrieval.retrieval_recall_at_{baseline['retrieval_k']}"] = gate
    before, after = flatten(baseline["scorecard"]), flatten(candidate["scorecard"])
    for name in gates:
        if name not in before or name not in after or metric_direction(name) == "neutral":
            raise ValueError(f"unknown/non-directional gate metric: {name}")
    rows = []
    status = "pass"
    for name in sorted(before.keys() | after.keys()):
        left, right = before.get(name), after.get(name)
        if any(value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value))
               for value in (left, right)):
            raise ValueError(f"non-finite/non-numeric metric: {name}")
        direction = metric_direction(name)
        gate_status = "pass"
        delta = None
        outcome = "unavailable"
        if left is not None and right is not None:
            delta = right - left
            if direction == "neutral":
                outcome = "changed" if delta else "unchanged"
            else:
                gain = delta if direction == "higher" else -delta
                outcome = "improved" if gain > 0 else "regressed" if gain < 0 else "unchanged"
                threshold = gates.get(name)
                if threshold:
                    tolerance = threshold.tolerance * abs(left) if threshold.relative else threshold.tolerance
                    if -gain > tolerance + 1e-12:
                        gate_status = threshold.severity
        elif left is not None and right is None:
            outcome = "lost_observation"
            gate_status = gates[name].severity if name in gates else "warn"
        if gate_status == "fail" or gate_status == "warn" and status == "pass":
            status = gate_status
        rows.append({"metric": name, "baseline": left, "candidate": right,
                     "delta": delta, "direction": direction, "outcome": outcome, "gate": gate_status})
    return {"baseline": baseline["run_name"], "candidate": candidate["run_name"],
            "dataset_fingerprint": baseline["dataset_fingerprint"],
            "evaluated_gates": [name for name in gates if before.get(name) is not None and after.get(name) is not None],
            "unavailable_gates": [name for name in gates if before.get(name) is None or after.get(name) is None],
            "status": status, "metrics": rows}


def comparison_markdown(result: dict) -> str:
    lines = ["# Tracker Evaluation Comparison", "",
             f"{result['baseline']} → {result['candidate']}: {result['status'].upper()}", "",
             f"Evaluated gates: {', '.join(result['evaluated_gates']) or 'none'}", "",
             f"Unavailable gates: {', '.join(result['unavailable_gates']) or 'none'}", "",
             "| Metric | Baseline | Candidate | Delta | Direction | Outcome | Gate |",
             "|---|---:|---:|---:|---|---|---|"]
    for row in result["metrics"]:
        lines.append("| " + " | ".join(display(row[key]) for key in (
            "metric", "baseline", "candidate", "delta", "direction", "outcome", "gate")) + " |")
    lines += ["", "N/A is not a passing quality score. Gates evaluate observed metrics only; losing an observation is flagged.",
              "Online search varies over time; this comparison alone is not a causal quality claim.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--output", type=Path, help="new Markdown file; existing file is rejected")
    args = parser.parse_args()
    try:
        thresholds = None
        if args.thresholds:
            thresholds = {key: RegressionThreshold(**value) for key, value in json.loads(args.thresholds.read_text()).items()}
        result = compare_eval_runs(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()), thresholds)
        markdown = comparison_markdown(result)
        if args.output:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(markdown)
        print(markdown)
        return 2 if result["status"] == "fail" else 0
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
