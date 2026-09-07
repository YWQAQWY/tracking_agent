import copy

import pytest

from src.eval.compare import RegressionThreshold, compare_eval_runs, metric_direction
from src.eval.report import EvaluationReportBuilder, build_metrics


def metadata():
    return {"run_name": "base", "dataset_fingerprint": "same", "completed": True, "selected_count": 1}


def test_report_sections_category_failure_and_na(row):
    row.status = "timed_out"
    row.error = {"error_type": "TimeoutError"}
    row.termination_reason = "overall_timeout"
    metrics = build_metrics([row], metadata())
    report = EvaluationReportBuilder().build([row], metadata(), metrics)
    for text in ("Run Information", "Dataset Summary", "Quality Metrics", "Retrieval Metrics",
                 "Agent Metrics", "Grounding Metrics", "Runtime Metrics", "Latency / Resource",
                 "Failures", "Per-category", "Limitations", "TimeoutError", "N/A"):
        assert text in report
    assert metrics["per_category"]["demo"]["runtime"]["runtime_timeout_rate"] == 1


def test_compare_delta_and_direction(row):
    baseline = build_metrics([row], metadata())
    candidate = copy.deepcopy(baseline)
    candidate["scorecard"]["resource"]["avg_latency_seconds"] = 1
    result = compare_eval_runs(baseline, candidate)
    latency = next(item for item in result["metrics"] if item["metric"] == "resource.avg_latency_seconds")
    assert latency["delta"] == -1
    assert latency["outcome"] == "improved"
    assert result["status"] == "pass"


@pytest.mark.parametrize("latency,success,expected", [(2, 1, "pass"), (3, 1, "warn"), (3, 0.9, "fail")])
def test_regression_gate(row, latency, success, expected):
    baseline = build_metrics([row], metadata())
    candidate = copy.deepcopy(baseline)
    candidate["scorecard"]["resource"]["avg_latency_seconds"] = latency
    candidate["scorecard"]["runtime"]["runtime_success_rate"] = success
    assert compare_eval_runs(baseline, candidate)["status"] == expected


@pytest.mark.parametrize("key,value,error", [
    ("dataset_fingerprint", "different", "fingerprint"),
    ("completed", False, "incomplete"), ("retrieval_k", 5, "K mismatch"),
    ("case_ids", ["other"], "IDs"),
])
def test_compare_rejects_invalid_experiment(row, key, value, error):
    baseline = build_metrics([row], metadata())
    candidate = copy.deepcopy(baseline)
    candidate[key] = value
    with pytest.raises(ValueError, match=error):
        compare_eval_runs(baseline, candidate)


def test_lost_grounding_is_not_improvement(row):
    baseline = build_metrics([row], metadata())
    candidate = copy.deepcopy(baseline)
    candidate["scorecard"]["grounding"]["citation_coverage"] = None
    assert compare_eval_runs(baseline, candidate)["status"] == "fail"


def test_custom_threshold_and_unknown_gate(row):
    baseline = build_metrics([row], metadata())
    candidate = copy.deepcopy(baseline)
    candidate["scorecard"]["resource"]["avg_latency_seconds"] = 3
    assert compare_eval_runs(baseline, candidate, {"resource.avg_latency_seconds": RegressionThreshold(2)})["status"] == "pass"
    with pytest.raises(ValueError, match="gate metric"):
        compare_eval_runs(baseline, candidate, {"typo": RegressionThreshold(1)})
    with pytest.raises(ValueError):
        RegressionThreshold(float("nan"))


def test_metadata_direction():
    assert metric_direction("retrieval.retrieval_recall_at_10") == "higher"
    assert metric_direction("grounding.unsupported_claim_rate") == "lower"
    assert metric_direction("agent.avg_research_rounds") == "neutral"
    assert metric_direction("llm_judged.clarity") == "neutral"


def test_nonfinite_metrics_cannot_bypass_gate(row):
    baseline = build_metrics([row], metadata())
    candidate = copy.deepcopy(baseline)
    candidate["scorecard"]["runtime"]["runtime_success_rate"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        compare_eval_runs(baseline, candidate)


def test_custom_k_uses_matching_default_recall_gate(row):
    baseline = build_metrics([row], metadata(), k=5)
    result = compare_eval_runs(baseline, baseline)
    assert result["status"] == "pass"
    assert "retrieval.retrieval_recall_at_5" in result["unavailable_gates"]
