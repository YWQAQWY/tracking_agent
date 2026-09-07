import pytest

from src.eval.metrics.agent import agent_metrics
from src.eval.metrics.base import classification, p95
from src.eval.metrics.grounding import grounding_metrics
from src.eval.metrics.planner import query_metrics
from src.eval.metrics.retrieval import ranking_metrics, retrieval_metrics
from src.eval.metrics.runtime import runtime_metrics


def test_query_metrics():
    result = query_metrics(["q1", "q1", "q2"], 3)
    assert result["exact_duplicate_query_rate"] == pytest.approx(1 / 3)
    assert result["query_count_valid_rate"] == 1
    assert query_metrics(["Q1", " q1 ", ""], 2)["empty_query_rate"] == pytest.approx(1 / 3)
    assert query_metrics(["Q1", " q1 "], 2)["normalized_duplicate_query_rate"] == 0.5
    assert query_metrics([], 2)["empty_query_rate"] is None


def test_critic_confusion_matrix():
    result = classification([True, False, False, True], [True, False, True, False])
    assert [result[key] for key in ("tp", "tn", "fp", "fn")] == [1, 1, 1, 1]
    assert result["accuracy"] == result["false_positive_rate"] == result["false_negative_rate"] == 0.5


def test_verifier_precision_recall_f1():
    result = classification([True, True, True, False], [True, True, False, False])
    assert result["precision"] == 1
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == 0.8
    assert classification([], [])["accuracy"] is None
    assert classification([True], [False])["f1"] == 0


def test_retrieval_at_k_unique_short_and_rank():
    assert ranking_metrics(["a", "a", "b", "c"], ["b", "d"], 2) == {
        "recall": 0.5, "precision": 0.5, "mrr": 0.5}
    assert ranking_metrics(["a"], ["a"], 10)["precision"] == 1
    assert ranking_metrics(["a", "b", "c", "d"], ["d"], 10)["mrr"] == 0.25
    assert ranking_metrics([], ["a"], 10) == {"recall": 0, "precision": 0, "mrr": 0}


@pytest.mark.parametrize("gold", [None, []])
def test_no_gold_is_unavailable(gold):
    assert ranking_metrics(["a"], gold, 10) == {"recall": None, "precision": None, "mrr": None}


def test_retrieval_url_normalization_and_missing_labels(row):
    row.labels["relevant_urls"] = ["https://EXAMPLE.org/cache/?utm_source=test"]
    result = retrieval_metrics([row], 10)
    assert result["retrieval_recall_at_10"] == 1
    assert result["url_labelled_count"] == 1
    assert result["chunk_recall_at_10"] is None


def test_followup_and_stop_distribution(row):
    result = agent_metrics([row])
    assert result["useful_follow_up_rate"] == 0.5
    assert result["avg_evidence_gain"] == 1
    assert result["avg_research_rounds"] == 3
    assert result["stop_reason_distribution"] == {"no_new_evidence": 1}


def test_no_followup_unavailable(row):
    row.raw["research_trace"]["rounds"] = row.raw["research_trace"]["rounds"][:1]
    assert agent_metrics([row])["useful_follow_up_rate"] is None


def test_grounding_counts_ratios_and_mapping(row):
    metrics = grounding_metrics([row])
    assert metrics["grounding_pass_rate"] == 0.8
    assert metrics["unsupported_claim_rate"] == 0.4
    assert metrics["claim_drop_rate"] == 0.2
    assert metrics["rewrite_recovery_rate"] == pytest.approx(2 / 3)
    assert metrics["citation_coverage"] == 1
    assert metrics["coverage_adequate_rate"] == 0
    row.raw["grounding_trace"]["claim_traces"][0]["evidence_ids"] = ["E99"]
    assert grounding_metrics([row])["citation_coverage"] == 7 / 8


def test_zero_claims_and_fallback_not_verified(row):
    for key in row.raw["grounding_trace"]:
        if key.endswith("count"):
            row.raw["grounding_trace"][key] = 0
    row.raw["grounding_trace"]["claim_traces"] = []
    assert grounding_metrics([row])["grounding_pass_rate"] is None
    row.raw["grounding_verified"] = False
    row.raw["grounding_trace"]["fallback_reason"] = "parse failed"
    assert grounding_metrics([row])["trace_count"] == 0
    assert grounding_metrics([row])["fallback_count"] == 1


def test_runtime_rates_latency_and_resources(row):
    rows = [row.model_copy(deep=True) for _ in range(3)]
    for item, status, latency in zip(rows, ["succeeded", "failed", "timed_out"], [1, 2, 9]):
        item.status, item.latency_seconds = status, latency
    reliability, resources = runtime_metrics(rows)
    assert reliability["runtime_success_rate"] == pytest.approx(1 / 3)
    assert reliability["runtime_timeout_rate"] == pytest.approx(1 / 3)
    assert resources["avg_latency_seconds"] == 4
    assert resources["median_latency_seconds"] == 2
    assert resources["p95_latency_seconds"] == 9
    assert resources["avg_llm_calls"] == 5


def test_nearest_rank_p95():
    assert p95(list(range(1, 101))) == 95
    assert p95([7]) == 7
    assert p95([]) is None
