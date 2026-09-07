"""Multi-dimensional scorecards and Markdown reports from persisted results."""

from collections import Counter

from src.eval.metrics.agent import agent_metrics
from src.eval.metrics.base import classification, mean, ratio
from src.eval.metrics.grounding import grounding_metrics
from src.eval.metrics.planner import planner_metrics
from src.eval.metrics.retrieval import retrieval_metrics
from src.eval.metrics.runtime import runtime_metrics


def component_classification(rows, task: str) -> dict:
    attempts = [row for row in rows if row.task == task]
    field, label = ("sufficient", "expected_sufficient") if task == "critic" else ("supported", "label")
    observed = [row for row in attempts if row.status == "succeeded" and field in row.raw]
    expected = [row.labels[label] if task == "critic" else row.labels[label] == "supported" for row in observed]
    return {"case_count": len(attempts),
            "component_success_rate": ratio(len(observed), len(attempts)),
            **classification(expected, [row.raw[field] for row in observed])}


def scorecard(rows, k: int) -> dict:
    runtime, resources = runtime_metrics(rows)
    grounding = grounding_metrics(rows)
    verifier = component_classification(rows, "verifier")
    return {
        "quality": {**planner_metrics(rows), "verifier_f1": verifier["f1"],
                    "grounding_pass_rate": grounding["grounding_pass_rate"],
                    "coverage_adequate_rate": grounding["coverage_adequate_rate"]},
        "critic": component_classification(rows, "critic"),
        "verifier": verifier,
        "retrieval": retrieval_metrics(rows, k),
        "agent": agent_metrics(rows), "grounding": grounding,
        "runtime": runtime, "resource": resources,
        "llm_judged": {
            "sample_count": sum(bool(row.llm_judged) and "error" not in row.llm_judged for row in rows),
            **{name: mean(row.llm_judged.get(name) for row in rows if row.llm_judged)
               for name in ("relevance", "completeness", "clarity")},
        },
    }


def build_metrics(rows, metadata: dict, k: int = 10) -> dict:
    return {
        "schema_version": 1, "run_name": metadata["run_name"],
        "dataset_fingerprint": metadata["dataset_fingerprint"],
        "completed": metadata["completed"], "case_count": len(rows),
        "selected_count": metadata["selected_count"],
        "case_ids": [row.case_id for row in rows], "retrieval_k": k,
        "status_counts": dict(Counter(row.status for row in rows)),
        "categories": dict(Counter(row.category for row in rows)),
        "difficulties": dict(Counter(row.difficulty or "unspecified" for row in rows)),
        "scorecard": scorecard(rows, k),
        "per_category": {category: scorecard([row for row in rows if row.category == category], k)
                         for category in sorted({row.category for row in rows})},
    }


def display(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


class EvaluationReportBuilder:
    def build(self, rows, metadata: dict, metrics: dict) -> str:
        lines = ["# Tracker Evaluation Report", "", "## Run Information", "",
                 f"Run: {metadata['run_name']}", "",
                 f"Git: {metadata.get('git_commit')} (dirty={metadata.get('git_dirty')})", "",
                 f"Dataset SHA256: {metadata['dataset_fingerprint']}", "",
                 f"Model: {metadata.get('system', {}).get('settings', {}).get('ollama_model', 'N/A')}", "",
                 f"Model variant: {metadata.get('model_variant')}; adapter: {metadata.get('adapter_name')}; prompt: {metadata.get('prompt_version')}", "",
                 f"Model digest: {metadata.get('system', {}).get('local_model', {}).get('digest', 'N/A')}", "",
                 f"Complete: {metadata['completed']}", "", "## Dataset Summary", "",
                 f"Completed/selected: {len(rows)}/{metadata['selected_count']}", "",
                 f"Statuses: {metrics['status_counts']}", "",
                 f"Categories: {metrics['categories']}", "",
                 f"Difficulties: {metrics['difficulties']}", ""]
        titles = {"quality": "Quality Metrics", "critic": "Critic Metrics",
                  "verifier": "Verifier Metrics", "retrieval": "Retrieval Metrics",
                  "agent": "Agent Metrics", "grounding": "Grounding Metrics",
                  "runtime": "Runtime Metrics", "resource": "Latency / Resource Metrics",
                  "llm_judged": "LLM-judged (optional; not ground truth)"}
        for group, values in metrics["scorecard"].items():
            lines += [f"## {titles[group]}", "", "| Metric | Value |", "|---|---:|"]
            lines += [f"| {key} | {display(value)} |" for key, value in values.items()]
            lines.append("")
        lines += ["## Failures", "", "| Case | Status | Termination | Error |", "|---|---|---|---|"]
        for row in rows:
            if row.status != "succeeded":
                error = (row.error or {}).get("error_type", "")
                lines.append(f"| {display(row.case_id)} | {row.status} | {display(row.termination_reason)} | {display(error)} |")
        lines += ["", "## Per-category Breakdown", "",
                  "| Category | Success | Avg latency | Rounds | Grounding pass | Judge relevance |",
                  "|---|---:|---:|---:|---:|---:|"]
        for category, values in metrics["per_category"].items():
            cells = [category, values["runtime"]["runtime_success_rate"],
                     values["resource"]["avg_latency_seconds"], values["agent"]["avg_research_rounds"],
                     values["grounding"]["grounding_pass_rate"], values["llm_judged"]["relevance"]]
            lines.append("| " + " | ".join(display(cell) for cell in cells) + " |")
        lines += ["", "## Limitations", "",
                  "- Search results and web content may change over time.",
                  "- Demo cases are not human-validated gold. Missing labels/observations produce N/A.",
                  "- Runtime success is completion, not answer correctness. Citation coverage and verifier-passed citations are not human citation accuracy.",
                  "- Grounding and agent summaries use available traces; failed runs may lack those traces. Inspect sample counts and runtime failures together.",
                  "- Ratios use pooled counts except retrieval/classification/planner rates as documented. Empty denominators are N/A.",
                  "- P95 uses nearest rank. Small datasets have unstable tail estimates.",
                  "- Stage times preserve existing trace semantics; concurrent I/O sums can exceed elapsed time and overlapping stages must not be added.",
                  "- Rewrite recovery uses non-null rewritten claims, not all attempted rewrites.",
                  "- LLM judge is optional, separate, and measures relevance/completeness/clarity, not real-world factual truth.", ""]
        return "\n".join(lines)
