import statistics

from src.eval.metrics.base import mean, p95, ratio


def runtime_metrics(rows) -> tuple[dict, dict]:
    attempts = [row for row in rows if row.task == "end_to_end"]
    traces = [row.raw["runtime_trace"] for row in attempts if "runtime_trace" in row.raw]
    latency = [row.latency_seconds for row in attempts]
    statuses = {"success": "succeeded", "failure": "failed", "timeout": "timed_out",
                "budget_exceeded": "budget_exceeded", "cancellation": "cancelled"}
    reliability = {
        "case_count": len(attempts), "trace_count": len(traces),
        **{f"runtime_{name}_rate": ratio(sum(row.status == status for row in attempts), len(attempts))
           for name, status in statuses.items()},
        "retry_rate": ratio(sum(trace["counters"]["retries"] > 0 for trace in traces), len(traces)),
    }
    resources = {
        "latency_sample_count": len(latency),
        "avg_latency_seconds": mean(latency),
        "median_latency_seconds": statistics.median(latency) if latency else None,
        "p95_latency_seconds": p95(latency),
        **{f"avg_{name}": mean(trace["counters"][name] for trace in traces) for name in (
            "research_rounds", "search_requests", "crawl_requests", "llm_calls", "retries", "timeouts"
        )},
        "avg_stage_seconds": {
            stage: mean(trace["stage_durations"].get(stage) for trace in traces)
            for stage in sorted({key for trace in traces for key in trace["stage_durations"]})
        },
    }
    return reliability, resources
