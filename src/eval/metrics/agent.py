from collections import Counter

from src.eval.metrics.base import mean, ratio


def agent_metrics(rows) -> dict:
    traces = [row.raw["research_trace"] for row in rows if "research_trace" in row.raw]
    followups = [round_ for trace in traces for round_ in trace["rounds"] if round_["round_index"] > 1]
    return {
        "trace_count": len(traces),
        "avg_research_rounds": mean(len(trace["rounds"]) for trace in traces),
        "avg_initial_query_count": mean(len(row.raw["plan"]["queries"]) for row in rows if "plan" in row.raw),
        "avg_follow_up_query_count": mean(len(item["queries"]) for item in followups),
        "follow_up_round_count": len(followups),
        "useful_follow_up_rate": ratio(sum(item["new_evidence_count"] > 0 for item in followups), len(followups)),
        "avg_evidence_gain": mean(item["new_evidence_count"] for item in followups),
        "evidence_gain_by_round": {
            str(index): mean(item["new_evidence_count"] for item in followups if item["round_index"] == index)
            for index in sorted({item["round_index"] for item in followups})
        },
        "stop_reason_distribution": dict(Counter(trace["stop_reason"] for trace in traces)),
    }
