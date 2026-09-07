from src.eval.metrics.base import mean, ratio


def grounding_metrics(rows) -> dict:
    # Legacy fallback placeholders contain zero counts, not observations of a
    # verified answer. Keep them out of the successful-grounding denominators.
    observed = [row for row in rows if row.raw.get("grounding_verified")
                and row.raw.get("grounding_trace")
                and not row.raw["grounding_trace"].get("fallback_reason")]
    traces = [row.raw["grounding_trace"] for row in observed]
    counts = {key: sum(trace[key] for trace in traces) for key in (
        "draft_claim_count", "verified_claim_count", "unsupported_claim_count",
        "rewritten_claim_count", "rewrite_passed_claim_count", "dropped_claim_count",
        "cited_source_count",
    )}
    draft = counts["draft_claim_count"]
    final = counts["verified_claim_count"]
    mapped = 0
    mapping_observed = True
    for row in observed:
        claim_traces = row.raw["grounding_trace"].get("claim_traces", [])
        claims = [claim for claim in claim_traces if claim["final_text"] is not None]
        if len(claims) != row.raw["grounding_trace"]["verified_claim_count"]:
            mapping_observed = False
        valid_ids = {f"E{index}" for index in range(1, len(row.raw.get("final_evidence", [])) + 1)}
        mapped += sum(claim["supported"] and bool(claim["evidence_ids"])
                      and set(claim["evidence_ids"]) <= valid_ids for claim in claims)
    coverages = [trace["coverage"] for trace in traces if trace.get("coverage") is not None]
    return {
        "trace_count": len(traces), **counts,
        "fallback_count": sum(bool(row.raw.get("grounding_trace", {}).get("fallback_reason"))
                              for row in rows if row.raw.get("grounding_trace")),
        "grounding_pass_rate": ratio(final, draft),
        "unsupported_claim_rate": ratio(counts["unsupported_claim_count"], draft),
        "claim_drop_rate": ratio(counts["dropped_claim_count"], draft),
        # Existing trace counts non-null rewrites, not attempted null rewrites.
        "rewrite_recovery_rate": ratio(counts["rewrite_passed_claim_count"], counts["rewritten_claim_count"]),
        "citation_coverage": ratio(mapped, final) if mapping_observed else None,
        "verifier_passed_citation_rate": ratio(mapped, final) if mapping_observed else None,
        "coverage_count": len(coverages),
        "coverage_adequate_rate": mean(float(item["adequate"]) for item in coverages),
        "avg_missing_aspect_count": mean(len(item["missing_aspects"]) for item in coverages),
    }
