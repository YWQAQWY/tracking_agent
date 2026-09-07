"""Serialize existing public traces without copying their model definitions."""

from typing import Any

from pydantic import TypeAdapter

from src.runtime.models import RuntimeResult


def json_value(value: Any) -> Any:
    return TypeAdapter(type(value)).dump_python(value, mode="json")


def extract_runtime(result: RuntimeResult) -> dict[str, Any]:
    research = result.research_result
    raw = {"run_id": result.run_id, "runtime_trace": json_value(result.trace)}
    if research is not None:
        raw.update(
            plan=json_value(research.plan),
            research_trace=json_value(research.research_trace),
            grounding_trace=json_value(research.grounding_trace),
            grounding_verified=research.grounding_verified,
            retrieval_traces=[json_value(item.retrieval_trace) for item in research.rounds],
            final_evidence=json_value(research.evidence),
            sources=json_value(research.sources),
            # Evidence doesn't retain original chunk IDs. Record only the
            # documented URL#chunk-index identity produced by its public API.
            retrieved_chunk_ids=[item.to_scored_chunk().chunk.id for item in research.evidence],
        )
    return raw
