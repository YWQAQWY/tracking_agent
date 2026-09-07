"""Component-only evaluation calls the existing component directly."""

from src.eval.extractor import json_value
from src.grounding.registry import EvidenceRegistry


class ComponentEvaluationRunner:
    def __init__(self, *, planner=None, critic=None, verifier=None, retriever=None, reranker=None):
        self.planner = planner
        self.critic = critic
        self.verifier = verifier
        self.retriever = retriever
        self.reranker = reranker

    def run(self, case) -> dict:
        if case.task == "planner":
            plan = self.planner.plan(case.question)
            return {"queries": plan.queries, "max_queries": self.planner.max_queries}
        if case.task == "critic":
            result = self.critic.evaluate(case.question, case.evidence, case.executed_queries)
            return json_value(result)
        if case.task == "verifier":
            result = self.verifier.verify_many([case.claim], EvidenceRegistry(case.evidence))
            return json_value(result[0])
        if case.task == "retrieval":
            try:
                candidates = self.retriever.retrieve(case.question, case.candidates)
                self.retriever.offload()
                evidence = self.reranker.rerank(case.question, candidates)
                # Retrieve source chunk IDs from the supplied fixed corpus.
                ids = {(item.url, item.chunk_index): item.id for item in case.candidates}
                return {"retrieved_urls": [item.url for item in evidence],
                        "retrieved_chunk_ids": [ids[item.url, item.chunk_index] for item in evidence],
                        "final_evidence": json_value(evidence)}
            finally:
                self.retriever.offload()
                offload = getattr(self.reranker, "offload", None)
                if callable(offload):
                    offload()
        raise ValueError(f"Unsupported component task: {case.task}")
