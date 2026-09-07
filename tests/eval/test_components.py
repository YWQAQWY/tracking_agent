import json

from src.agent.critic import EvidenceCritic
from src.agent.critic_context import CriticContextBuilder
from src.eval.components import ComponentEvaluationRunner
from src.eval.dataset import EvaluationDatasetLoader
from src.eval.extractor import json_value
from src.eval.judges.llm_judge import LLMJudge
from src.grounding.verifier import CitationVerifier
from src.plan.search_planner import SearchPlanner


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def chat(self, user_prompt, system_prompt):
        self.calls += 1
        return json.dumps(self.payload)


def test_planner_component_existing_parser():
    llm = FakeLLM({"queries": [" query ", "query"]})
    case = EvaluationDatasetLoader().load("eval_data/planner/demo.jsonl")[0]
    output = ComponentEvaluationRunner(planner=SearchPlanner(llm)).run(case)
    assert output == {"queries": ["query"], "max_queries": 3}
    assert llm.calls == 1


def test_critic_component_existing_parser():
    llm = FakeLLM({"sufficient": True, "reason": "supported definition"})
    case = EvaluationDatasetLoader().load("eval_data/critic/demo.jsonl")[0]
    output = ComponentEvaluationRunner(critic=EvidenceCritic(llm, CriticContextBuilder())).run(case)
    assert output["sufficient"] is True


def test_verifier_component_existing_parser():
    llm = FakeLLM({"results": [{"claim_id": "C1", "supported": True,
                               "supported_evidence_ids": ["E1"], "reason": "direct support"}]})
    case = EvaluationDatasetLoader().load("eval_data/verifier/demo.jsonl")[0]
    output = ComponentEvaluationRunner(verifier=CitationVerifier(llm)).run(case)
    assert output["supported"] is True


def test_retrieval_component_reuses_corpus_ids_and_embedding_ablation():
    from src.models.evidence import ScoredChunk
    from src.retrieval.reranker import EmbeddingOnlyReranker
    case = EvaluationDatasetLoader().load("eval_data/retrieval/demo.jsonl")[0]
    class Retriever:
        def retrieve(self, question, chunks):
            return [ScoredChunk(chunks[0], 0.9)]
        def offload(self):
            pass
    output = ComponentEvaluationRunner(retriever=Retriever(), reranker=EmbeddingOnlyReranker()).run(case)
    assert output["retrieved_chunk_ids"] == ["cache-0"]


def test_optional_judge_is_structured():
    llm = FakeLLM({"relevance": 4, "completeness": 2, "clarity": 5, "reason": "opinion"})
    case = EvaluationDatasetLoader().load("eval_data/end_to_end/demo.jsonl")[0]
    assert json_value(LLMJudge(llm).evaluate(case, "answer"))["clarity"] == 5


def test_production_does_not_import_eval():
    import ast
    from pathlib import Path
    for path in Path("src").rglob("*.py"):
        if "eval" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("src.eval"), str(path)
            if isinstance(node, ast.Import):
                assert all(not item.name.startswith("src.eval") for item in node.names), str(path)


def test_metadata_excludes_future_secrets(monkeypatch):
    from tracker.eval import system_metadata
    from src.config import Settings
    class FutureSettings:
        def model_dump(self, mode, include):
            values = {**Settings().model_dump(mode=mode), "api_key": "secret", "password": "secret"}
            return {key: value for key, value in values.items() if key in include}
    metadata = system_metadata(FutureSettings())
    assert "secret" not in json.dumps(metadata)
    assert metadata["settings"]["max_run_seconds"] == 900
    assert len(metadata["source_fingerprint"]) == 64


def test_model_metadata_failure_is_nonfatal(monkeypatch):
    import httpx
    from tracker.eval import local_model_metadata
    from src.config import Settings
    def unavailable(*args, **kwargs):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr(httpx.Client, "get", unavailable)
    assert local_model_metadata(Settings())["digest"] is None
