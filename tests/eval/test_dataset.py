import json

import pytest

from src.eval.dataset import DatasetError, EvaluationDatasetLoader, dataset_fingerprint


def test_load_and_fingerprint_ignore_json_spacing(tmp_path):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    first.write_text('{"id":"a","question":"Question?"}\n')
    second.write_text('  {"question": "Question?", "id": "a"}\n\n')
    loader = EvaluationDatasetLoader()
    assert dataset_fingerprint(loader.load(first)) == dataset_fingerprint(loader.load(second))
    second.write_text('{"id":"a","question":"Different?"}\n')
    assert dataset_fingerprint(loader.load(first)) != dataset_fingerprint(loader.load(second))


@pytest.mark.parametrize("content,match", [
    ("not-json", ":1:"), ("[]", "JSON object"),
    ('{"question":"q"}', "id"), ('{"id":"a","question":" "}', "question"),
    ('{"id":"a","question":"q","mystery":1}', "mystery"),
    ('{"id":"a","question":"q","task":"unknown"}', "unknown task"),
    ('{"id":"a","question":"q","difficulty":"impossible"}', "difficulty"),
    ('{"id":"a","question":"q"}\n{"id":"a","question":"different"}', "duplicate ID"),
    ('{"id":"a","question":"q"}\n{"id":"b","question":" Q "}', "duplicate question"),
    ("\n", "empty"),
])
def test_reject_invalid_dataset(tmp_path, content, match):
    path = tmp_path / "invalid.jsonl"
    path.write_text(content)
    with pytest.raises(DatasetError, match=match):
        EvaluationDatasetLoader().load(path)


def test_fingerprint_includes_limit_labels_and_order(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('\n'.join(json.dumps({"id": str(i), "question": f"q{i}"}) for i in range(3)))
    cases = EvaluationDatasetLoader().load(path)
    assert dataset_fingerprint(cases) != dataset_fingerprint(cases[:2])
    assert dataset_fingerprint(cases) != dataset_fingerprint(cases[::-1])


def test_all_demo_datasets_valid():
    from pathlib import Path
    for path in Path("eval_data").glob("*/demo.jsonl"):
        cases = EvaluationDatasetLoader().load(path)
        assert all(case.metadata == {"demo": True, "human_validated": False} for case in cases)


def test_reject_invalid_relevance_url(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id":"a","question":"q","relevant_urls":["not-a-url"]}')
    with pytest.raises(DatasetError, match="HTTP"):
        EvaluationDatasetLoader().load(path)


def test_reject_duplicate_component_fixture(tmp_path):
    case = EvaluationDatasetLoader().load("eval_data/verifier/demo.jsonl")[0]
    path = tmp_path / "dupe.jsonl"
    path.write_text(case.model_dump_json() + '\n' + case.model_copy(update={"id": "other"}).model_dump_json())
    with pytest.raises(DatasetError, match="duplicate question/evidence"):
        EvaluationDatasetLoader().load(path)
