import json

from pydantic import ValidationError

from evals.config import EvaluationConfig
from evals.dataset import EvaluationSample, GroundTruthDataset
from evals.evaluator import Evaluator
from evals.metrics.accuracy import AccuracyMetric
from evals.metrics.toxicity import ToxicityMetric


def test_evaluation_sample_defaults():
    sample = EvaluationSample(query="Q", generated_output="A")
    assert sample.contexts_as_list == []
    assert sample.ground_truth is None


def test_dataset_json_roundtrip(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps([{"query": "Q", "generated_output": "A"}]))
    dataset = GroundTruthDataset.load_from_json(str(path))
    assert len(dataset.samples) == 1


def test_invalid_config_rejected():
    try:
        EvaluationConfig(concurrency_limit=0)
    except ValidationError:
        return
    assert False, "invalid concurrency should be rejected"


def test_evaluator_requires_unique_metrics():
    metric = ToxicityMetric()
    try:
        Evaluator([metric, metric])
    except ValueError:
        return
    assert False, "duplicate metric names should be rejected"


def test_accuracy_without_ground_truth_is_explicitly_failed():
    result = AccuracyMetric().evaluate(
        EvaluationSample(query="Q", generated_output="A")
    )
    assert result.passed is False
    assert result.metadata["method"] == "skipped"
