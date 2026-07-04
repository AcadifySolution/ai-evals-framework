import pytest
from evals.dataset import EvaluationSample, GroundTruthDataset
from evals.metrics.accuracy import AccuracyMetric
from evals.metrics.semantic import SemanticSimilarityMetric
from evals.metrics.toxicity import ToxicityMetric
from evals.metrics.hallucination import HallucinationMetric
from evals.evaluator import Evaluator


def test_accuracy_metric_exact_match():
    metric = AccuracyMetric(threshold=0.9, primary_metric="exact_match")
    sample = EvaluationSample(
        query="Tell me hello.",
        ground_truth="Hello World",
        generated_output="hello world"
    )
    result = metric.evaluate(sample)
    assert result.score == 1.0
    assert result.passed is True


def test_accuracy_metric_rouge_similarity():
    metric = AccuracyMetric(threshold=0.5, primary_metric="rouge")
    sample = EvaluationSample(
        query="Name the fruit.",
        ground_truth="This is a red apple",
        generated_output="This is an apple"
    )
    result = metric.evaluate(sample)
    assert result.score > 0.6
    assert result.passed is True


def test_semantic_similarity_fallback():
    metric = SemanticSimilarityMetric(threshold=0.5)
    sample = EvaluationSample(
        query="Write a slogan.",
        ground_truth="Safe and secure data systems",
        generated_output="Secure data systems"
    )
    result = metric.evaluate(sample)
    assert result.score > 0.5
    assert result.passed is True
    assert result.metadata["method"] == "local_tf_idf_fallback"


def test_toxicity_metric_pii_detection():
    metric = ToxicityMetric(threshold=0.1)
    
    # Secure output (no PII or toxicity)
    safe_sample = EvaluationSample(
        query="Say hello.",
        generated_output="Hello! Welcome to our service."
    )
    safe_result = metric.evaluate(safe_sample)
    assert safe_result.score == 0.0
    assert safe_result.passed is True

    # Leaked email output
    leaked_sample = EvaluationSample(
        query="Show emails.",
        generated_output="Contact john.doe@example.com for more info."
    )
    leaked_result = metric.evaluate(leaked_sample)
    assert leaked_result.score > 0.1
    assert leaked_result.passed is False


def test_hallucination_metric_nli_overlap():
    metric = HallucinationMetric(threshold=0.7)
    
    # Grounded response (words match context)
    grounded_sample = EvaluationSample(
        query="Where is Paris?",
        context="Paris is the capital and most populous city of France.",
        generated_output="Paris is a city in France."
    )
    grounded_result = metric.evaluate(grounded_sample)
    assert grounded_result.score >= 0.7
    assert grounded_result.passed is True

    # Hallucinated response (claims not in context)
    hallucinated_sample = EvaluationSample(
        query="Where is Paris?",
        context="Paris is the capital and most populous city of France.",
        generated_output="Paris is a city in Germany where they speak Spanish."
    )
    hallucinated_result = metric.evaluate(hallucinated_sample)
    assert hallucinated_result.score < 0.7
    assert hallucinated_result.passed is False


def test_evaluator_orchestration():
    dataset = GroundTruthDataset([
        EvaluationSample(
            query="Q1",
            context="The sky is blue.",
            ground_truth="Sky is blue.",
            generated_output="The sky is blue."
        )
    ])
    metrics = [
        AccuracyMetric(threshold=0.8),
        ToxicityMetric(threshold=0.1)
    ]
    evaluator = Evaluator(metrics=metrics)
    detailed, summary = evaluator.evaluate_dataset(dataset)
    
    assert summary["total_samples"] == 1
    assert summary["passed_samples"] == 1
    assert "accuracy" in summary["metrics"]
    assert "toxicity" in summary["metrics"]
