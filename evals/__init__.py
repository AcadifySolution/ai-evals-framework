"""
AI Evals Framework (ai-evals-framework)
An enterprise-grade package for evaluating LLM responses against ground-truth datasets.
"""

from evals.config import EvaluationConfig
from evals.dataset import EvaluationSample, GroundTruthDataset
from evals.evaluator import Evaluator
from evals.metrics.accuracy import AccuracyMetric
from evals.metrics.base import BaseMetric
from evals.metrics.hallucination import HallucinationMetric
from evals.metrics.semantic import SemanticSimilarityMetric
from evals.metrics.toxicity import ToxicityMetric

__version__ = "0.1.0"
__all__ = [
    "AccuracyMetric",
    "BaseMetric",
    "EvaluationConfig",
    "EvaluationSample",
    "Evaluator",
    "GroundTruthDataset",
    "HallucinationMetric",
    "SemanticSimilarityMetric",
    "ToxicityMetric",
]
