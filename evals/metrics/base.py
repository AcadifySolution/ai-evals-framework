from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from evals.dataset import EvaluationSample


class MetricResult(BaseModel):
    """Encapsulates the output of a single evaluation metric run."""
    name: str = Field(..., description="Name of the evaluated metric")
    score: float = Field(..., description="Calculated evaluation score, typically in range [0.0, 1.0]")
    passed: bool = Field(..., description="Flag indicating if the score passed defined thresholds")
    reason: str = Field(default="", description="Detailed qualitative rationale for the score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metric execution metadata (e.g. execution time)")


class BaseMetric(ABC):
    """Abstract base class representing an evaluation metric."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identification of the metric."""
        pass

    @abstractmethod
    def evaluate(self, sample: EvaluationSample) -> MetricResult:
        """Executes metrics calculation over a single sample.
        
        Args:
            sample: The EvaluationSample payload containing input, context, output.

        Returns:
            A MetricResult containing the score, pass/fail status, and explanation.
        """
        pass
