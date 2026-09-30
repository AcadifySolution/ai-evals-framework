from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from evals.dataset import EvaluationSample


class MetricResult(BaseModel):
    name: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=1.0)
    passed: bool
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseMetric(ABC):
    def __init__(self, threshold: float = 0.5):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, sample: EvaluationSample) -> MetricResult:
        raise NotImplementedError
