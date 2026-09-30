from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from evals.config import EvaluationConfig
from evals.dataset import EvaluationSample, GroundTruthDataset
from evals.metrics.base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class Evaluator:
    """Concurrent, ordered evaluation engine with per-metric error isolation."""

    def __init__(self, metrics: list[BaseMetric], config: EvaluationConfig | None = None):
        if not metrics:
            raise ValueError("At least one metric is required")
        names = [m.name for m in metrics]
        if len(names) != len(set(names)):
            raise ValueError("Metric names must be unique")
        self.metrics = metrics
        self.config = config or EvaluationConfig.load_from_env()

    def evaluate_sample(self, sample: EvaluationSample) -> dict[str, MetricResult]:
        results: dict[str, MetricResult] = {}
        for metric in self.metrics:
            started = time.perf_counter()
            try:
                result = metric.evaluate(sample)
            except Exception:
                logger.exception("Metric %s failed", metric.name)
                result = MetricResult(
                    name=metric.name,
                    score=0.0,
                    passed=False,
                    reason="Metric execution failed; see logs for details.",
                    metadata={"error_type": "metric_execution_error"},
                )
            result.metadata["latency_sec"] = round(time.perf_counter() - started, 6)
            results[metric.name] = result
        return results

    def evaluate_dataset(self, dataset: GroundTruthDataset) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        started = time.perf_counter()
        samples = dataset.samples
        ordered: list[dict[str, Any] | None] = [None] * len(samples)

        with ThreadPoolExecutor(max_workers=self.config.concurrency_limit) as executor:
            futures = {
                executor.submit(self.evaluate_sample, sample): (idx, sample)
                for idx, sample in enumerate(samples)
            }
            for future in as_completed(futures):
                idx, sample = futures[future]
                try:
                    metric_results = future.result()
                    ordered[idx] = self._build_record(idx, sample, metric_results)
                except Exception:
                    logger.exception("Sample %d failed", idx)
                    ordered[idx] = {
                        "sample_index": idx,
                        "query": sample.query,
                        "passed": False,
                        "error": "sample_execution_error",
                    }

        results = [record for record in ordered if record is not None]
        summary = self._compute_summary(results, time.perf_counter() - started)
        return results, summary

    @staticmethod
    def _build_record(
        idx: int, sample: EvaluationSample, metric_results: dict[str, MetricResult]
    ) -> dict[str, Any]:
        return {
            "sample_index": idx,
            "query": sample.query,
            "context": sample.context,
            "ground_truth": sample.ground_truth,
            "generated_output": sample.generated_output,
            "metrics": {name: result.model_dump() for name, result in metric_results.items()},
            "passed": all(result.passed for result in metric_results.values()),
            "metadata": sample.metadata,
        }

    @staticmethod
    def _compute_summary(results: list[dict[str, Any]], duration: float) -> dict[str, Any]:
        total = len(results)
        metric_values: dict[str, list[MetricResult]] = {}
        for record in results:
            for name, raw in record.get("metrics", {}).items():
                metric_values.setdefault(name, []).append(MetricResult.model_validate(raw))

        summary: dict[str, Any] = {
            "total_samples": total,
            "passed_samples": sum(bool(r.get("passed")) for r in results),
            "overall_pass_ratio": (sum(bool(r.get("passed")) for r in results) / total) if total else 0.0,
            "execution_duration_sec": round(duration, 6),
            "metrics": {},
        }
        for name, values in metric_values.items():
            summary["metrics"][name] = {
                "mean_score": sum(v.score for v in values) / len(values),
                "pass_ratio": sum(v.passed for v in values) / len(values),
                "total_runs": len(values),
            }
        return summary
