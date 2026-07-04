import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
import pandas as pd
from evals.dataset import GroundTruthDataset, EvaluationSample
from evals.config import EvaluationConfig
from evals.metrics.base import BaseMetric, MetricResult

logger = logging.getLogger(__name__)


class Evaluator:
    """Core engine responsible for running evaluation tasks across datasets."""

    def __init__(self, metrics: List[BaseMetric], config: EvaluationConfig = None):
        self.metrics = metrics
        self.config = config or EvaluationConfig.load_from_env()

    def evaluate_sample(self, sample: EvaluationSample) -> Dict[str, MetricResult]:
        """Runs all registered metrics on a single evaluation sample."""
        results = {}
        for metric in self.metrics:
            start_time = time.time()
            try:
                result = metric.evaluate(sample)
                result.metadata["latency_sec"] = time.time() - start_time
                results[metric.name] = result
            except Exception as e:
                logger.error(f"Error executing metric {metric.name}: {str(e)}")
                results[metric.name] = MetricResult(
                    name=metric.name,
                    score=0.0,
                    passed=False,
                    reason=f"Execution error: {str(e)}",
                    metadata={"error": str(e), "latency_sec": time.time() - start_time}
                )
        return results

    def evaluate_dataset(self, dataset: GroundTruthDataset) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Orchestrates evaluation of a dataset using thread-based concurrency.
        
        Returns:
            Tuple containing:
            1. List of detailed results for each sample.
            2. Dictionary summarizing aggregate metrics and stats.
        """
        start_time = time.time()
        detailed_results = []
        concurrency = self.config.concurrency_limit

        if self.config.verbose:
            print(f"Starting evaluation of {len(dataset.samples)} samples with concurrency={concurrency}...")

        # Execute evaluations concurrently
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(self.evaluate_sample, sample): (idx, sample)
                for idx, sample in enumerate(dataset.samples)
            }

            # Pre-populate list to maintain order
            ordered_results = [None] * len(dataset.samples)

            for future in as_completed(futures):
                idx, sample = futures[future]
                try:
                    metrics_output = future.result()
                    
                    # Compute aggregate indicators
                    all_passed = all(res.passed for res in metrics_output.values())
                    
                    # Package detailed record
                    record = {
                        "sample_index": idx,
                        "query": sample.query,
                        "context": sample.context,
                        "ground_truth": sample.ground_truth,
                        "generated_output": sample.generated_output,
                        "metrics": {name: res.model_dump() for name, res in metrics_output.items()},
                        "passed": all_passed,
                        "metadata": sample.metadata
                    }
                    ordered_results[idx] = record
                except Exception as e:
                    logger.error(f"Sample index {idx} failed in evaluation thread: {str(e)}")
                    ordered_results[idx] = {
                        "sample_index": idx,
                        "query": sample.query,
                        "passed": False,
                        "error": str(e)
                    }

            detailed_results = [r for r in ordered_results if r is not None]

        elapsed_time = time.time() - start_time
        summary = self._compute_summary(detailed_results, elapsed_time)
        
        # Telemetry persist hook
        if self.config.db.enabled:
            self._log_to_db(detailed_results, summary)

        return detailed_results, summary

    def _compute_summary(self, detailed_results: List[Dict[str, Any]], duration: float) -> Dict[str, Any]:
        """Calculates statistical averages and pass ratios across the evaluation run."""
        total_samples = len(detailed_results)
        if total_samples == 0:
            return {"total_samples": 0}

        passed_samples = sum(1 for r in detailed_results if r.get("passed", False))
        
        # Initialize aggregator
        metric_sums = {}
        metric_counts = {}
        metric_passes = {}

        for record in detailed_results:
            if "metrics" not in record:
                continue
            for metric_name, res in record["metrics"].items():
                metric_sums[metric_name] = metric_sums.get(metric_name, 0.0) + res["score"]
                metric_counts[metric_name] = metric_counts.get(metric_name, 0) + 1
                metric_passes[metric_name] = metric_passes.get(metric_name, 0) + (1 if res["passed"] else 0)

        # Build final report summary
        summary = {
            "project_id": self.config.db.project_id,
            "total_samples": total_samples,
            "passed_samples": passed_samples,
            "overall_pass_ratio": float(passed_samples / total_samples),
            "execution_duration_sec": duration,
            "metrics": {}
        }

        for name in metric_sums:
            count = metric_counts[name]
            summary["metrics"][name] = {
                "mean_score": float(metric_sums[name] / count),
                "pass_ratio": float(metric_passes[name] / count),
                "total_runs": count
            }

        return summary

    def _log_to_db(self, detailed_results: List[Dict[str, Any]], summary: Dict[str, Any]):
        """Skeleton illustrating connection and database insertion for metric run telemetry.
        
        In a production scenario, this integrates with an RDS postgres or BigQuery database.
        """
        logger.info(f"Persisting {len(detailed_results)} evaluation logs to {self.config.db.connection_string}")
        # e.g., conn = create_engine(self.config.db.connection_string)
        # df = pd.DataFrame(detailed_results)
        # df.to_sql('evaluation_logs', con=conn, if_exists='append')
        pass
