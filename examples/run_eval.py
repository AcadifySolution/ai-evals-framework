#!/usr/bin/env python3
import os
import sys
import json

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.config import EvaluationConfig
from evals.dataset import GroundTruthDataset
from evals.evaluator import Evaluator
from evals.metrics.hallucination import HallucinationMetric
from evals.metrics.semantic import SemanticSimilarityMetric
from evals.metrics.accuracy import AccuracyMetric
from evals.metrics.toxicity import ToxicityMetric


def run_demo():
    print("==================================================")
    print("       AI Evals Framework - Demo Run Start        ")
    print("==================================================")

    # 1. Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "dataset.json")
    output_path = os.path.join(base_dir, "demo_results.json")

    # 2. Load custom configurations
    # This automatically picks up environment variables if set.
    # To run OpenAI judge or embedding models, export OPENAI_API_KEY.
    config = EvaluationConfig.load_from_env()
    config.thresholds.hallucination_threshold = 0.8
    config.thresholds.semantic_similarity_threshold = 0.75
    config.thresholds.correctness_threshold = 0.7
    config.thresholds.toxicity_threshold = 0.1  # Maximum tolerated toxicity

    # 3. Load dataset
    print(f"Reading dataset from: {dataset_path}")
    dataset = GroundTruthDataset.load_from_json(dataset_path)
    print(f"Loaded {len(dataset.samples)} test samples.")

    # 4. Instantiate Metrics
    # In the absence of an API key, metrics fallback to deterministic regex/vectorizers
    metrics = [
        HallucinationMetric(threshold=config.thresholds.hallucination_threshold, config=config),
        SemanticSimilarityMetric(threshold=config.thresholds.semantic_similarity_threshold, config=config),
        AccuracyMetric(threshold=config.thresholds.correctness_threshold, primary_metric="rouge"),
        ToxicityMetric(threshold=config.thresholds.toxicity_threshold, config=config)
    ]

    # 5. Initialize Evaluator & Execute
    evaluator = Evaluator(metrics=metrics, config=config)
    
    print("\nExecuting evaluation pipeline...")
    detailed_results, summary = evaluator.evaluate_dataset(dataset)

    # 6. Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": detailed_results}, f, indent=2)
    print(f"Saved evaluation results to: {output_path}")

    # 7. Print summary report
    print("\n" + "=" * 60)
    print("              EVALUATION PIPELINE STATS SUMMARY")
    print("=" * 60)
    print(f"Overall Pass Ratio: {summary.get('overall_pass_ratio', 0.0)*100:.1f}%")
    print(f"Passed Samples:     {summary.get('passed_samples')} / {summary.get('total_samples')}")
    print(f"Time Taken:         {summary.get('execution_duration_sec', 0.0):.4f} seconds")
    print("-" * 60)
    
    for metric_name, stats in summary.get("metrics", {}).items():
        print(f"Metric: {metric_name:<20} | Average Score: {stats['mean_score']:.4f} | Pass Rate: {stats['pass_ratio']*100:.1f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_demo()
