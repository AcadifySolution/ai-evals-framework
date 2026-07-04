import argparse
import json
import os
import sys
from typing import List
from evals.config import EvaluationConfig
from evals.dataset import GroundTruthDataset
from evals.evaluator import Evaluator
from evals.metrics.hallucination import HallucinationMetric
from evals.metrics.semantic import SemanticSimilarityMetric
from evals.metrics.accuracy import AccuracyMetric
from evals.metrics.toxicity import ToxicityMetric
from evals.metrics.base import BaseMetric


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Evals Framework CLI - Benchmark LLM outputs with rigorous testing metrics."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to the dataset file (JSON, JSONL, or CSV)"
    )
    parser.add_argument(
        "--output",
        default="evaluation_report.json",
        help="Path to save the detailed evaluation output JSON"
    )
    parser.add_argument(
        "--metrics",
        default="hallucination,semantic_similarity,accuracy,toxicity",
        help="Comma-separated list of metrics to evaluate"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        help="Overwrite environment concurrency limits"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # 1. Load Configurations
    config = EvaluationConfig.load_from_env()
    if args.concurrency:
        config.concurrency_limit = args.concurrency

    # 2. Parse and Validate Dataset
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file not found at {args.dataset}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset from: {args.dataset}")
    ext = os.path.splitext(args.dataset)[1].lower()
    try:
        if ext == ".json":
            dataset = GroundTruthDataset.load_from_json(args.dataset)
        elif ext == ".jsonl":
            dataset = GroundTruthDataset.load_from_jsonl(args.dataset)
        elif ext in [".csv", ".txt"]:
            dataset = GroundTruthDataset.load_from_csv(args.dataset)
        else:
            print(f"Unsupported file format '{ext}'. Must be .json, .jsonl, or .csv", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Failed to load dataset: {str(e)}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(dataset.samples)} evaluation samples successfully.")

    # 3. Instantiate Metrics
    requested_metrics = [m.strip().lower() for m in args.metrics.split(",")]
    metrics: List[BaseMetric] = []

    if "hallucination" in requested_metrics:
        metrics.append(HallucinationMetric(threshold=config.thresholds.hallucination_threshold, config=config))
    if "semantic_similarity" in requested_metrics:
        metrics.append(SemanticSimilarityMetric(threshold=config.thresholds.semantic_similarity_threshold, config=config))
    if "accuracy" in requested_metrics:
        metrics.append(AccuracyMetric(threshold=config.thresholds.correctness_threshold))
    if "toxicity" in requested_metrics:
        metrics.append(ToxicityMetric(threshold=config.thresholds.toxicity_threshold, config=config))

    if not metrics:
        print("Error: No valid metrics selected for evaluation.", file=sys.stderr)
        sys.exit(1)

    # 4. Execute Evaluation
    evaluator = Evaluator(metrics=metrics, config=config)
    detailed_results, summary = evaluator.evaluate_dataset(dataset)

    # 5. Output Report Details
    final_output = {
        "summary": summary,
        "results": detailed_results
    }

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2)
        print(f"\nSaved detailed evaluation report to: {args.output}")
    except Exception as e:
        print(f"Failed to save output report: {str(e)}", file=sys.stderr)

    # 6. Display Dashboard Summary Table
    print("\n" + "=" * 65)
    print("                 EVALUATION RUN REPORT SUMMARY")
    print("=" * 65)
    print(f"Project ID:       {summary.get('project_id')}")
    print(f"Total Samples:    {summary.get('total_samples')}")
    print(f"Passed Samples:   {summary.get('passed_samples')} / {summary.get('total_samples')} "
          f"({summary.get('overall_pass_ratio', 0.0)*100:.1f}%)")
    print(f"Duration:         {summary.get('execution_duration_sec', 0.0):.2f} seconds")
    print("-" * 65)
    print(f"{'Metric Name':<25} | {'Mean Score':<12} | {'Pass Ratio':<12} | {'Total Runs':<10}")
    print("-" * 65)
    
    metrics_summary = summary.get("metrics", {})
    for m_name, stats in metrics_summary.items():
        print(f"{m_name:<25} | {stats['mean_score']:<12.4f} | {stats['pass_ratio']*100:<10.1f}% | {stats['total_runs']:<10}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
