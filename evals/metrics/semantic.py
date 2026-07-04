import os
import logging
import numpy as np
from evals.dataset import EvaluationSample
from evals.metrics.base import BaseMetric, MetricResult
from evals.config import EvaluationConfig

logger = logging.getLogger(__name__)


class SemanticSimilarityMetric(BaseMetric):
    """Measures semantic similarity between candidate output and ground truth."""

    def __init__(self, threshold: float = 0.75, config: EvaluationConfig = None):
        super().__init__(threshold=threshold)
        self.config = config or EvaluationConfig.load_from_env()

    @property
    def name(self) -> str:
        return "semantic_similarity"

    def evaluate(self, sample: EvaluationSample) -> MetricResult:
        """Computes semantic similarity score."""
        ground_truth = sample.ground_truth
        output = sample.generated_output

        if not ground_truth:
            return MetricResult(
                name=self.name,
                score=0.0,
                passed=False,
                reason="Skipped: Ground truth is not provided. Cannot compute similarity.",
                metadata={"method": "skipped"}
            )

        api_key = self.config.llm.api_key or os.getenv("OPENAI_API_KEY")

        if api_key and self.config.llm.provider == "openai":
            return self._evaluate_via_openai_embeddings(ground_truth, output)
        else:
            return self._evaluate_via_local_fallback(ground_truth, output)

    def _evaluate_via_openai_embeddings(self, ground_truth: str, output: str) -> MetricResult:
        """Retrieves OpenAI Embeddings and computes cosine similarity."""
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self.config.llm.api_key.get_secret_value() if self.config.llm.api_key else os.getenv("OPENAI_API_KEY"),
                base_url=self.config.llm.api_base
            )

            # Get embedding representation of both texts
            res = client.embeddings.create(
                input=[ground_truth, output],
                model="text-embedding-3-small"
            )

            emb_gt = np.array(res.data[0].embedding)
            emb_out = np.array(res.data[1].embedding)

            # Cosine similarity calculation
            dot_product = np.dot(emb_gt, emb_out)
            norm_gt = np.linalg.norm(emb_gt)
            norm_out = np.linalg.norm(emb_out)
            
            similarity = float(dot_product / (norm_gt * norm_out))
            passed = similarity >= self.threshold

            return MetricResult(
                name=self.name,
                score=similarity,
                passed=passed,
                reason=f"Embedding cosine similarity computed using OpenAI text-embedding-3-small.",
                metadata={
                    "method": "openai_embeddings",
                    "similarity": similarity
                }
            )

        except ImportError:
            logger.warning("OpenAI SDK not installed. Falling back to local vectorizer.")
            return self._evaluate_via_local_fallback(ground_truth, output, "OpenAI SDK missing. Fallback applied.")
        except Exception as e:
            logger.error(f"Failed to fetch embeddings: {str(e)}")
            return self._evaluate_via_local_fallback(ground_truth, output, f"API Exception: {str(e)}. Fallback applied.")

    def _evaluate_via_local_fallback(self, ground_truth: str, output: str, fallback_reason: str = "Demo Fallback") -> MetricResult:
        """Calculates similarity using a lightweight TF-IDF-like word-vector representation."""
        import re
        from collections import Counter

        def text_to_vector(text: str):
            words = re.sub(r'[^\w\s]', '', text.lower()).split()
            return Counter(words)

        def get_cosine_sim(vec1, vec2):
            intersection = set(vec1.keys()) & set(vec2.keys())
            numerator = sum([vec1[x] * vec2[x] for x in intersection])

            sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
            sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
            denominator = np.sqrt(sum1) * np.sqrt(sum2)

            if not denominator:
                return 0.0
            return float(numerator) / denominator

        v1 = text_to_vector(ground_truth)
        v2 = text_to_vector(output)
        
        similarity = get_cosine_sim(v1, v2)
        passed = similarity >= self.threshold

        return MetricResult(
            name=self.name,
            score=similarity,
            passed=passed,
            reason=f"Lexical cosine vector similarity calculated ({fallback_reason}).",
            metadata={
                "method": "local_tf_idf_fallback",
                "similarity": similarity
            }
        )
