import re
from typing import Dict, Any, List
from evals.dataset import EvaluationSample
from evals.metrics.base import BaseMetric, MetricResult


class AccuracyMetric(BaseMetric):
    """Computes lexical correctness metrics: Exact Match, BLEU, and ROUGE."""

    def __init__(self, threshold: float = 0.7, primary_metric: str = "rouge"):
        super().__init__(threshold=threshold)
        self.primary_metric = primary_metric

    @property
    def name(self) -> str:
        return "accuracy"

    def evaluate(self, sample: EvaluationSample) -> MetricResult:
        ground_truth = sample.ground_truth
        output = sample.generated_output

        if not ground_truth:
            return MetricResult(
                name=self.name,
                score=0.0,
                passed=False,
                reason="Skipped: Ground truth is not provided. Cannot compute lexical accuracy.",
                metadata={"method": "skipped"}
            )

        # Tokenize text
        gt_tokens = self._tokenize(ground_truth)
        out_tokens = self._tokenize(output)

        if not gt_tokens or not out_tokens:
            return MetricResult(
                name=self.name,
                score=0.0,
                passed=False,
                reason="Tokenization yielded empty sequences for ground truth or output.",
                metadata={}
            )

        # 1. Exact Match (case insensitive)
        exact_match = 1.0 if ground_truth.strip().lower() == output.strip().lower() else 0.0

        # 2. ROUGE-L approximation (longest common subsequence)
        lcs_len = self._lcs(gt_tokens, out_tokens)
        rouge_p = lcs_len / len(out_tokens)
        rouge_r = lcs_len / len(gt_tokens)
        rouge_l = (2 * rouge_p * rouge_r) / (rouge_p + rouge_r) if (rouge_p + rouge_r) > 0 else 0.0

        # 3. BLEU approximation (simple unigram/bigram overlap precision with brevity penalty)
        bleu = self._bleu_approx(gt_tokens, out_tokens)

        # Select primary score
        score = rouge_l if self.primary_metric == "rouge" else (bleu if self.primary_metric == "bleu" else exact_match)
        passed = score >= self.threshold

        reason = (
            f"Lexical comparison completed using {self.primary_metric}. "
            f"Exact Match: {exact_match:.2f}, ROUGE-L: {rouge_l:.2f}, BLEU: {bleu:.2f}."
        )

        return MetricResult(
            name=self.name,
            score=score,
            passed=passed,
            reason=reason,
            metadata={
                "exact_match": exact_match,
                "rouge_l": rouge_l,
                "bleu": bleu,
                "primary_metric": self.primary_metric
            }
        )

    def _tokenize(self, text: str) -> List[str]:
        """Lowers text and extracts word tokens."""
        return re.sub(r'[^\w\s]', '', text.lower()).split()

    def _lcs(self, x: List[str], y: List[str]) -> int:
        """Computes Longest Common Subsequence length using dynamic programming."""
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def _bleu_approx(self, reference: List[str], candidate: List[str]) -> float:
        """Computes a simplified BLEU score (unigram and bigram overlap with brevity penalty)."""
        # Unigrams
        ref_unigrams = set(reference)
        cand_unigrams = candidate
        unigram_matches = sum(1 for w in cand_unigrams if w in ref_unigrams)
        p1 = unigram_matches / len(candidate) if candidate else 0

        # Bigrams
        ref_bigrams = set(zip(reference[:-1], reference[1:]))
        cand_bigrams = list(zip(candidate[:-1], candidate[1:]))
        bigram_matches = sum(1 for bg in cand_bigrams if bg in ref_bigrams)
        p2 = bigram_matches / len(cand_bigrams) if cand_bigrams else 0

        # Geometric mean of precisions
        import math
        if p1 == 0:
            return 0.0
        
        # Heuristic combination
        p_geom = math.sqrt(p1 * (p2 if p2 > 0 else p1))

        # Brevity Penalty
        c = len(candidate)
        r = len(reference)
        if c > r:
            bp = 1.0
        else:
            bp = math.exp(1 - (r / c)) if c > 0 else 0.0

        return bp * p_geom
