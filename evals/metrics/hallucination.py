import os
import json
import logging
from typing import Dict, Any, List
from evals.dataset import EvaluationSample
from evals.metrics.base import BaseMetric, MetricResult
from evals.config import EvaluationConfig

logger = logging.getLogger(__name__)


class HallucinationMetric(BaseMetric):
    """Evaluates whether the generated response contains facts unsupported by the context."""

    def __init__(self, threshold: float = 0.8, config: EvaluationConfig = None):
        super().__init__(threshold=threshold)
        self.config = config or EvaluationConfig.load_from_env()

    @property
    def name(self) -> str:
        return "hallucination"

    def evaluate(self, sample: EvaluationSample) -> MetricResult:
        """Runs hallucination check using configured method.
        
        Falls back to lexical entailment checking if LLM endpoints are not configured.
        """
        context = sample.context_as_str
        output = sample.generated_output

        if not context:
            return MetricResult(
                name=self.name,
                score=1.0,
                passed=True,
                reason="Skipped: Context is empty. Cannot determine hallucination without reference context.",
                metadata={"method": "skipped"}
            )

        # Retrieve OpenAI key from configuration or environment
        api_key = self.config.llm.api_key or os.getenv("OPENAI_API_KEY")

        if api_key and self.config.llm.provider == "openai":
            return self._evaluate_via_llm_judge(context, output)
        else:
            return self._evaluate_via_nli_fallback(context, output)

    def _evaluate_via_llm_judge(self, context: str, output: str) -> MetricResult:
        """Uses an external LLM as a judge to assess factual grounding."""
        # This showcases how a real system parses LLM responses into structured evaluations
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self.config.llm.api_key.get_secret_value() if self.config.llm.api_key else os.getenv("OPENAI_API_KEY"),
                base_url=self.config.llm.api_base
            )

            # System prompt instructing the LLM to perform exact statement extraction and validation
            system_prompt = (
                "You are an expert AI quality auditor. Your task is to check a candidate LLM response "
                "against a retrieved reference context for hallucinations. Follow these steps:\n"
                "1. Segment the candidate response into individual factual statements.\n"
                "2. For each statement, verify if it is directly supported, partially supported, or contradicted "
                "by the reference context.\n"
                "3. Compute a factual consistency score: (number of supported statements) / (total statements).\n"
                "4. Output your analysis ONLY as a valid JSON object matching the following structure:\n"
                "{\n"
                "  \"score\": 0.95,\n"
                "  \"total_statements\": 5,\n"
                "  \"supported_statements\": 4,\n"
                "  \"reason\": \"Detailed explanation of any hallucinated or unsupported statements found.\",\n"
                "  \"unsupported_statements\": [\"Specific statement not found in context\"]\n"
                "}"
            )

            user_prompt = f"### REFERENCE CONTEXT:\n{context}\n\n### CANDIDATE RESPONSE:\n{output}\n"

            response = client.chat.completions.create(
                model=self.config.llm.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                response_format={"type": "json_object"}
            )

            result_data = json.loads(response.choices[0].message.content)
            score = float(result_data.get("score", 0.0))
            reason = result_data.get("reason", "Evaluation complete.")
            
            return MetricResult(
                name=self.name,
                score=score,
                passed=score >= self.threshold,
                reason=reason,
                metadata={
                    "method": "llm_as_judge",
                    "model": self.config.llm.model_name,
                    "total_statements": result_data.get("total_statements"),
                    "unsupported_statements": result_data.get("unsupported_statements", [])
                }
            )

        except ImportError:
            logger.warning("OpenAI SDK not installed. Falling back to NLI overlap model.")
            return self._evaluate_via_nli_fallback(context, output, "OpenAI SDK missing. Fallback applied.")
        except Exception as e:
            logger.error(f"LLM-as-a-judge exception: {str(e)}")
            return self._evaluate_via_nli_fallback(context, output, f"LLM-as-a-judge error: {str(e)}. Fallback applied.")

    def _evaluate_via_nli_fallback(self, context: str, output: str, fallback_reason: str = "Demo Fallback") -> MetricResult:
        """Calculates score based on keyword semantic alignment and sentence-level overlap.
        
        This serves as a mock Natural Language Inference skeleton model in the absence of a live LLM endpoint.
        """
        # Split candidate response into sentences
        import re
        sentences = [s.strip() for s in re.split(r'[.!?]+', output) if s.strip()]
        
        if not sentences:
            return MetricResult(
                name=self.name,
                score=1.0,
                passed=True,
                reason="Candidate output is empty or lacks clear sentences.",
                metadata={"method": "nli_fallback"}
            )

        # Lexical matching over simple lowercase strings
        context_lower = context.lower()
        supported = 0
        unsupported = []
        total_overlap = 0.0

        for sentence in sentences:
            sentence_clean = re.sub(r'[^\w\s]', '', sentence.lower())
            words = sentence_clean.split()
            if not words:
                supported += 1
                total_overlap += 1.0
                continue
            
            # Simple keyword coverage metric
            matched_words = [w for w in words if w in context_lower]
            overlap_ratio = len(matched_words) / len(words)
            total_overlap += overlap_ratio
            
            # Threshold to consider sentence supported (heuristic)
            if overlap_ratio >= 0.75:
                supported += 1
            else:
                unsupported.append(sentence)

        score = total_overlap / len(sentences)

        # Heuristic adjustment for complete overlaps
        if all(word in context_lower for word in re.sub(r'[^\w\s]', '', output.lower()).split()[:15]):
            score = max(score, 0.95)

        passed = score >= self.threshold
        reason = (
            f"Factual consistency score determined via word overlap analysis ({fallback_reason}). "
            f"Validated {supported}/{len(sentences)} sentence tokens."
        )
        if unsupported:
            reason += f" Suspected unsupported segments: {unsupported[:2]}..."

        return MetricResult(
            name=self.name,
            score=score,
            passed=passed,
            reason=reason,
            metadata={
                "method": "nli_overlap_fallback",
                "unsupported_sentences": unsupported,
                "overlap_ratio": score
            }
        )
