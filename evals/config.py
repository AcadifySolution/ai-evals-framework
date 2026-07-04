import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, SecretStr


class LLMConfig(BaseModel):
    """Configuration for LLMs used as judges or evaluators."""
    provider: str = Field(default="openai", description="LLM provider (e.g., openai, anthropic, custom)")
    model_name: str = Field(default="gpt-4o-mini", description="Model identifier for evaluations")
    temperature: float = Field(default=0.0, description="Temperature parameter for deterministic grading")
    api_key: Optional[SecretStr] = Field(default=None, description="API Key for the provider (fallback to env)")
    api_base: Optional[str] = Field(default=None, description="Custom API endpoint (useful for self-hosted or proxy)")
    max_tokens: int = Field(default=1000, description="Max tokens to generate in evaluator outputs")


class DatabaseConfig(BaseModel):
    """Database configuration for storing persistent execution logs."""
    connection_string: Optional[str] = Field(default=None, description="DB Connection string (SQLAlchemy compatible)")
    enabled: bool = Field(default=False, description="Whether to persist results in external database")
    project_id: str = Field(default="default-project", description="Project identifier to group evaluation runs")


class MetricThresholds(BaseModel):
    """Threshold standards to pass evaluation assertions."""
    hallucination_threshold: float = Field(default=0.8, description="Minimum score to consider response hallucination-free")
    semantic_similarity_threshold: float = Field(default=0.75, description="Minimum embedding similarity score")
    correctness_threshold: float = Field(default=0.7, description="Minimum accuracy score (lexical or LLM grading)")
    toxicity_threshold: float = Field(default=0.1, description="Maximum tolerance score for toxic content")


class EvaluationConfig(BaseModel):
    """Root evaluation configurations."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    db: DatabaseConfig = Field(default_factory=DatabaseConfig)
    thresholds: MetricThresholds = Field(default_factory=MetricThresholds)
    concurrency_limit: int = Field(default=5, description="Max concurrent threads for remote evaluation calls")
    verbose: bool = Field(default=True, description="Enable detailed logging of evaluations")

    @classmethod
    def load_from_env(cls) -> "EvaluationConfig":
        """Instantiates configurations from environment variables."""
        return cls(
            llm=LLMConfig(
                provider=os.getenv("EVALS_LLM_PROVIDER", "openai"),
                model_name=os.getenv("EVALS_LLM_MODEL", "gpt-4o-mini"),
                temperature=float(os.getenv("EVALS_LLM_TEMPERATURE", "0.0")),
                api_base=os.getenv("EVALS_LLM_API_BASE"),
            ),
            db=DatabaseConfig(
                connection_string=os.getenv("EVALS_DB_CONNECTION"),
                enabled=os.getenv("EVALS_DB_ENABLED", "false").lower() == "true",
                project_id=os.getenv("EVALS_PROJECT_ID", "default-project"),
            ),
            thresholds=MetricThresholds(
                hallucination_threshold=float(os.getenv("EVALS_HALLUCINATION_THRESHOLD", "0.8")),
                semantic_similarity_threshold=float(os.getenv("EVALS_SEMANTIC_THRESHOLD", "0.75")),
                correctness_threshold=float(os.getenv("EVALS_CORRECTNESS_THRESHOLD", "0.7")),
                toxicity_threshold=float(os.getenv("EVALS_TOXICITY_THRESHOLD", "0.1")),
            ),
            concurrency_limit=int(os.getenv("EVALS_CONCURRENCY_LIMIT", "5")),
            verbose=os.getenv("EVALS_VERBOSE", "true").lower() == "true",
        )
