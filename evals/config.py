from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel, Field, SecretStr, field_validator


class LLMConfig(BaseModel):
    provider: str = Field(default="openai")
    model_name: str = Field(default="gpt-4o-mini", min_length=1, max_length=200)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    api_key: Optional[SecretStr] = None
    api_base: Optional[str] = None
    max_tokens: int = Field(default=1000, ge=1, le=10000)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        value = value.strip().lower()
        allowed = {"openai", "anthropic", "custom"}
        if value not in allowed:
            raise ValueError(f"provider must be one of: {', '.join(sorted(allowed))}")
        return value


class DatabaseConfig(BaseModel):
    connection_string: Optional[SecretStr] = None
    enabled: bool = False
    project_id: str = Field(default="default-project", min_length=1, max_length=200)


class MetricThresholds(BaseModel):
    hallucination_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    semantic_similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    correctness_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    toxicity_threshold: float = Field(default=0.1, ge=0.0, le=1.0)


class EvaluationConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    db: DatabaseConfig = Field(default_factory=DatabaseConfig)
    thresholds: MetricThresholds = Field(default_factory=MetricThresholds)
    concurrency_limit: int = Field(default=5, ge=1, le=64)
    verbose: bool = True

    @classmethod
    def load_from_env(cls) -> "EvaluationConfig":
        return cls(
            llm=LLMConfig(
                provider=os.getenv("EVALS_LLM_PROVIDER", "openai"),
                model_name=os.getenv("EVALS_LLM_MODEL", "gpt-4o-mini"),
                temperature=float(os.getenv("EVALS_LLM_TEMPERATURE", "0.0")),
                api_key=SecretStr(os.environ["OPENAI_API_KEY"]) if os.getenv("OPENAI_API_KEY") else None,
                api_base=os.getenv("EVALS_LLM_API_BASE"),
                max_tokens=int(os.getenv("EVALS_LLM_MAX_TOKENS", "1000")),
            ),
            db=DatabaseConfig(
                connection_string=SecretStr(os.environ["EVALS_DB_CONNECTION"]) if os.getenv("EVALS_DB_CONNECTION") else None,
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
