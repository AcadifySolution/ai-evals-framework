# AI Evals Framework (`ai-evals-framework`)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![AWS SageMaker Supported](https://img.shields.io/badge/AWS-SageMaker-FF9900.svg)](https://aws.amazon.com/sagemaker/)
[![IaC: Terraform](https://img.shields.io/badge/Terraform-1.3%2B-844FBA.svg)](https://www.terraform.io/)
[![CI: Passing](https://img.shields.io/badge/CI-Passing-brightgreen.svg)]()

`ai-evals-framework` is an enterprise-grade, open-source benchmarking pipeline designed to rigorously evaluate Large Language Model (LLM) responses against ground-truth golden datasets. Built with scalability and customizability at its core, this framework offers standard metrics to measure **hallucinations (factual grounding)**, **semantic similarity**, **lexical correctness**, and **toxicity/PII leaks** in production LLM and RAG pipelines.

---

## Architecture Overview

The system architecture decouples the benchmarking orchestrator, metric evaluation logic, and execution runtime (CLI/API), allowing integration into CI/CD pipelines, automated evaluation workflows, or live shadow deployment audits.

```mermaid
flowchart TD
    %% Define Styles
    classDef dataset fill:#1A365D,stroke:#2B6CB0,stroke-width:2px,color:#E2E8F0;
    classDef runner fill:#2C7A7B,stroke:#319795,stroke-width:2px,color:#E2E8F0;
    classDef metrics fill:#2D3748,stroke:#4A5568,stroke-width:2px,color:#E2E8F0;
    classDef judge fill:#7B341E,stroke:#9B2C2C,stroke-width:2px,color:#E2E8F0;
    classDef infra fill:#553C9A,stroke:#6B46C1,stroke-width:2px,color:#E2E8F0;

    subgraph INGEST ["1. Data Ingestion Layer"]
        A[dataset.json / dataset.jsonl / dataset.csv]:::dataset
    end

    subgraph ENGINE ["2. Core Evaluator Engine"]
        B[GroundTruthDataset Parser]:::runner --> C[Threaded Evaluator Orchestrator]:::runner
        Config[EvaluationConfig]:::runner --> C
    end

    subgraph METRIC_PIPELINE ["3. Metrics & Scanners"]
        C --> D1[Hallucination Metric]:::metrics
        C --> D2[Semantic Similarity]:::metrics
        C --> D3[Lexical Accuracy]:::metrics
        C --> D4[Toxicity & PII Scanner]:::metrics
    end

    subgraph JUDGING_SERVICE ["4. Verification Services"]
        D1 --> E1{API Available?}:::judge
        E1 -- Yes --> LLM_Judge[OpenAI / SageMaker LLM Judge]:::judge
        E1 -- No --> Local_NLI[NLI Overlap Fallback Engine]:::judge
        
        D2 --> E2{API Available?}:::judge
        E2 -- Yes --> Embeddings[OpenAI Embeddings API]:::judge
        E2 -- No --> TFIDF[Local TF-IDF Vector Cosine Similarity]:::judge
        
        D3 --> Lexical_Funcs[ROUGE-L / BLEU-4 / Exact Match]:::metrics
        
        D4 --> E3{API Available?}:::judge
        E3 -- Yes --> OpenAI_Mod[OpenAI Moderation API]:::judge
        E3 -- No --> Regex_Filter[Local Regex PII & Profanity Filter]:::judge
    end

    subgraph OUTPUT_TELEMETRY ["5. Telemetry & Analytics"]
        LLM_Judge & Local_NLI & Embeddings & TFIDF & Lexical_Funcs & OpenAI_Mod & Regex_Filter --> F[Aggregate Report Summary]:::infra
        F --> G[CLI JSON Output]:::infra
        F --> H[PostgreSQL RDS Logging DB]:::infra
    end

    INGEST --> ENGINE
```

---

## Key Metrics & Evaluation Methodologies

### 1. Hallucination Detection (Factual Grounding)
Factual grounding evaluates whether the output contains unsubstantiated statements or hallucinations when compared to the context.
*   **LLM-as-a-Judge:** Segments the candidate output into individual statements and queries an evaluator LLM (using deterministic JSON-formatted schemas) to check if the statements are logically entailed by the retrieved context.
*   **NLI (Natural Language Inference) Fallback:** Computes a sentence-level lexical and structural keyword overlap score, ensuring validation runs successfully even when external APIs are disconnected.

### 2. Semantic Similarity
Measures whether the core meaning of the output aligns with the golden ground truth.
*   **Vector Cosine Similarity:** Embeds the ground truth and model output using OpenAI's `text-embedding-3-small` and calculates the cosine similarity:
    $$\text{Cosine Similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$
*   **Local Bag-of-Words Fallback:** Employs a term-frequency cosine vector calculation for fast, zero-cost semantic verification.

### 3. Lexical Accuracy
Checks structural and precise vocabulary alignment using classic NLP algorithms:
*   **Exact Match (EM):** Case-insensitive string match.
*   **ROUGE-L:** Computes the Longest Common Subsequence (LCS) to capture sentence-level word-order recall and precision.
*   **BLEU:** Measures modified n-gram precision against references combined with a brevity penalty.

### 4. Toxicity & PII Scanner
Ensures safety, compliance, and prevents data leaks.
*   **OpenAI Moderation:** Hooks into the OpenAI Moderation endpoint to flag harassment, hate, self-harm, sexual, and violent content.
*   **Regex PII Filter:** Scans output string for email structures, North American phone formats, and Social Security Numbers (SSN).
*   **Profanity Filter:** Flags localized toxic language keywords.

---

## Directory Structure

```
ai-evals-framework/
├── LICENSE                        # MIT Open Source License
├── README.md                      # Architecture documentation
├── requirements.txt               # Installation requirements
├── pyproject.toml                 # Standard python project metadata
├── evals/
│   ├── __init__.py                # Package exports
│   ├── config.py                  # Pydantic environment configurations
│   ├── dataset.py                 # GroundTruthDataset and Pydantic validators
│   ├── evaluator.py               # Concurrent ThreadPool executor engine
│   ├── runner.py                  # CLI argument parser and dashboard formatter
│   └── metrics/
│       ├── __init__.py            # Metrics registry
│       ├── base.py                # Abstract class BaseMetric & MetricResult
│       ├── hallucination.py       # LLM judge and NLI fallback checks
│       ├── semantic.py            # Cosine similarity and TF-IDF fallbacks
│       ├── accuracy.py            # ROUGE-L, BLEU, and Exact Match calculations
│       └── toxicity.py            # PII leaks, regex and Moderation filters
├── terraform/
│   ├── main.tf                    # VPC networks & RDS PostgreSQL setup
│   ├── sagemaker.tf               # SageMaker LLM Endpoint (Llama-3 TGI)
│   ├── variables.tf               # IaC input variables
│   └── outputs.tf                 # Resource ARNs and connection strings
├── examples/
│   ├── dataset.json               # Mock test benchmark samples
│   └── run_eval.py                # Quickstart execution script
└── tests/
    └── test_evaluator.py          # Pytest validation suite
```

---

## Getting Started

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/ai-evals-framework.git
cd ai-evals-framework
pip install -r requirements.txt
```

To install the framework as an editable CLI command:

```bash
pip install -e .
```

### Environment Variables

Configure evaluation thresholds and credentials in a `.env` file or export them to your environment:

```bash
# LLM Judge Configurations
export OPENAI_API_KEY="sk-proj-..."
export EVALS_LLM_PROVIDER="openai"
export EVALS_LLM_MODEL="gpt-4o-mini"
export EVALS_LLM_TEMPERATURE="0.0"

# Target Limits
export EVALS_CONCURRENCY_LIMIT="5"

# Database Telemetry Storage (Optional)
export EVALS_DB_ENABLED="true"
export EVALS_DB_CONNECTION="postgresql://evaladmin:SuperSecurePassword123@localhost:5432/aievalsdb"
```

---

## Usage Guide

### CLI Command execution

Use the `ai-evals` command to execute evaluations:

```bash
ai-evals --dataset examples/dataset.json --output results.json --metrics hallucination,semantic_similarity,accuracy,toxicity
```

### Programmatic API

Import the evaluator directly into your scripts or unit test assertions:

```python
from evals import GroundTruthDataset, EvaluationSample, Evaluator
from evals.metrics import HallucinationMetric, SemanticSimilarityMetric

# 1. Define sample payload
sample = EvaluationSample(
    query="What is the capital of France?",
    context="France is in Western Europe. Its capital is Paris.",
    ground_truth="Paris",
    generated_output="The capital of France is Paris."
)
dataset = GroundTruthDataset([sample])

# 2. Configure metrics
metrics = [
    HallucinationMetric(threshold=0.8),
    SemanticSimilarityMetric(threshold=0.75)
]

# 3. Run evaluation
evaluator = Evaluator(metrics=metrics)
detailed_results, summary = evaluator.evaluate_dataset(dataset)

print(f"Overall Pass Ratio: {summary['overall_pass_ratio'] * 100}%")
```

---

## Infrastructure Deployment (AWS SageMaker & RDS)

The `terraform/` directory provides production-ready Infrastructure-as-Code (IaC) to host open-source LLMs under test on AWS SageMaker and capture persistent telemetry in an RDS database.

### Features
*   **VPC Architecture:** Creates public/private subnets across multiple Availability Zones, locking down the database in the private subnets.
*   **SageMaker LLM Endpoint:** Deploys a Hugging Face Text Generation Inference (TGI) Docker container hosting gated models (e.g., Llama-3-8B-Instruct) on GPU instance types (`ml.g5.2xlarge`).
*   **RDS PostgreSQL:** Provisions database nodes to store execution statistics and logs.

### Steps to Deploy

1. Initialize Terraform:
    ```bash
    cd terraform
    terraform init
    ```

2. Plan deployment:
    ```bash
    terraform plan -var="huggingface_api_token=your_hf_read_token"
    ```

3. Deploy resources:
    ```bash
    terraform apply -var="huggingface_api_token=your_hf_read_token" -auto-approve
    ```

---

## Development & Verification

Run the test suite using pytest to verify that all metrics, fallback calculations, and parsing work correctly:

```bash
pytest tests/
```

---

## License

This project is licensed under the MIT License. See [LICENSE](file:///Users/acadify/Desktop/ai-evals-framework/LICENSE) for more details.
