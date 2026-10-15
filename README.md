# AI Evals Framework — Python LLM and RAG Evaluation

[![Quality checks](https://github.com/AcadifySolution/ai-evals-framework/actions/workflows/quality.yml/badge.svg)](https://github.com/AcadifySolution/ai-evals-framework/actions/workflows/quality.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Evaluate language-model answers against versioned datasets with a Python CLI and
JSON reports. Use it to compare prompt changes, check RAG grounding, and inspect
answer accuracy, similarity, and pattern-based PII or toxicity signals.

## Try an offline evaluation

Python 3.10 or newer is required. This example uses the lexical accuracy metric;
it does not need a model-provider key or cloud infrastructure.

```bash
git clone https://github.com/AcadifySolution/ai-evals-framework.git
cd ai-evals-framework
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
ai-evals --dataset examples/dataset.json --metrics accuracy --output evaluation_report.json
```

Open `evaluation_report.json` for per-sample results and aggregate scores. The
included dataset is a demonstration, not a representative model benchmark.

## Choose the metric for your question

| Metric name           | What it checks                                                | Interpretation                                                       |
| --------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------- |
| `accuracy`            | Lexical overlap against expected answers                      | Wording changes can affect scores even when meaning is similar       |
| `hallucination`       | Grounding against supplied context, with optional LLM judging | The local fallback uses overlap heuristics, not factual verification |
| `semantic_similarity` | Answer similarity, with optional provider embeddings          | The offline fallback is lexical and can miss paraphrases             |
| `toxicity`            | Configured toxic-language and PII patterns                    | Pattern matches are screening signals, not comprehensive detection   |

Select several names with `--metrics accuracy,hallucination`. Provider-backed
metrics require configuration and can transmit evaluation content to a provider.
Review [configuration](evals/config.py) and [security guidance](SECURITY.md) before
using sensitive datasets. Scores do not establish regulatory compliance or prove
that an answer is safe or true.

## Explore the implementation

- [Example dataset](examples/dataset.json): start with the expected sample format.
- [Dataset loaders](evals/dataset.py): JSON, JSONL, and CSV ingestion.
- [Evaluation engine](evals/evaluator.py): concurrent execution and ordered results.
- [CLI](evals/runner.py): metric selection, concurrency, and JSON output.
- [Metrics](evals/metrics/): inspect scoring behavior before setting thresholds.
- [Tests](tests/): regression examples and validation boundaries.
- [Infrastructure examples](terraform/): Terraform scaffolding, requiring a separate deployment review; not a prerequisite for local evaluation.

## Develop and contribute

```bash
ruff check .
black --check .
pytest -q
```

Use a small reproducible dataset when reporting a scoring issue. Include the
metric, expected behavior, configuration, and package version; remove secrets and
personal data. See [CONTRIBUTING.md](CONTRIBUTING.md).

For retrieval implementation, explore [Enterprise RAG Boilerplate](https://github.com/AcadifySolution/enterprise-rag-boilerplate).
For human review records, explore [ASR Feedback](https://github.com/AcadifySolution/asr-feedback).
If this tool is useful, star it to save it, fork it for experiments, or contribute
a reproducible issue or pull request.

## License

[MIT](LICENSE).
