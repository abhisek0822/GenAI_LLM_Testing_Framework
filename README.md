# GenAI / LLM Testing Framework

Production-style Python 3.11 framework for testing chatbots, RAG systems, summarizers, prompt-response APIs, and agent-like GenAI workflows. It supports deterministic API assertions and probabilistic quality evaluation, while staying runnable locally without real LLM credentials.

## Architecture

Test data flows through `pytest`, a client adapter, the system under test, response capture, evaluator engines, threshold gates, and reports. The repository keeps those responsibilities separate:

- `clients/`: mock and HTTP API clients with latency capture, auth, timeout, and error handling.
- `models/`: shared GenAI test case, API response, and evaluation result models.
- `evaluators/`: deterministic checks, safety checks, semantic similarity, diagnostic logic, and optional DeepEval/RAGAS adapters.
- `datasets/`: golden JSON datasets with readable IDs such as `TC_RAG_001`.
- `tests/`: pytest automation using dataset parameterization.
- `scripts/`: local evaluation and regression comparison entrypoints.
- `mock_app/`: optional FastAPI mock system under test.
- `ci/`: Azure Pipelines example.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/run_evaluation.py datasets/chatbot_cases.json datasets/rag_cases.json datasets/summarization_cases.json datasets/safety_cases.json
```

Run the optional mock API:

```bash
uvicorn mock_app.main:app --reload
```

Switch from mock mode to a real API by setting:

```bash
GENAI_MODE=real
GENAI_BASE_URL=https://your-api.example.com
GENAI_API_TOKEN=...
```

Never commit a real `.env`; use `.env.example` as the template.

## Evaluation Strategy

Use deterministic evaluation whenever a reliable assertion is available: exact match, contains, regex, JSON fields, required keywords, prohibited strings, refusal phrases, numeric checks, status codes, and schema validation.

Use semantic similarity when wording can vary but meaning is comparable.

Use LLM-as-a-judge when contextual or subjective reasoning is required: faithfulness, relevancy, completeness, professional tone, role adherence, summarization quality, and business-rule adherence.

## DeepEval Integration

The optional adapter in `evaluators/deepeval_evaluator.py` follows the current DeepEval single-turn pattern from official docs: create an `LLMTestCase` with `input`, `actual_output`, `expected_output`, and `retrieval_context`, then measure metrics such as `AnswerRelevancyMetric` and `FaithfulnessMetric`. Custom rubric-based judging can use `GEval` with explicit `evaluation_steps`.

Install optional dependencies and configure evaluator credentials before using these live judge metrics:

```bash
pip install -e ".[eval]"
export EVALUATOR_API_KEY=...
export EVALUATOR_MODEL=gpt-4o-mini
```

## RAGAS Integration

The optional RAGAS adapter maps this framework’s common model to modern RAGAS fields:

| Framework | RAGAS |
| --- | --- |
| `input` | `user_input` |
| `actual_output` | `response` |
| `expected_output` | `reference` |
| `retrieved_context` | `retrieved_contexts` |

The adapter is designed for metrics such as faithfulness, response relevancy, context precision, and context recall. Some RAGAS metrics need real LLM/embedding configuration and are best run as marked external evaluation jobs, not as default CI smoke tests.

## Diagnostics

RAG failures are classified with human-readable guidance:

- Low context precision or recall: retrieval issue.
- Good context retrieval with low faithfulness: generation issue.
- High faithfulness with low correctness: source or golden-data issue.
- Good factual metrics with toxicity failure: safety issue.

## Dataset Coverage

The sample golden datasets include happy paths, edge cases, ambiguous/no-answer prompts, typos, prompt paraphrases, incomplete/conflicting/wrong context patterns, prompt injection, toxic prompts, bias-oriented checks, multi-turn-ready metadata, and summarization scenarios.

## CI/CD

`ci/azure-pipelines.yml` installs Python 3.11, runs pytest, and executes an offline quality evaluation against selected datasets. External evaluator runs should be separated behind credentials and marked with `external_eval`.

## Official API Verification

Before implementation, official docs were checked on 2026-09-29:

- [DeepEval Answer Relevancy](https://deepeval.com/docs/metrics-answer-relevancy)
- [DeepEval single-turn evaluation](https://deepeval.com/docs/evaluation-end-to-end-single-turn)
- [DeepEval G-Eval examples](https://deepeval.com/guides/guides-g-eval-examples)
- [RAGAS available metrics](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/)
- [RAGAS GitHub README](https://github.com/amarshp/ragas/blob/main/README.md)
