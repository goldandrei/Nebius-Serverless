# Nebius Serverless Eval Dashboard

Compare LLMs side by side on your own tasks, running against real Nebius
infrastructure — no mocks, no simulated scores. A local dashboard lets you pick
a scoring method, enter data (or use a built-in example), choose up to 3
models, and get a leaderboard with accuracy, latency, and cost.

![Model comparison leaderboard](docs/dashboard.png)

*Leaderboard, score-vs-cost and latency charts, and per-sample answers after
comparing three models on the NL → JSON Command task.*

## Backends

Each model is routed automatically by its catalog entry — a single comparison
can mix both:

- **Token Factory** (`basis: hosted`) — Nebius's hosted inference API. No GPU
  provisioning, billed per token, starts immediately.
- **Serverless Endpoints** (`basis: self-hosted`) — deploys a GPU VM on demand
  (vLLM), runs the eval, tears it down. Billed per GPU-hour from create to
  delete. Takes 5–20 minutes to provision.

![Self-hosted model picker](docs/backends-endpoint.png)

*Self-hosted (Serverless Endpoint) models, grouped by family, with GPU preset
and hourly rate.*

![Token Factory model picker](docs/backends-tokenfactory.png)

*Instant (Token Factory) models — no GPU deployment, priced per input/output
token.*

## Scoring methods

- **Programmatic** — exact string match, numeric extraction, or F1 over
  JSON fields. No API calls beyond the model itself.
- **Reference match** — lexical (token F1 word overlap) or embedding
  (cosine similarity) comparison against a reference answer.
- **LLM judge** — a judge model (default `deepseek-ai/DeepSeek-V4-Pro`, configurable via
  `JUDGE_MODEL`) grades each answer 1–N against a rubric you write. Judge
  token cost is tracked separately from the models being compared.

![Choosing a scoring method](docs/scoring-methods.png)

*Picking a scoring method and entering data in the dataset builder — here,
programmatic scoring with json_fields comparison.*

## What this demonstrates

- **Deploy tax is real and capacity-dependent** — self-hosted endpoints take
  5–20 min to provision, so the cost model splits deploy cost from eval cost
  using create/ready/eval-done timestamps rather than lumping them together.
- **Self-hosting only wins at high utilization** — hosted per-token pricing is
  cheaper at low request volume; the crossover point depends on how many
  requests amortize the GPU-hour.
- **Reasoning models break naive scoring** — their `<think>` blocks must be
  stripped before every scoring path (exact, numeric, JSON-fields, reference
  match, and the judge), or correct answers score zero or get silently
  corrupted.
- **Embedding-based reference scoring has a length bias** that can invert
  rankings, so the dashboard flags it as a caveat.
- **LLM-judge cost is tracked separately** from the models being compared —
  it's meta-evaluation cost, not a contestant's cost.

## Setup

**Prerequisites:**
- Python 3.10 or newer.
- For self-hosted (endpoint) models only: the Nebius CLI must be installed
  and authenticated (the harness calls it to provision and tear down GPU
  endpoints). On Windows it runs via WSL. Not needed if you only compare
  hosted (Token Factory) models.

```bash
cp .env.example .env
# fill in NEBIUS_API_KEY, NEBIUS_PROJECT_ID, NEBIUS_SUBNET_ID, NEBIUS_REGION
# HF_TOKEN only needed for gated models you self-host
pip install -r requirements.txt
```

`NEBIUS_API_KEY` is required for both backends (Token Factory inference and
the embedding/judge scorers). `NEBIUS_PROJECT_ID`/`NEBIUS_SUBNET_ID` are only
needed if you compare any self-hosted (endpoint) models.

## Running

```bash
make serve
# or: python scripts/server.py [port]
```

Open `http://localhost:7860`. Pick a scoring method and enter data (or click
a built-in example), select up to 3 models, click **Run comparison**.

If a Serverless Endpoint deploy fails, is interrupted, or you close the tab
mid-run, `make clean` (or `scripts/cleanup.sh`) deletes any endpoints still
billing under the `eval-` name prefix.

## Project layout

```
dashboard/dashboard.html   single-file dashboard UI (no build step)
scripts/server.py          local dev server — dashboard, /api/* routes, eval runs
src/eval_runner.py         harness: task loading, backend routing, judge client
src/scoring.py             scorers: programmatic, reference_match, llm_judge
src/orchestrator.py        Serverless Endpoint create/wait/delete (nebius CLI)
src/storage.py             optional S3-compatible result upload
config/catalog.yaml        every model the harness knows how to run
config/models.yaml         current comparison selection (written by the dashboard)
config/prices.yaml         Token Factory per-token price snapshot (manual, dated)
data/*.jsonl                built-in example tasks, one scorer type each
```

## Adding a model

Add an entry to `config/catalog.yaml` with `id`, `basis` (`hosted` or
`self-hosted`), and either `preset`/`instance_type`/`rate_hr` (self-hosted) or
`tokenfactory_ok: true` (hosted). If hosted, add its price to
`config/prices.yaml` — prices are a manual snapshot (Token Factory has no
public pricing API), never guessed.

## License

MIT — see `LICENSE`.
