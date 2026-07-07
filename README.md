# Nebius Serverless Eval Dashboard

Compare LLMs side by side on your own tasks, running against real Nebius
infrastructure — no mocks, no simulated scores. A local dashboard lets you pick
a scoring method, enter data (or use a built-in example), choose up to 3
models, and get a leaderboard with accuracy, latency, and cost.

## Backends

Each model is routed automatically by its catalog entry — a single comparison
can mix both:

- **Token Factory** (`basis: hosted`) — Nebius's hosted inference API. No GPU
  provisioning, billed per token, starts immediately.
- **Serverless Endpoints** (`basis: self-hosted`) — deploys a GPU VM on demand
  (vLLM), runs the eval, tears it down. Billed per GPU-hour from create to
  delete. Takes 5–20 minutes to provision.

## Scoring methods

- **Programmatic** — exact string match, numeric extraction, or F1 over
  JSON fields. No API calls beyond the model itself.
- **Reference match** — lexical (token F1 word overlap) or embedding
  (cosine similarity) comparison against a reference answer.
- **LLM judge** — a judge model (default `zai-org/GLM-5.2`, configurable via
  `JUDGE_MODEL`) grades each answer 1–N against a rubric you write. Judge
  token cost is tracked separately from the models being compared.

## Setup

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
