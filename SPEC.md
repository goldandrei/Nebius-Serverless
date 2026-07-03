# Technical Specification — Nebius Model Eval Harness

Reference for implementation. Describes contracts, schemas, data flow, and
invariants. Read alongside `PLAN_V2.md` (the roadmap) and `SCORING_SPEC.md`
(the scorer implementations).

---

## 1. System overview

```
User (browser)
  │
  ▼
scripts/server.py  (local HTTP server, port 7860)
  │── GET /api/catalog        → config/catalog.yaml
  │── GET /api/tasks          → data/*.jsonl (metadata only)
  │── GET /api/routing        → routing plan for selected models (dry-run)
  │── POST /api/build         → assembles data/custom.jsonl from builder inputs
  │── POST /api/run           → runs eval, returns results JSON
  │── GET /api/progress       → poll during run
  │── GET /api/prices         → config/prices.yaml snapshot
  │
  ▼
src/eval_runner.run()
  │── splits models by basis (catalog.basis field)
  │── basis: hosted      → _run_tokenfactory()  (Token Factory REST API)
  │── basis: self-hosted → _run_endpoint()      (Nebius Serverless AI Endpoint, sequential)
  │── merges leaderboards, returns results dict
  │
  ├── src/orchestrator.py     (endpoint lifecycle — CLI via WSL on Windows)
  ├── src/scoring.py          (three scorers)
  ├── src/cost.py             (cost derivation helpers)
  └── src/storage.py          (S3-compatible upload, optional)
```

---

## 2. Config schemas

### `config/catalog.yaml`

```yaml
models:
  - id: <HuggingFace model ID>          # required; used as model name in API calls
    tokenfactory_ok: true|false         # is this model available on Token Factory?
    basis: self-hosted|hosted           # routing key (derived from tokenfactory_ok; explicit)
    preset: gpu-l40s-a|gpu-h100-sxm    # Nebius GPU platform
    instance_type: 1gpu-8vcpu-32gb|...  # Nebius instance preset
    rate_hr: 1.55                       # GPU $/hr — verify on nebius.com/pricing
    gated: true                         # optional; requires HF_TOKEN
    mock:                               # offline test parameters
      p_bad_json: 0.0                   # prob of malformed JSON reply (programmatic tasks)
      p_wrong_intent: 0.0              # prob of wrong intent in JSON
      p_slot_err: 0.0                   # prob of slot error in JSON
      p_ref_err: 0.0                    # prob of wrong factual answer (reference_match)
      judge_score_mean: 0.9             # mean judge score (llm_judge)
      judge_score_sd: 0.05
      lat_mean: 1.0                     # simulated latency seconds
      lat_sd: 0.1
```

Routing rule (enforced in `eval_runner.run()`):
- `basis: hosted`      → `_run_tokenfactory()` — no GPU infra, billed per token
- `basis: self-hosted` → `_run_endpoint()`     — creates/tears down GPU VM, billed per hour

### `config/models.yaml`

```yaml
compare:
  - <HF model ID>    # must exist in catalog.yaml
  - ...
```

Written by the dashboard on every run (POST /api/run). Max 3 models.

### `config/prices.yaml`

```yaml
prices_as_of: "YYYY-MM-DD"
tokenfactory:
  <model_id>:
    price_in_per_1m: 0.00   # $/1M input tokens
    price_out_per_1m: 0.00  # $/1M output tokens
```

Loaded once at server startup. Not fetched per-run. Update manually —
Token Factory pricing is not a public REST API.

---

## 3. Task JSONL schemas

Tasks are stored in `data/*.jsonl`, one JSON object per line.
Assembled by `POST /api/build`; users never write JSONL directly.

### Programmatic scorer

```json
{
  "id": 1,
  "instruction": "Classify sentiment. Reply with one word.",
  "input": "The food was cold.",
  "gold": "negative",
  "scorer": "programmatic",
  "compare": "exact"
}
```

`compare` values: `exact` (default), `numeric`, `json_fields`.

### Reference-match scorer

```json
{
  "id": 1,
  "instruction": "Answer in one sentence.",
  "input": "Why is the sky blue?",
  "reference": "Rayleigh scattering — blue light scatters more than red.",
  "scorer": "reference_match",
  "metric": "lexical",
  "threshold": 0.6
}
```

`metric` values: `lexical` (token F1, no model), `embedding` (cosine similarity,
requires NEBIUS_API_KEY for the embedding model).

### LLM-judge scorer

```json
{
  "id": 1,
  "instruction": "Write a one-line apology.",
  "input": "Order #123 arrived 3 days late.",
  "rubric": "Score 1-5: apologizes (2), warm tone (2), concise (1).",
  "scorer": "llm_judge",
  "scale": 5
}
```

---

## 4. Backend routing

### Rule

```python
# src/eval_runner.run()
ep_models = [m for m in models if m.get("basis", "self-hosted") == "self-hosted"]
tf_models  = [m for m in models if m.get("basis", "self-hosted") == "hosted"]
```

A single run can use both backends. Results are merged into one leaderboard
sorted by accuracy descending.

### Dry-run endpoint

`GET /api/routing?models=ModelA,ModelB` — returns the routing plan without
running anything:

```json
{
  "endpoint":     ["Qwen/Qwen2.5-0.5B-Instruct"],
  "tokenfactory": ["Qwen/Qwen3-32B"],
  "unknown":      []
}
```

---

## 5. Endpoint lifecycle (`src/orchestrator.py`)

Wraps the `nebius` CLI. On Windows, invoked via WSL:
```python
wsl bash -c 'export PATH="$HOME/.nebius/bin:$PATH"; nebius <args>'
```

### State sequence

```
create --async
  └─► PROVISIONING  (GPU being allocated)
       └─► STARTING  (vLLM container starting, weights downloading)
            └─► RUNNING  (container up — weights may still be loading)
                 └─► poll GET {url}/v1/models with bearer token until 200
                      └─► READY FOR INFERENCE
```

`RUNNING ≠ weights loaded`. The second poll is required.

### Key functions

```python
create_endpoint(model_id, platform, preset, auth_token=None) -> str
    # Calls nebius ai endpoint create --async
    # Returns endpoint_id (aiendpoint-e00...)
    # Billing starts here

wait_ready(endpoint_id, auth_token, timeout_s=900) -> str
    # Phase 1: poll nebius ai endpoint get until state == "RUNNING" and public URL present
    # Phase 2: poll GET {url}/v1/models with bearer token until 200
    # Returns HTTPS base URL (no /v1 suffix)

get_endpoint(endpoint_id) -> dict
    # nebius ai endpoint get --id <id> --format json

list_endpoints() -> list
    # nebius ai endpoint list --parent-id PROJECT_ID --format json → items[]

delete_endpoint(endpoint_id) -> None
    # nebius ai endpoint delete --id <id>
    # Billing stops here
```

### Auth token

Generated as `secrets.token_hex(32)` before `create_endpoint()`. Passed as
`--auth token --token <hex>` at create time. Used as `Authorization: Bearer <token>`
for all vLLM inference calls on that endpoint. The caller holds the token — it is
NOT retrievable from the endpoint spec after creation.

### CLI output quirk

`nebius ai endpoint create --async` always outputs text to stdout regardless of
`--format json`:
```
Token: <token>
Endpoint ID: aiendpoint-e00...
```
`_run_create_async()` parses "Endpoint ID:" directly from the text output.

`public_endpoints` in the CLI JSON response is a list of strings:
```json
"public_endpoints": ["https://port8000-<hash>.tunnel.applications.eu-north1.nebius.cloud"]
```
`_extract_url()` handles both strings and `{"url": "..."}` dicts.

---

## 6. Cost model

### Self-hosted endpoints

```python
rate_hr    = catalog[model]["rate_hr"]           # e.g. 1.55 for L40S
deploy_cost = (t_ready - t_created) / 3600 * rate_hr   # startup tax
eval_cost   = (t_eval_done - t_ready) / 3600 * rate_hr  # steady-state work
total_cost  = (t_eval_done - t_created) / 3600 * rate_hr
ss_per_1k   = eval_cost / total_output_tokens * 1000    # steady-state $/1K tok
```

Typical L40S numbers (Qwen2.5-0.5B, 3 items):
- Deploy: ~10 min → $0.26
- Eval: ~2 s → $0.0001
- Total: ~$0.27 (dominated by startup tax)
- Steady-state $/1K tok: ~$0.01 (ignoring startup)

The startup tax is reported separately so it's not mistaken for per-token cost.
For steady-state cost, keep the endpoint up for many items or across multiple
models before deleting.

### Hosted (Token Factory)

```python
p_in   = prices[model]["price_in_per_1m"]   # from prices.yaml
p_out  = prices[model]["price_out_per_1m"]
run_cost = (total_in * p_in + total_out * p_out) / 1_000_000
c1k      = run_cost / (total_in + total_out) * 1000
```

No deploy cost. Billed per token, regardless of time. Prices loaded once at
server startup from `config/prices.yaml`.

---

## 7. Results JSON schema

```json
{
  "meta": {
    "mode": "auto",
    "task": "factual_qa",
    "task_name": "Factual Q&A",
    "scorer": "reference_match",
    "n_models": 2,
    "n_tasks": 3,
    "benchmark": "Factual Q&A",
    "routing": {
      "endpoint":     ["Qwen/Qwen2.5-0.5B-Instruct"],
      "tokenfactory": ["Qwen/Qwen3-32B"]
    }
  },
  "leaderboard": [
    {
      "model":                  "Qwen/Qwen3-32B",
      "preset":                 "gpu-h100-sxm / 1gpu-20vcpu-200gb",
      "cost_basis":             "hosted",
      "accuracy":               0.9167,
      "correct":                11,
      "n":                      12,
      "mean_latency_s":         1.23,
      "p95_latency_s":          2.10,
      "total_in_tokens":        480,
      "total_out_tokens":       240,
      "cost_per_1k_tokens_usd": 0.000042,
      "total_run_cost_usd":     0.000030
    },
    {
      "model":                  "Qwen/Qwen2.5-0.5B-Instruct",
      "preset":                 "gpu-l40s-a / 1gpu-8vcpu-32gb",
      "cost_basis":             "self-hosted",
      "accuracy":               0.6667,
      "correct":                2,
      "n":                      3,
      "mean_latency_s":         0.49,
      "p95_latency_s":          1.13,
      "cost_per_1k_tokens_usd": 0.01142,
      "total_run_cost_usd":     0.2685,
      "t_created":              1751537052.0,
      "t_ready":                1751537673.0,
      "t_eval_done":            1751537675.0,
      "deploy_cost_usd":        0.2677,
      "eval_cost_usd":          0.000082
    }
  ],
  "samples": [
    {
      "q": "Why does the sky appear blue?",
      "expected": "Rayleigh scattering...",
      "scorer": "reference_match",
      "answers": {
        "Qwen/Qwen2.5-0.5B-Instruct": {
          "text": "...", "correct": true, "score": 0.72,
          "latency_s": 1.13, "out_tokens": 16
        },
        "Qwen/Qwen3-32B": {
          "text": "...", "correct": true, "score": 0.91,
          "latency_s": 0.82, "in_tokens": 40, "out_tokens": 24
        }
      }
    }
  ]
}
```

`cost_basis` per row drives the dashboard basis-tag rendering.
`deploy_cost_usd` / `eval_cost_usd` are only present for self-hosted rows.

---

## 8. API endpoints (`scripts/server.py`)

| Method | Path | Input | Output |
|---|---|---|---|
| GET | `/` | — | `dashboard/dashboard.html` |
| GET | `/api/catalog` | — | array of catalog model objects |
| GET | `/api/selection` | — | array of selected model IDs |
| GET | `/api/tasks` | — | array of task metadata objects |
| GET | `/api/tasks/:id` | — | array of JSONL records for that task |
| GET | `/api/routing?models=A,B` | — | `{endpoint:[...], tokenfactory:[...], unknown:[...]}` |
| GET | `/api/progress` | — | `{running, done, model, model_idx, n_models, item_idx, n_items}` |
| GET | `/api/prices` | — | `{prices:{...}, prices_as_of, source}` |
| GET | `/results.json` | — | last results.json from disk |
| POST | `/api/run` | `{models:[...], task:"..."}` | results JSON |
| POST | `/api/build` | builder payload | `{n, task}` |
| POST | `/api/upload?task=...` | JSONL body | `{n, task}` |

`POST /api/run` body — no `backend` field; routing is auto-derived per model:
```json
{"models": ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen3-32B"], "task": "factual_qa"}
```

---

## 9. Environment variables

| Variable | Required | Description |
|---|---|---|
| `NEBIUS_PROJECT_ID` | endpoint runs | Nebius project ID (`project-e00...`) |
| `NEBIUS_SUBNET_ID` | endpoint runs | VPC subnet for the endpoint (`vpcsubnet-e00...`) |
| `NEBIUS_REGION` | optional | Default `eu-north1` |
| `NEBIUS_API_KEY` | TF + embedding scoring | API key for Token Factory + embedding model |
| `NEBIUS_BASE_URL` | TF | Token Factory base URL (`https://api.tokenfactory.nebius.com/v1/`) |
| `HF_TOKEN` | gated models | HuggingFace token; passed as `--env HF_TOKEN=...` to endpoint container |
| `EMBEDDING_MODEL` | embedding scoring | Default `Qwen/Qwen3-Embedding-8B` |
| `STORAGE_BUCKET` | optional | S3 bucket name for results upload |
| `STORAGE_ENDPOINT` | optional | S3 endpoint URL |

All loaded from `.env` at startup. `.env` is gitignored — never committed.

---

## 10. Budget guardrails

- **One endpoint at a time.** `_run_endpoint()` is sequential: create → eval → delete per model.
- **Delete in `finally` block.** The endpoint delete always runs even if eval throws.
- **Timeout.** `wait_ready()` has a 15-minute ceiling; raises `TimeoutError` if exceeded. Endpoint is then left in a non-RUNNING state — run `orchestrator.delete_endpoint()` manually.
- **`list_endpoints()` before run.** `scripts/run_phase1.py` checks for existing endpoints before creating a new one.
- **Cost check before spending.** Confirm `NEBIUS_API_KEY` is set before any TF call.
- **L40S startup cost.** ~10 min × $1.55/hr = ~$0.26 per model regardless of eval size. For 3 models sequential: ~$0.78 in startup alone. Factor in when deciding how many models to benchmark.
- **H100/H200 warning.** H100 is $3.80/hr, H200 $5.50/hr. A 10-min startup costs $0.63 / $0.92. Only use for the hardware sweep phase; delete immediately after each run.
- **Emergency cleanup.** `orchestrator.list_endpoints()` lists all live endpoints. Delete any with `eval-` prefix if a run was interrupted.
