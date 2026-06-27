# Build Plan — Nebius Serverless Model-Eval Harness

> Handoff doc for Claude Code. Read top to bottom. Build in phase order.
> **Core** must ship; **Optional** only if time allows (2-week timeline).

---

## 1. What we're building

A reproducible pipeline that evaluates several small open LLMs on a **user-defined task**, reporting **quality + latency + real per-token serving cost** on Nebius Serverless, and renders a small **static dashboard** from the results.

- Models under test run as **Endpoints** (vLLM, OpenAI-compatible).
- The evaluation runs as a batch **Job** that queries the endpoints, scores answers, measures latency, derives cost, and writes `results.json` to **Object Storage**.
- An **orchestrator** creates the endpoints, launches the job, and tears everything down.
- The dashboard is a static HTML file that reads `results.json`. Nothing runs while you view it.

**Headline value:** public leaderboards rank models on generic benchmarks; they can't tell you how a model does on *your* task or what it costs to *self-host*. This measures both.

---

## 2. Hard constraints (do not violate)

- Must use Nebius Serverless **Jobs and/or Endpoints** as the core compute. (We use both.)
- **Public data only.** No personal/private data.
- **Reproducible** by a stranger from the README alone — one command path.
- **Public repo, OSS license (MIT or Apache-2.0), no secrets committed.**
- Secrets (Nebius creds, HF token) via env vars / CLI profile — never in code.

---

## 3. Stack & prerequisites

- **Python 3.11+**, `openai` (client), `pyyaml`, `boto3` (S3-compatible object storage), `requests`.
- **Nebius CLI** installed and authenticated (`nebius` profile configured). Verify with `nebius ai job list` early.
- **Endpoint image:** `vllm/vllm-openai:latest` (stock, no build needed).
- **Job image:** stock `python:3.11-slim` (or `pytorch`), with the runner script **mounted from object storage** — avoid building/pushing custom Docker images (the cookbook's pattern; far simpler + more reproducible).
- **Models:** non-gated, small — e.g. `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct` (swap Llama for another non-gated 3B if HF gating is a hassle).
- **Preset for core:** `gpu-l40s-a` / `1gpu-8vcpu-32gb` (cheapest GPU). Region with L40S: `eu-north1`.
- **Subnet:** get with `nebius vpc subnet list`.

> Note: this sandbox can't reach Nebius or Hugging Face, so real runs happen on the user's machine/account. The local-mock phase (Phase 1) lets you build and test logic offline for free.

---

## 4. Repo structure to create

```
nebius-model-eval/
├── README.md                  # setup, hardware, expected output, cost, runtime
├── LICENSE                    # MIT or Apache-2.0
├── .gitignore                 # .env, results/*.json, __pycache__, .venv
├── .env.example               # placeholders only (no real values)
├── requirements.txt
├── Makefile                   # local-mock / up / eval / down / dashboard / sweep / clean
├── config/
│   └── models.yaml            # models under test: id + preset
├── data/
│   └── assistant_commands.jsonl   # the public task set (start from our file)
├── src/
│   ├── orchestrator.py        # create / poll-readiness / delete endpoints (wraps nebius CLI)
│   ├── eval_runner.py         # JOB ENTRYPOINT: query endpoints, score, measure, write results
│   ├── scoring.py             # programmatic + reference + judge scorers
│   ├── cost.py                # cost derivation from uptime + preset rate
│   ├── storage.py             # object storage put/get (S3-compatible)
│   └── sweep.py               # OPTIONAL: best model across GPU presets
├── dashboard/
│   └── dashboard.html         # static, reads results.json (start from our file)
├── scripts/
│   ├── run_local_mock.py      # offline dry-run (start from our eval_pipeline.py)
│   └── cleanup.sh             # delete ALL leftover endpoints (safety net)
└── results/
    └── .gitkeep               # results.json lands here; commit one sample for the dashboard demo
```

**Starter files already built** (in the user's downloads — drop them in):
- `eval_pipeline.py`  → `scripts/run_local_mock.py`
- `task_eval.py`      → fold its loader + scorer into `src/eval_runner.py` + `src/scoring.py`
- `assistant_commands.jsonl` → `data/assistant_commands.jsonl`
- `dashboard.html`    → `dashboard/dashboard.html`

---

## 5. Architecture / run flow

```
Phase A  orchestrator: create Endpoint (model i) on L40S  ── per-second billing starts
Phase B  poll readiness (GET /v1/models until 200)
Phase C  eval Job: load task set → query endpoint → score → record latency/tokens/cost
Phase D  orchestrator: DELETE endpoint i                  ── billing stops
         (repeat A–D per model, SEQUENTIALLY, to stay on one GPU at a time)
Phase E  write/merge results.json → Object Storage
Phase F  (optional) hardware sweep: best model across L40S/H100/H200, fixed load
Phase G  dashboard reads results.json
```

Sequential (one endpoint up at a time) is the budget-safe default.

---

## 6. Build phases (ordered)

### Phase 0 — Scaffold  *(Core)*
- Create the repo tree, `LICENSE` (MIT), `.gitignore`, `.env.example`, `requirements.txt`, empty `Makefile`.
- `.env.example` keys: `NEBIUS_PROJECT_ID`, `NEBIUS_REGION`, `NEBIUS_SUBNET_ID`, `HF_TOKEN` (optional), `STORAGE_BUCKET`, `STORAGE_ENDPOINT`.
- **Done when:** repo initialized, `pip install -r requirements.txt` works in a fresh venv.

### Phase 1 — Local mock pipeline  *(Core — do this fully before touching Nebius)*
- Port `run_local_mock.py`. Confirm it produces `results.json` + renders `dashboard.html`.
- Refactor the eval logic into `src/eval_runner.py` (harness) + `src/scoring.py` (scorers) + `src/cost.py`, with a `--mock` flag that uses fake endpoints.
- Implement the scorers in `scoring.py` **following `SCORING_SPEC.md` exactly** (per-method inputs, internal record, and evaluation logic are all specified there). Core: `programmatic` + `reference_match` (lexical) — pure code, no model. Optional/stretch: `reference_match` (embedding) and `llm_judge` — both need a second model, so stub their callables for now.
- The user provides plain pieces (one `instruction`, a list of `inputs`, and aligned `answers`/`references` or one `rubric`) plus a chosen method; the tool **assembles the JSONL internally** — the user never writes JSONL. Task loader reads the assembled `data/*.jsonl` records.
- **Done when:** `make local-mock` runs end-to-end offline and opens a populated dashboard. This is your free, fast iteration loop.

### Phase 2 — One real endpoint, end-to-end  *(Core)*
- `src/orchestrator.py`:
  - `create_endpoint(model_id, preset)` → wraps `nebius ai endpoint create` (see §7), returns endpoint id.
  - `wait_ready(endpoint)` → poll `GET {url}/v1/models` with the bearer token until 200 or timeout.
  - `get_url(endpoint)`, `delete_endpoint(endpoint)`.
- Swap the mock model call in `eval_runner.py` for a real `openai.OpenAI(base_url, api_key)` call (gated behind removing `--mock`).
- Run against **one** model: create → wait → eval 12 items → delete.
- **Done when:** one real model produces real `results.json` with real latencies, and the endpoint is confirmed deleted (`nebius ai endpoint list` shows none left).

### Phase 3 — Multi-model + storage + real dashboard  *(Core — MINIMUM VIABLE SUBMISSION)*
- `config/models.yaml`: 3 models, each with `id` + `preset`.
- Orchestrator loops the models **sequentially** (up → eval → down per model).
- `src/storage.py`: write `results.json` to Object Storage (S3-compatible, see §7); README documents the bucket.
- Point `dashboard/dashboard.html` at the produced `results.json` (embed or fetch).
- `Makefile`: `up`, `eval`, `down`, `dashboard`, `clean`.
- Capture **proof of execution**: screenshots of job logs + endpoint list, save to `docs/`.
- **Done when:** `make eval` runs all 3 models, writes results to storage, tears down every endpoint, and the dashboard renders real numbers. *At this point you have a valid, complete submission.*

### Phase 4 — Hardware sweep  *(OPTIONAL — only if Phase 3 lands with days to spare)*
- `src/sweep.py`: take the top model from results, redeploy on 2–3 presets (`gpu-l40s-a`, `gpu-h100-sxm`, `gpu-h200-sxm`), push a **short fixed-load** request batch, record throughput + latency + $/1k tokens per preset, append to `results.json`.
- Keep each high-end run to a few minutes. **Delete immediately after each.**
- Add a hardware-cost panel to the dashboard.
- **Done when:** dashboard shows the winning model's cost/throughput across GPUs, and no endpoints remain.

### Phase 5 — Polish & submit  *(Core)*
- README: problem, architecture, **exact run commands**, hardware config, expected output, **approximate runtime & cost**, teardown warning.
- Blog post (≥600 words): problem → architecture → implementation → results, tagged **#NebiusServerlessChallenge**, links the repo. (Optional 3–10 min video.)
- Final repo check: no secrets, license present, public, README reproducible.
- **Done when:** repo + blog are public and submitted via the Nebius Academy Submit tab before **June 30, 23:59 UTC**.

---

## 7. Key implementation details

**Create an endpoint (vLLM, OpenAI-compatible):**
```bash
AUTH_TOKEN=$(openssl rand -hex 32)
SUBNET=$(nebius vpc subnet list --format json | jq -r '.items[0].metadata.id')
nebius ai endpoint create \
  --name eval-<model-slug> \
  --image vllm/vllm-openai:latest \
  --container-command "python3 -m vllm.entrypoints.openai.api_server" \
  --args "--model <HF_MODEL_ID> --host 0.0.0.0 --port 8000" \
  --platform gpu-l40s-a --preset 1gpu-8vcpu-32gb \
  --public --container-port 8000 \
  --shm-size 16Gi --disk-size 450Gi \
  --auth token --token "$AUTH_TOKEN" \
  --subnet-id "$SUBNET"
```

**Readiness poll** (vLLM downloads weights on boot — don't eval until ready):
```python
# GET http://{ip}/v1/models  with  Authorization: Bearer $AUTH_TOKEN  → wait for 200
```

**Query a model** (this is the only thing that differs from the mock):
```python
from openai import OpenAI
client = OpenAI(base_url=f"http://{ip}/v1", api_key=AUTH_TOKEN)
r = client.chat.completions.create(model=MODEL_ID, messages=msgs, temperature=0)
text = r.choices[0].message.content
usage = r.usage  # prompt_tokens, completion_tokens
```

**Cost derivation** (`src/cost.py`): record endpoint `t_up` and `t_down`; per-second billing means
`serving_cost = (t_down - t_up) * (preset_rate_per_hour / 3600)`, and
`cost_per_1k_tokens = serving_cost / total_completion_tokens * 1000`.
Put the per-preset hourly rate in `config/models.yaml`; **verify exact rates on the Nebius pricing page** before reporting.

**Object storage** (S3-compatible): endpoint `https://storage.<region>.nebius.cloud`, use `boto3` with the project's access keys (from env), `put_object` the `results.json`.

**Teardown safety** (`scripts/cleanup.sh`): list all endpoints and delete any with the `eval-` prefix. Call it in a `Makefile` `clean` target and after every run. A forgotten endpoint silently burns credits — H100/H200 especially.

**Run the eval as a Nebius Job** (not just locally): mount the runner from storage and run on a stock image —
```bash
nebius ai job create --name eval-run \
  --image python:3.11-slim \
  --container-command bash \
  --args "-c 'pip install -q openai boto3 pyyaml && python /mnt/data/eval_runner.py'" \
  --platform cpu-d3 --preset 4vcpu-16gb \
  --volume "${BUCKET_ID}:/mnt/data:rw" --timeout 30m
```
(The eval job is CPU-only — it just makes HTTP calls; the GPUs are the endpoints.)

---

## 8. Cost & budget guardrails (~$100)

- Core (3 small models on L40S, sequential, ~12–150 task items) is **cheap** — well under budget.
- Always `temperature=0` for reproducibility.
- One endpoint up at a time. Delete immediately after each model.
- Optional sweep: H100/H200 runs are minutes, not hours; delete on completion.
- Run `make clean` if anything errors mid-run. Check `nebius ai endpoint list` is empty before walking away.

---

## 9. Definition of done (submission)

- [ ] Public repo: code, `Dockerfile`-or-stock-image reference, README (setup/hardware/output/runtime/cost), OSS license, no secrets.
- [ ] `make eval` reproduces results from scratch on a fresh Nebius account.
- [ ] `results.json` + dashboard render real numbers; proof-of-execution screenshots in `docs/`.
- [ ] Blog post ≥600 words, tagged #NebiusServerlessChallenge, links repo.
- [ ] (Optional) 3–10 min video.
- [ ] Submitted before June 30, 23:59 UTC.

---

## 10. First prompt to give Claude Code

> "Read BUILD_PLAN.md. Do Phase 0 and Phase 1 only: scaffold the repo exactly as in §4, then port the local-mock pipeline so `make local-mock` runs fully offline and produces results.json + a populated dashboard.html. Use the four starter files I'll place in the repo root. Stop after Phase 1 so I can review before we touch Nebius."

Then proceed phase by phase, reviewing at each "Done when."

---

## 11. To verify on the real account (open items)

- Exact per-hour rates for `gpu-l40s-a`, `gpu-h100-sxm`, `gpu-h200-sxm` (pricing page).
- L40S quota/availability on the new account & correct region.
- Whether chosen models are gated on HF (need `HF_TOKEN`) — prefer non-gated to avoid friction.
- Exact `nebius ai endpoint create` flag names for your CLI version (the API evolves — check `nebius ai endpoint create --help`).
- Object-storage access-key setup for `boto3`.
