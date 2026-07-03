# Project Plan v2 — "Which model, where, at what cost?"

## Status snapshot (2026-07-03)

| Phase | Status | Notes |
|---|---|---|
| 0 · Scaffold | ✅ Done | Repo, venv, .env, gitignore |
| 1 · Local mock pipeline | ✅ Done | Dashboard, scorers, mock backend |
| 1.5 · Real endpoint validation | ✅ Done | Confirmed Serverless AI Endpoint path; real run on record |
| 1.6 · Per-model auto-routing | ✅ Done | Basis tags, routing preview, merged leaderboard |
| **2 · Real mixed comparison** | **🔜 Next** | Endpoint + Token Factory in one run |
| 3 · Object storage | 🔜 | Upload results.json to Nebius S3 |
| 4 · Hardware sweep | Optional | Top model across L40S/H100/H200 |
| 5 · Submission | 🔜 | README, blog post, proof-of-execution |

---

## Architecture (as-built)

Two backends, auto-selected per model from `catalog.yaml`:

| Basis | Backend | Which models | Billing unit | Who hosts |
|---|---|---|---|---|
| `self-hosted` | `endpoint` | small (0.5B–14B) — no TF image | $/hr (GPU) | You, on Nebius Serverless AI Endpoint |
| `hosted` | `tokenfactory` | large (32B+) — available on TF | $/M tokens | Nebius (Token Factory hosted API) |

Small models have no Token Factory image — endpoint is the only option.
Large models could be self-hosted but the startup tax ($0.27 for ~10 min) dominates at low eval volume.

```
Per endpoint run (self-hosted):
  create → PROVISIONING → STARTING → RUNNING → /v1/models 200 → eval → DELETE
  t_created          t_ready                                          t_eval_done
  deploy_cost = (t_ready - t_created) / 3600 * rate_hr
  eval_cost   = (t_eval_done - t_ready) / 3600 * rate_hr
  total_cost  = (t_eval_done - t_created) / 3600 * rate_hr

Per TF model run (hosted):
  POST /v1/chat/completions → no infrastructure; billed per token
```

---

## What's done

### Phase 0 — Scaffold ✅
Repo tree, LICENSE (MIT), `.gitignore`, `.env.example`, `requirements.txt`.
`uv run --with pyyaml,openai,requests python scripts/server.py` starts the dashboard.

### Phase 1 — Local pipeline + dashboard ✅
- `src/scoring.py` — three scorers: programmatic, reference_match (lexical + embedding), llm_judge
- `src/eval_runner.py` — task loading, Token Factory eval loop, leaderboard builder
- `data/*.jsonl` — assistant_commands, factual_qa, instruction_following datasets
- `dashboard/dashboard.html` — model picker, task builder, live leaderboard, score/cost chart, samples table
- `scripts/server.py` — local dev server (catalog, selection, tasks, run, build, upload, progress)
- `config/prices.yaml` + `/api/prices` — per-run token cost from a hand-updated file (TF pricing not public API)
- `config/catalog.yaml` — ~30 models with preset, rate_hr, mock profiles
- Token Factory path confirmed working against live api.tokenfactory.nebius.com

### Phase 1.5 — Real Nebius Serverless Endpoint (validation) ✅
**Commits:** 62f3647, 62aa1e2

Key discoveries during this phase:
- The challenge requires **Nebius Serverless AI Endpoints** (`nebius ai endpoint create`), NOT Token Factory Dedicated Endpoints — these are two separate products. Phase 1's first attempt used the wrong product.
- `nebius` CLI is Linux/macOS only; on Windows, invoked via WSL (`wsl bash -c "..."`).
- `nebius ai endpoint create --async` outputs text to stdout regardless of `--format json`; must parse "Endpoint ID:" line directly (`_run_create_async()`).
- `public_endpoints` in the CLI JSON response is a list of strings, not dicts; `_extract_url()` handles both.
- State sequence: `PROVISIONING → STARTING → RUNNING`; then separately poll `GET /v1/models` with bearer token until 200 (RUNNING ≠ weights loaded).

**Real run results (commit 62aa1e2):**
```
endpoint_id  = aiendpoint-e00pwhsk7m4hfxv6bv
model        = Qwen/Qwen2.5-0.5B-Instruct (vllm/vllm-openai:latest)
platform     = gpu-l40s-a / 1gpu-8vcpu-32gb
t_created    = 09:04:12 UTC
t_ready      = 09:14:33 UTC  (+10.4 min — PROVISIONING → STARTING → RUNNING + /v1/models 200)
t_eval_done  = 09:14:35 UTC  (+2s from ready)
deploy_cost  = $0.2677  (10.4 min × $1.55/hr)
eval_cost    = $0.000082
total_cost   = $0.2685
$/1K tok ss  = $0.01142
accuracy     = 2/3 (67%)
teardown     = clean (0 endpoints remaining)
```

### Phase 1.6 — Per-model auto-routing ✅
**Commit:** 067c64d

- `config/catalog.yaml`: every model now has `basis: self-hosted | hosted`
- `src/eval_runner.py`: `run()` no longer takes a `backend` param; splits selected models by `basis`, runs each group through its backend, merges leaderboards; `meta.routing` records which model → which backend
- `scripts/server.py`: `/api/run` no longer takes `backend`; new `GET /api/routing?models=a,b` returns routing plan without running anything
- `dashboard/dashboard.html`: backend dropdown removed; model cards show `[endpoint GPU]` / `[hosted API]` tags; run bar shows live routing preview as models are selected; post-run status shows `N endpoint + N hosted`

**Dry-run verified for mixed selection:**
```
Qwen/Qwen2.5-0.5B-Instruct  →  endpoint   (gpu-l40s-a / 1gpu-8vcpu-32gb, $1.55/hr)
Qwen/Qwen3-32B               →  tokenfactory  (hosted API)
```

---

## What's next

### Phase 2 — Real mixed comparison 🔜

**Goal:** run `Qwen/Qwen2.5-0.5B-Instruct` (endpoint) + `Qwen/Qwen3-32B` (Token Factory) in one comparison and produce a merged leaderboard showing both bases side by side.

**Budget constraint:** each endpoint deploy costs ~$0.27 for the 10-min startup tax. Keep to **1–2 small endpoint models max** per run.

**Tasks:**
1. Confirm Token Factory API key is set and Qwen3-32B is live on TF (`GET /v1/models`)
2. Run `python scripts/run_phase1.py` → verify routing console output shows the split
3. Inspect merged leaderboard JSON: both rows present, `cost_basis` correct per row, `meta.routing` populated
4. Dashboard: load results.json, confirm both rows render with basis tags and correct cost columns
5. Check that hosted models show `$/M tok` cost (from prices.yaml) and endpoint models show `$/1K tok (ss)` + deploy split

**Stretch for Phase 2:** add `Qwen/Qwen2.5-1.5B-Instruct` or `Qwen/Qwen2.5-3B-Instruct` as a second endpoint model for a small-model scaling curve.

### Phase 3 — Object storage 🔜

Upload `results.json` to Nebius S3 after each run so the dashboard can load from a public URL (currently it reads from `results/results.json` on disk).

- `src/storage.py` already exists; needs `STORAGE_BUCKET` + `STORAGE_ENDPOINT` in `.env`
- Endpoint: `https://storage.eu-north1.nebius.cloud`; auth via project access keys

### Phase 4 — Hardware sweep (optional)

Take the best small model from Phase 2 results; redeploy on `gpu-l40s-a` vs `gpu-h100-sxm`; record throughput + latency + $/1K tok per preset. Each run is a few minutes. Add hardware panel to dashboard.

### Phase 5 — Submission

- README: problem, architecture, exact run commands, hardware, expected output, approximate runtime + cost, teardown warning
- Blog post ≥600 words, tagged `#NebiusServerlessChallenge`, links repo
- Proof-of-execution screenshots in `docs/` (endpoint list, real leaderboard)
- Final checks: no secrets, license present, public repo

---

## Open items

- Token Factory per-token prices for Qwen3-32B: must be manually entered in `config/prices.yaml` (TF pricing is not a public API — SPA with internal IAM-authenticated billing)
- Confirm Qwen3-32B is available on current TF account: `GET /v1/models` with NEBIUS_API_KEY
- Object storage access keys: not yet set up in `.env`
- README: not yet started
- Blog post: not started
