# Project Plan v2 — "Which model, where, at what cost?"

## The architecture (decided)

Two ways a model runs, shown clearly to the user:

| Basis | Backend | Which models | Pricing unit | Who hosts |
|---|---|---|---|---|
| **self-hosted** | `endpoint` | small (0.5B–7B), *must* use this | $/hour (GPU) | you, on Nebius Serverless |
| **hosted** | `tokenfactory` | big (32B+), *chosen* for cost | $/token | Nebius (hosted API) |

Small models can ONLY run via endpoint. Big models COULD self-host but are
routed to Token Factory because self-hosting a 70B burns the budget.
Optional final form: an **AI Job** runs the evaluator and calls both.

## Work items (in build order)

### 1. Model list — show basis + correct price per model
- Each catalog model tagged `basis: self-hosted | hosted`.
- `self-hosted` models: carry `gpu_rate_per_hr` (L40S 1.55, H100 3.8 — already known).
- `hosted` models: carry `price_in_per_1m` + `price_out_per_1m` (entered by hand from
  the Token Factory console — NOT scrapable; leave TODO nulls until filled).
- UI: show a **basis badge** on each model card, and the *relevant* price only
  (per-hr for self-hosted, per-token for hosted). Stop showing $/hr on hosted models.

### 2. Cost columns — two bases, one comparable number
- Shared column **`$ / 1K tokens`** — both bases normalize into it. This is what
  Score-vs-Cost plots.
- Per-row **`basis`** tag (hosted / self-hosted) so units aren't confused.
- Endpoint (self-hosted) also gets **deploy cost** and **eval cost** split (see #3).

### 3. Endpoint results — deploy cost vs eval cost (self-hosted only)
Record three timestamps per endpoint: `t_created`, `t_ready`, `t_eval_done`.
- **load/deploy time** = t_ready − t_created
- **deploy cost** = (t_ready − t_created) × $/hr   ← the startup tax
- **eval cost** = (t_eval_done − t_ready) × $/hr    ← steady-state work
- **total cost** = (t_eval_done − t_created) × $/hr  ← what you actually pay (honest headline)
- **$/1K tok (steady-state)** = eval cost ÷ tokens × 1000
(Token Factory has no deploy cost — you don't deploy it. Only endpoints show this.)

### 4. THE MILESTONE — run the endpoint backend for real (needs CLI, done later)
- Install `nebius` CLI + project + subnet id (deferred until plan is locked).
- Reconcile orchestrator flags against the real CLI (`nebius ai endpoint create --help`).
- One small model (`Qwen/Qwen2.5-0.5B-Instruct`) on L40S: create → ready → eval → delete.
- Confirm the 3 timestamps + real costs populate. THIS is what makes the project valid.

### 5. Scale + optional big-model axis
- Loop 3 small models on endpoints (the core comparison).
- Optionally add 1–2 big models via Token Factory for the small-vs-big story.
- Optionally wrap the evaluator in a Nebius **Job** (uses both services).

### 6. Submission essentials (before June 30)
- README: setup, hardware, expected output, runtime, cost, teardown warning.
- Proof-of-execution screenshots (endpoint logs, job logs).
- Blog post ≥600 words, tagged #NebiusServerlessChallenge, links repo.

## Still missing / decisions
- Endpoint path: coded, NEVER run. Highest risk. → item 4.
- Token Factory per-token prices: must be hand-entered (not scrapable).
- README: not started.
- CLI + project + subnet: not set up (deferred, but blocks item 4).
