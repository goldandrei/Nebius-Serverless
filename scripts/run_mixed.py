#!/usr/bin/env python3
"""
Real mixed comparison: Qwen/Qwen2.5-0.5B-Instruct (endpoint) + Qwen/Qwen3-32B (Token Factory).
Task: factual_qa, first 5 items.

Runs through the actual eval_runner backend functions to validate the routing.
Reports merged leaderboard with basis tags, deploy/eval/total cost, TF latency,
routing metadata, and teardown confirmation.
"""
import datetime
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# ── env checks ────────────────────────────────────────────────────────────────
REQUIRED = {
    "endpoint": ["NEBIUS_PROJECT_ID", "NEBIUS_SUBNET_ID"],
    "both":     ["NEBIUS_API_KEY", "NEBIUS_BASE_URL"],
}
missing = [v for v in REQUIRED["endpoint"] + REQUIRED["both"] if not os.environ.get(v)]
if missing:
    print(f"ERROR: missing env vars: {', '.join(missing)}")
    sys.exit(1)

import yaml
from src import eval_runner, orchestrator, scoring
from src.eval_runner import (
    load_catalog, _make_embed, _run_tokenfactory, _run_endpoint,
    _load_prices_from_file, task_expected,
)

N_TASKS = 5
EP_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TF_MODEL = "Qwen/Qwen3-32B"
TASK_FILE = "factual_qa"


def ts(t: float) -> str:
    return datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).strftime("%H:%M:%S UTC")


def banner(s: str):
    print(f"\n{'─'*60}")
    print(s)
    print('─'*60)


# ── load catalog + tasks ──────────────────────────────────────────────────────
catalog = load_catalog()
ep_model = catalog.get(EP_MODEL)
tf_model = catalog.get(TF_MODEL)
if not ep_model:
    print(f"ERROR: {EP_MODEL} not in catalog"); sys.exit(1)
if not tf_model:
    print(f"ERROR: {TF_MODEL} not in catalog"); sys.exit(1)

task_path = ROOT / "data" / f"{TASK_FILE}.jsonl"
all_tasks = [json.loads(l) for l in task_path.read_text(encoding="utf-8").splitlines() if l.strip()]
tasks = all_tasks[:N_TASKS]

prices = _load_prices_from_file()

# ── routing plan ──────────────────────────────────────────────────────────────
banner("ROUTING PLAN")
routing = {
    "endpoint":     [EP_MODEL],
    "tokenfactory": [TF_MODEL],
}
print(f"  Task        : {TASK_FILE} ({N_TASKS}/{len(all_tasks)} items)")
print(f"  endpoint    : {EP_MODEL}")
print(f"    basis     : {ep_model.get('basis')}")
print(f"    preset    : {ep_model['preset']} / {ep_model['instance_type']}")
print(f"    rate      : ${ep_model['rate_hr']}/hr")
print(f"  tokenfactory: {TF_MODEL}")
print(f"    basis     : {tf_model.get('basis')}")

# ── pre-run: confirm zero endpoints ──────────────────────────────────────────
banner("PRE-RUN STATE")
existing = orchestrator.list_endpoints()
active   = [e for e in existing if e.get("status", {}).get("state") not in ("DELETING", "DELETED")]
print(f"  Active endpoints: {len(active)}")
if active:
    for e in active:
        print(f"    {e['metadata']['id']} [{e.get('status',{}).get('state')}]")
    print("  WARNING: unexpected existing endpoints — proceeding anyway")

# ── run Token Factory (Qwen3-32B) ─────────────────────────────────────────────
banner("TOKEN FACTORY RUN — Qwen/Qwen3-32B")
print(f"  Calling api.tokenfactory.nebius.com ({N_TASKS} items)...")

t_tf_start = time.time()
tf_result  = _run_tokenfactory(
    tasks, [tf_model],
    task_label="Factual Q&A", first_scorer="reference_match",
    task_file=TASK_FILE, progress_cb=None, prices=prices,
)
t_tf_done  = time.time()
tf_elapsed = t_tf_done - t_tf_start

tf_row = tf_result["leaderboard"][0]
tf_lat = tf_row["mean_latency_s"]
tf_samples = tf_result["samples"]

print(f"  Done in {tf_elapsed:.1f}s")
print(f"  accuracy    : {tf_row['correct']}/{tf_row['n']}  ({tf_row['accuracy']*100:.1f}%)")
print(f"  mean lat    : {tf_lat:.2f}s")
print(f"  in tokens   : {tf_row.get('total_in_tokens', '?')}")
print(f"  out tokens  : {tf_row.get('total_out_tokens', '?')}")
if tf_row.get("total_run_cost_usd") is not None:
    print(f"  run cost    : ${tf_row['total_run_cost_usd']:.6f}")
    print(f"  $/1K tok    : ${tf_row['cost_per_1k_tokens_usd']:.6f}")
else:
    print("  run cost    : — (add Qwen3-32B prices to config/prices.yaml)")

# ── run Endpoint (Qwen2.5-0.5B) ───────────────────────────────────────────────
banner("ENDPOINT RUN — Qwen/Qwen2.5-0.5B-Instruct")
print(f"  Creating Serverless endpoint on {ep_model['preset']} / {ep_model['instance_type']}...")

ep_result = _run_endpoint(
    tasks, [ep_model],
    task_label="Factual Q&A", first_scorer="reference_match",
    task_file=TASK_FILE,
)

ep_row     = ep_result["leaderboard"][0]
ep_samples = ep_result["samples"]

# ── post-run: confirm teardown ────────────────────────────────────────────────
banner("TEARDOWN CHECK")
remaining = orchestrator.list_endpoints()
still_active = [e for e in remaining if e.get("status", {}).get("state") not in ("DELETING", "DELETED")]
print(f"  Endpoints after run: {len(remaining)} listed")
for e in remaining:
    st = e.get("status", {}).get("state", "?")
    print(f"    {e['metadata']['id']} [{st}]")
print(f"  Active (non-deleted): {len(still_active)}")
teardown_ok = len(still_active) == 0

# ── merge samples ─────────────────────────────────────────────────────────────
samples_map: dict = {}
for result in [tf_result, ep_result]:
    for s in result.get("samples", []):
        key = s["q"]
        if key not in samples_map:
            samples_map[key] = {k: v for k, v in s.items() if k != "answers"}
            samples_map[key]["answers"] = {}
        samples_map[key]["answers"].update(s.get("answers", {}))

leaderboard = sorted(
    tf_result["leaderboard"] + ep_result["leaderboard"],
    key=lambda r: r["accuracy"], reverse=True,
)

# ── build results dict ────────────────────────────────────────────────────────
results = {
    "meta": {
        "mode":       "auto",
        "task":       TASK_FILE,
        "task_name":  "Factual Q&A",
        "scorer":     "reference_match",
        "n_models":   2,
        "n_tasks":    N_TASKS,
        "benchmark":  "Factual Q&A",
        "routing":    routing,
    },
    "leaderboard": leaderboard,
    "samples":     list(samples_map.values()),
}

(ROOT / "results" / "results.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
)

# ── final report ──────────────────────────────────────────────────────────────
banner("MERGED LEADERBOARD")
print(f"  {'Model':<42} {'Basis':<12} {'Score':>6} {'Correct':>8} {'MeanLat':>8} {'$/1Ktok':>10} {'RunCost':>10}")
print(f"  {'─'*42} {'─'*12} {'─'*6} {'─'*8} {'─'*8} {'─'*10} {'─'*10}")
for r in leaderboard:
    name   = r["model"].split("/")[1][:40]
    basis  = r["cost_basis"]
    acc    = f"{r['accuracy']*100:.1f}%"
    cor    = f"{r['correct']}/{r['n']}"
    lat    = f"{r['mean_latency_s']:.2f}s"
    c1k    = f"${r['cost_per_1k_tokens_usd']:.5f}" if r.get("cost_per_1k_tokens_usd") is not None else "—"
    cost   = f"${r['total_run_cost_usd']:.4f}"    if r.get("total_run_cost_usd") is not None else "—"
    print(f"  {name:<42} {basis:<12} {acc:>6} {cor:>8} {lat:>8} {c1k:>10} {cost:>10}")

print()
banner("ENDPOINT COST BREAKDOWN (Qwen2.5-0.5B)")
if ep_row.get("deploy_cost_usd") is not None:
    rate    = ep_model["rate_hr"]
    dep_s   = ep_row["t_ready"] - ep_row["t_created"]
    eval_s  = ep_row["t_eval_done"] - ep_row["t_ready"]
    total_s = ep_row["t_eval_done"] - ep_row["t_created"]
    print(f"  t_created   = {ts(ep_row['t_created'])}")
    print(f"  t_ready     = {ts(ep_row['t_ready'])}   (+{dep_s/60:.1f} min)")
    print(f"  t_eval_done = {ts(ep_row['t_eval_done'])}   (+{eval_s:.0f}s from ready)")
    print(f"  deploy cost = {dep_s/60:.1f} min × ${rate}/hr  =  ${ep_row['deploy_cost_usd']:.4f}")
    print(f"  eval cost   = {eval_s:.0f}s × ${rate}/hr       =  ${ep_row['eval_cost_usd']:.6f}")
    print(f"  total cost  = {total_s/60:.1f} min × ${rate}/hr  =  ${ep_row['total_run_cost_usd']:.4f}")
    print(f"  $/1K tok (ss)                      =  ${ep_row['cost_per_1k_tokens_usd']:.5f}")
else:
    print("  (cost fields not populated — endpoint run may have failed)")

banner("TOKEN FACTORY LATENCY (Qwen3-32B)")
print(f"  Wall time for {N_TASKS} items: {tf_elapsed:.1f}s")
print(f"  Mean per-item latency : {tf_lat:.2f}s")
print(f"  p95 latency           : {tf_row['p95_latency_s']:.2f}s")

banner("ROUTING METADATA")
print(f"  meta.mode    : {results['meta']['mode']}")
print(f"  endpoint     : {results['meta']['routing']['endpoint']}")
print(f"  tokenfactory : {results['meta']['routing']['tokenfactory']}")

banner("TEARDOWN")
if teardown_ok:
    print("  ✓ Clean — 0 active endpoints remaining")
else:
    print(f"  ✗ WARNING: {len(still_active)} endpoint(s) still active — delete manually!")
    for e in still_active:
        print(f"    {e['metadata']['id']}")

print()
print("  results.json saved to results/results.json")
print()
banner("PHASE 2 COMPLETE")
print(f"  {EP_MODEL}")
print(f"    → endpoint · accuracy {ep_row['correct']}/{ep_row['n']}")
if ep_row.get("total_run_cost_usd") is not None:
    print(f"    → total ${ep_row['total_run_cost_usd']:.4f} ({dep_s/60:.1f} min deploy + {eval_s:.0f}s eval)")
print(f"  {TF_MODEL}")
print(f"    → tokenfactory · accuracy {tf_row['correct']}/{tf_row['n']}")
print(f"    → mean lat {tf_lat:.2f}s")
