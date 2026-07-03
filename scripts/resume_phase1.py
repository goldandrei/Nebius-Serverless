#!/usr/bin/env python3
"""
Resume Phase 1 from an already-created Serverless endpoint.

The endpoint was created by the orchestrator's create_endpoint() call
(nebius ai endpoint create --async ...) but the Python JSON-parse of
the async response failed. The endpoint itself is live.

This script resumes at wait_ready() and runs through eval + delete
via the same orchestrator functions used in _run_endpoint().
"""
import datetime
import json
import os
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

from src import orchestrator, scoring
from src.eval_runner import _make_embed

ENDPOINT_ID  = "aiendpoint-e00sqvcjtk9yanbtse"
AUTH_TOKEN   = "7c5fbf3f344af1e5e7958bb01866b29ce83be668bdaf4dcc35f8b6f62ccfb2c2"
MODEL_ID     = "Qwen/Qwen2.5-0.5B-Instruct"
RATE_HR      = 1.55
# created_at from the API (UTC)
T_CREATED_ISO = "2026-07-03T08:54:56.220789Z"


def ts(t: float) -> str:
    return datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).strftime("%H:%M:%S UTC")


def load_tasks(n: int = 3) -> list:
    path = ROOT / "data" / "factual_qa.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:n]


t_created = datetime.datetime.fromisoformat(T_CREATED_ISO.replace("Z", "+00:00")).timestamp()

print("=" * 60)
print("Phase 1 — Nebius Serverless AI Endpoint (real run)")
print(f"  endpoint_id = {ENDPOINT_ID}")
print(f"  model       = {MODEL_ID}")
print(f"  t_created   = {ts(t_created)}")
print("=" * 60)

# Show current state
print("\nCurrent endpoint state:")
ep = orchestrator.get_endpoint(ENDPOINT_ID)
print(f"  state           = {ep.get('status', {}).get('state')}")
print(f"  public_endpoint = {ep.get('status', {}).get('public_endpoints', [])}")

results = []
try:
    # ── wait_ready via orchestrator ───────────────────────────────
    print("\n[1/3] orchestrator.wait_ready() ...")
    base_url = orchestrator.wait_ready(ENDPOINT_ID, AUTH_TOKEN, timeout_s=900)
    t_ready  = time.time()
    print(f"  base_url = {base_url}")
    print(f"  t_ready  = {ts(t_ready)}  (+{(t_ready - t_created)/60:.1f} min from create)")

    # ── run 3 factual_qa items ────────────────────────────────────
    print("\n[2/3] Running 3 factual_qa items ...")
    import openai
    embed  = _make_embed()
    client = openai.OpenAI(base_url=f"{base_url}/v1", api_key=AUTH_TOKEN)
    tasks  = load_tasks(3)

    for i, task in enumerate(tasks, 1):
        msgs = []
        if task.get("instruction"):
            msgs.append({"role": "system", "content": task["instruction"]})
        msgs.append({"role": "user", "content": task["input"]})

        t0   = time.time()
        resp = client.chat.completions.create(
            model=MODEL_ID, messages=msgs, temperature=0
        )
        lat        = time.time() - t0
        raw        = resp.choices[0].message.content.strip()
        out_tokens = resp.usage.completion_tokens
        in_tokens  = resp.usage.prompt_tokens

        # Simple lexical correctness check (reference_match uses embedding)
        reference = task.get("reference", "")
        # check if key phrase from reference appears in answer
        key = reference.split("—")[0].split(".")[0].strip()[:30].lower()
        correct = key in raw.lower() if key else False

        s, _ = scoring.score(raw, task)   # full scorer without embed
        correct = s >= 0.5

        print(f"\n  Item {i}/{len(tasks)}:")
        print(f"    Q:     {task['input']}")
        print(f"    A:     {raw}")
        print(f"    ref:   {reference[:120]}")
        print(f"    score={s:.3f}  correct={correct}  lat={lat:.2f}s  "
              f"in={in_tokens}  out={out_tokens}")
        results.append({"score": s, "correct": correct, "lat": lat,
                        "out_tokens": out_tokens, "in_tokens": in_tokens})

    t_eval_done = time.time()

finally:
    # ── delete via orchestrator ───────────────────────────────────
    print("\n[3/3] orchestrator.delete_endpoint() ...")
    orchestrator.delete_endpoint(ENDPOINT_ID)

# ── verify teardown ───────────────────────────────────────────────
remaining = orchestrator.list_endpoints()
active = [e for e in remaining
          if e.get("status", {}).get("state") not in ("DELETING", "DELETED")]
print(f"\nRemaining active endpoints: {len(active)}")
for e in remaining:
    st = e.get("status", {}).get("state", "?")
    print(f"  {e['metadata']['id']} [{st}]")

# ── cost report ───────────────────────────────────────────────────
deploy_s    = t_ready - t_created
eval_s      = t_eval_done - t_ready
total_s     = t_eval_done - t_created
deploy_cost = deploy_s / 3600 * RATE_HR
eval_cost   = eval_s   / 3600 * RATE_HR
total_cost  = total_s  / 3600 * RATE_HR
total_out   = sum(r["out_tokens"] for r in results) or 1
c1k_steady  = eval_cost / total_out * 1000
n_correct   = sum(r["correct"] for r in results)
accuracy    = n_correct / len(results) if results else 0
mean_lat    = sum(r["lat"] for r in results) / len(results) if results else 0

print()
print("=" * 60)
print("PHASE 1 RESULTS  — Nebius Serverless AI Endpoint")
print("=" * 60)
print(f"  endpoint_id = {ENDPOINT_ID}")
print(f"  model       = {MODEL_ID}")
print(f"  platform    = gpu-l40s-a / 1gpu-8vcpu-32gb")
print()
print(f"  t_created   = {ts(t_created)}")
print(f"  t_ready     = {ts(t_ready)}   (+{deploy_s/60:.1f} min)")
print(f"  t_eval_done = {ts(t_eval_done)}   (+{eval_s:.0f}s from ready)")
print()
print(f"  deploy cost = {deploy_s/60:.1f} min × ${RATE_HR}/hr  =  ${deploy_cost:.4f}")
print(f"  eval cost   = {eval_s:.0f}s × ${RATE_HR}/hr        =  ${eval_cost:.6f}")
print(f"  total cost  = {total_s/60:.1f} min × ${RATE_HR}/hr  =  ${total_cost:.4f}")
print(f"  output tok  = {total_out} tokens")
print(f"  $/1K tok (steady-state eval)    =  ${c1k_steady:.5f}")
print()
print(f"  accuracy    = {n_correct}/{len(results)} = {accuracy:.0%}")
print(f"  mean lat    = {mean_lat:.2f}s per item")
print()
print("Endpoint deleted. Phase 1 milestone complete.")
