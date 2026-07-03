#!/usr/bin/env python3
"""
Phase 1 endpoint validation.

Creates one dedicated endpoint (meta-llama/Meta-Llama-3.1-8B-Instruct-fast,
gpu-l40s-d, 1 GPU, eu-north1), runs 2-3 factual_qa items, deletes endpoint,
and reports t_created / t_ready / t_eval_done timestamps and costs.

Usage:
  uv run --with pyyaml,openai,requests python scripts/run_phase1.py
"""
import datetime
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

API_KEY = os.environ.get("NEBIUS_API_KEY")
if not API_KEY:
    print("ERROR: NEBIUS_API_KEY not set")
    sys.exit(1)

from src import orchestrator

MODEL_NAME  = "meta-llama/Meta-Llama-3.1-8B-Instruct-fast"
FLAVOR      = "base"
GPU_TYPE    = "gpu-l40s-d"
GPU_COUNT   = 1
REGION      = "eu-north1"
RATE_HR     = 1.55


def ts(t: float) -> str:
    return datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).strftime("%H:%M:%S UTC")


def load_tasks(n: int = 3) -> list:
    path = ROOT / "data" / "factual_qa.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:n]


def main():
    print("=" * 60)
    print("Phase 1 — Endpoint validation")
    print(f"Model  : {MODEL_NAME}")
    print(f"GPU    : {GPU_TYPE} × {GPU_COUNT}")
    print(f"Region : {REGION}")
    print("=" * 60)

    # ── verify no existing endpoints ──────────────────────────────
    existing = orchestrator.list_endpoints()
    if existing:
        print(f"\nWARNING: {len(existing)} existing endpoint(s) found:")
        for ep in existing:
            print(f"  {ep.get('id')} — {ep.get('model_name')} [{ep.get('deployment', {}).get('status')}]")
        print("Proceeding anyway.")
    else:
        print("\nNo existing endpoints — clean slate.")

    # ── create ────────────────────────────────────────────────────
    print("\n[1/4] Creating endpoint...")
    t_created = time.time()
    endpoint_id, routing_key = orchestrator.create_endpoint(
        MODEL_NAME, FLAVOR, GPU_TYPE, GPU_COUNT, REGION
    )
    print(f"  endpoint_id  = {endpoint_id}")
    print(f"  routing_key  = {routing_key}")
    print(f"  t_created    = {ts(t_created)}")

    try:
        # ── wait ready ────────────────────────────────────────────
        print("\n[2/4] Waiting for ready...")
        base_url = orchestrator.wait_ready(endpoint_id, timeout_s=900)
        t_ready  = time.time()
        deploy_min = (t_ready - t_created) / 60
        print(f"  t_ready      = {ts(t_ready)}")
        print(f"  deploy time  = {deploy_min:.1f} min")

        # ── run factual_qa items ──────────────────────────────────
        print("\n[3/4] Running 3 factual_qa items...")
        import openai
        client = openai.OpenAI(base_url=base_url, api_key=API_KEY)
        tasks  = load_tasks(3)
        results = []

        for i, task in enumerate(tasks, 1):
            msgs = []
            if task.get("instruction"):
                msgs.append({"role": "system", "content": task["instruction"]})
            msgs.append({"role": "user", "content": task["input"]})

            t0   = time.time()
            resp = client.chat.completions.create(
                model=routing_key, messages=msgs, temperature=0
            )
            lat        = time.time() - t0
            raw        = resp.choices[0].message.content.strip()
            out_tokens = resp.usage.completion_tokens
            in_tokens  = resp.usage.prompt_tokens

            reference = task.get("reference", "")
            # simple lexical check
            correct = reference.lower() in raw.lower() if reference else False

            print(f"\n  Item {i}/{len(tasks)}:")
            print(f"    Q: {task['input'][:80]}")
            print(f"    A: {raw[:120]}")
            print(f"    ref: {reference}")
            print(f"    correct: {correct}  lat: {lat:.2f}s  in={in_tokens} out={out_tokens}")
            results.append({"correct": correct, "lat": lat, "out_tokens": out_tokens})

        t_eval_done = time.time()

    finally:
        # ── delete ────────────────────────────────────────────────
        print("\n[4/4] Deleting endpoint...")
        orchestrator.delete_endpoint(endpoint_id)
        t_deleted = time.time()

    # ── verify none remain ────────────────────────────────────────
    remaining = orchestrator.list_endpoints()
    print(f"\n  Remaining endpoints: {len(remaining)}")
    for ep in remaining:
        print(f"    {ep.get('id')} [{ep.get('deployment', {}).get('status')}]")

    # ── cost report ───────────────────────────────────────────────
    deploy_s    = t_ready - t_created
    eval_s      = t_eval_done - t_ready
    total_s     = t_eval_done - t_created
    deploy_cost = deploy_s / 3600 * RATE_HR
    eval_cost   = eval_s / 3600 * RATE_HR
    total_cost  = total_s / 3600 * RATE_HR
    total_out   = sum(r["out_tokens"] for r in results) or 1
    c1k_steady  = eval_cost / total_out * 1000

    n_correct = sum(r["correct"] for r in results)
    accuracy  = n_correct / len(results) if results else 0.0

    print()
    print("=" * 60)
    print("PHASE 1 RESULTS")
    print("=" * 60)
    print(f"  t_created   = {ts(t_created)}")
    print(f"  t_ready     = {ts(t_ready)}  (+{deploy_s/60:.1f} min)")
    print(f"  t_eval_done = {ts(t_eval_done)}  (+{eval_s/60:.1f} min)")
    print()
    print(f"  deploy time  = {deploy_s/60:.1f} min  → ${deploy_cost:.4f}")
    print(f"  eval time    = {eval_s/60:.1f} min  → ${eval_cost:.4f}")
    print(f"  total cost   = ${total_cost:.4f}  ({RATE_HR} $/hr × {total_s/3600:.3f} hr)")
    print(f"  $/1K tok (ss)= ${c1k_steady:.5f}  ({total_out} output tokens)")
    print()
    print(f"  accuracy     = {n_correct}/{len(results)} = {accuracy:.0%}")
    print()
    print("Phase 1 complete. Endpoint deleted. Ready for Phase 2.")


if __name__ == "__main__":
    main()
