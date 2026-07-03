#!/usr/bin/env python3
"""
Phase 1 — Nebius Serverless AI Endpoint validation.

Creates one endpoint (Qwen/Qwen2.5-0.5B-Instruct, vllm/vllm-openai:latest,
gpu-l40s-a, 1gpu-8vcpu-32gb), runs 3 factual_qa items, deletes the endpoint,
and reports t_created / t_ready / t_eval_done timestamps and costs.

Requirements:
  - nebius CLI authenticated (Linux/macOS: in PATH; Windows: installed in WSL)
  - NEBIUS_PROJECT_ID, NEBIUS_SUBNET_ID set in .env

Usage:
  uv run --with pyyaml,openai,requests python scripts/run_phase1.py
"""
import datetime
import json
import os
import secrets
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

for var in ("NEBIUS_PROJECT_ID", "NEBIUS_SUBNET_ID"):
    if not os.environ.get(var):
        print(f"ERROR: {var} not set in .env")
        sys.exit(1)

from src import orchestrator

MODEL_ID    = "Qwen/Qwen2.5-0.5B-Instruct"
PLATFORM    = "gpu-l40s-a"
PRESET      = "1gpu-8vcpu-32gb"
RATE_HR     = 1.55


def ts(t: float) -> str:
    return datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).strftime("%H:%M:%S UTC")


def load_tasks(n: int = 3) -> list:
    path = ROOT / "data" / "factual_qa.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:n]


def main():
    print("=" * 60)
    print("Phase 1 — Nebius Serverless AI Endpoint validation")
    print(f"  model    : {MODEL_ID}")
    print(f"  platform : {PLATFORM}  preset: {PRESET}")
    print(f"  rate     : ${RATE_HR}/hr (L40S)")
    print("=" * 60)

    existing = orchestrator.list_endpoints()
    if existing:
        print(f"\nWARNING: {len(existing)} existing endpoint(s):")
        for ep in existing:
            print(f"  {ep['metadata']['id']} — {ep['metadata']['name']} "
                  f"[{ep.get('status', {}).get('state', '?')}]")
    else:
        print("\nNo existing Serverless endpoints — clean slate.")

    # ── create ────────────────────────────────────────────────────
    print("\n[1/4] Creating endpoint...")
    auth_token  = secrets.token_hex(32)
    t_created   = time.time()
    endpoint_id = orchestrator.create_endpoint(MODEL_ID, PLATFORM, PRESET, auth_token)
    print(f"  endpoint_id = {endpoint_id}")
    print(f"  t_created   = {ts(t_created)}")

    results = []
    try:
        # ── wait ready ────────────────────────────────────────────
        print("\n[2/4] Waiting for ready...")
        base_url = orchestrator.wait_ready(endpoint_id, auth_token, timeout_s=900)
        t_ready  = time.time()
        print(f"  t_ready     = {ts(t_ready)}  (+{(t_ready-t_created)/60:.1f} min)")

        # ── run 3 factual_qa items ────────────────────────────────
        print("\n[3/4] Running 3 factual_qa items...")
        import openai
        client = openai.OpenAI(base_url=f"{base_url}/v1", api_key=auth_token)
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

            reference = task.get("reference", "")
            correct   = reference[:20].lower() in raw.lower() if reference else False

            print(f"\n  Item {i}/{len(tasks)}:")
            print(f"    Q:   {task['input']}")
            print(f"    A:   {raw[:200]}")
            print(f"    ref: {reference[:100]}")
            print(f"    ok={correct}  lat={lat:.2f}s  in={in_tokens}  out={out_tokens}")
            results.append({"correct": correct, "lat": lat,
                            "out_tokens": out_tokens, "in_tokens": in_tokens})

        t_eval_done = time.time()

    finally:
        # ── delete ────────────────────────────────────────────────
        print("\n[4/4] Deleting endpoint...")
        orchestrator.delete_endpoint(endpoint_id)

    remaining = orchestrator.list_endpoints()
    print(f"\n  Remaining endpoints: {len(remaining)}")

    # ── cost report ───────────────────────────────────────────────
    deploy_s    = t_ready - t_created
    eval_s      = t_eval_done - t_ready
    total_s     = t_eval_done - t_created
    deploy_cost = deploy_s / 3600 * RATE_HR
    eval_cost   = eval_s   / 3600 * RATE_HR
    total_cost  = total_s  / 3600 * RATE_HR
    total_out   = sum(r["out_tokens"] for r in results) or 1
    c1k_steady  = eval_cost / total_out * 1000

    n_correct = sum(r["correct"] for r in results)
    accuracy  = n_correct / len(results) if results else 0

    print()
    print("=" * 60)
    print("PHASE 1 RESULTS")
    print("=" * 60)
    print(f"  t_created   = {ts(t_created)}")
    print(f"  t_ready     = {ts(t_ready)}   (+{deploy_s/60:.1f} min)")
    print(f"  t_eval_done = {ts(t_eval_done)}   (+{eval_s:.0f}s from ready)")
    print()
    print(f"  deploy time  = {deploy_s/60:.1f} min  → deploy_cost  = ${deploy_cost:.4f}")
    print(f"  eval time    = {eval_s:.0f} s       → eval_cost    = ${eval_cost:.6f}")
    print(f"  total cost                    = ${total_cost:.4f}  "
          f"({RATE_HR} $/hr × {total_s/3600:.3f} hr)")
    print(f"  output tokens = {total_out}           $/1K tok (ss) = ${c1k_steady:.5f}")
    print()
    print(f"  accuracy = {n_correct}/{len(results)} = {accuracy:.0%}")
    print()
    print("Phase 1 complete. Endpoint deleted. Ready for Phase 2.")


if __name__ == "__main__":
    main()
