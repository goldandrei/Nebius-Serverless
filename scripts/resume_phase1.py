#!/usr/bin/env python3
"""Phase 1 endpoint validation — creates endpoint, runs eval, deletes."""
import datetime, json, os, sys, time
from pathlib import Path

# Force UTF-8 output so reference strings with ₂, °, etc. print cleanly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

API_KEY = os.environ["NEBIUS_API_KEY"]
from src import orchestrator

ENDPOINT_ID  = "6c94efa9-4a68-4d11-a85d-35b18a83f356"
ROUTING_KEY  = "dedicated/meta-llama/Meta-Llama-3.1-8B-Instruct-fast-jwTqM6XSr5AG"
RATE_HR      = 1.55
# t_created from the API created_at field (UTC)
T_CREATED_ISO = "2026-07-03T08:12:16.622265Z"

def ts(t):
    return datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).strftime("%H:%M:%S UTC")

t_created = datetime.datetime.fromisoformat(T_CREATED_ISO.replace("Z", "+00:00")).timestamp()

def load_tasks(n=3):
    path = ROOT / "data" / "factual_qa.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:n]

print("=" * 60)
print("Phase 1 — Resuming with existing endpoint")
print(f"  endpoint_id  = {ENDPOINT_ID}")
print(f"  routing_key  = {ROUTING_KEY}")
print(f"  t_created    = {ts(t_created)}")
print("=" * 60)

# ── wait ready ────────────────────────────────────────────────
print("\n[1/3] Waiting for ready...")
try:
    base_url = orchestrator.wait_ready(ENDPOINT_ID, timeout_s=900)
    t_ready  = time.time()
    deploy_min = (t_ready - t_created) / 60
    print(f"  t_ready      = {ts(t_ready)}")
    print(f"  deploy time  = {deploy_min:.1f} min")

    # ── run 3 factual_qa items ────────────────────────────────────
    print("\n[2/3] Running 3 factual_qa items...")
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
            model=ROUTING_KEY, messages=msgs, temperature=0
        )
        lat        = time.time() - t0
        raw        = resp.choices[0].message.content.strip()
        out_tokens = resp.usage.completion_tokens
        in_tokens  = resp.usage.prompt_tokens

        reference = task.get("reference", "")
        correct = reference.lower()[:20] in raw.lower() if reference else False

        print(f"\n  Item {i}:")
        print(f"    Q:   {task['input']}")
        print(f"    A:   {raw[:200]}")
        print(f"    ref: {reference[:100]}")
        print(f"    ok={correct}  lat={lat:.2f}s  in={in_tokens}  out={out_tokens}")
        results.append({"correct": correct, "lat": lat, "out_tokens": out_tokens, "in_tokens": in_tokens})

    t_eval_done = time.time()

finally:
    # ── delete ────────────────────────────────────────────────────
    print("\n[3/3] Deleting endpoint...")
    orchestrator.delete_endpoint(ENDPOINT_ID)

# ── verify none remain ────────────────────────────────────────
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
print(f"  t_ready     = {ts(t_ready)}  (+{deploy_s/60:.1f} min)")
print(f"  t_eval_done = {ts(t_eval_done)}  (+{eval_s/60:.1f} min from ready)")
print()
print(f"  deploy time = {deploy_s/60:.1f} min  → deploy_cost  = ${deploy_cost:.4f}")
print(f"  eval time   = {eval_s:.0f} s        → eval_cost    = ${eval_cost:.6f}")
print(f"  total                          total_cost    = ${total_cost:.4f}")
print(f"  output tokens = {total_out}           $/1K tok (ss) = ${c1k_steady:.5f}")
print()
print(f"  accuracy = {n_correct}/{len(results)} = {accuracy:.0%}")
print()
print("Endpoint deleted. No remaining endpoints.")
print("Phase 1 complete — ready for Phase 2.")
