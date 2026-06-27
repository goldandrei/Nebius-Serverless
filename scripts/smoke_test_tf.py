#!/usr/bin/env python3
"""
Smoke test: run 3 factual_qa items against Token Factory using Llama-3.3-70B.
Prints raw answers and per-item scores. Does NOT touch Nebius Serverless endpoints or jobs.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src.eval_runner import load_tasks, load_catalog, task_expected
from src import scoring

MODEL    = "meta-llama/Llama-3.3-70B-Instruct"
N_ITEMS  = 3
TASK     = "factual_qa"

def main():
    import os, time, openai

    base_url = os.environ["NEBIUS_BASE_URL"]
    api_key  = os.environ["NEBIUS_API_KEY"]

    # Confirm model is in catalog and tokenfactory_ok
    catalog = load_catalog()
    spec    = catalog.get(MODEL)
    if not spec:
        print(f"ERROR: {MODEL} not in catalog.yaml", file=sys.stderr); sys.exit(1)
    if not spec.get("tokenfactory_ok"):
        print(f"ERROR: {MODEL} has tokenfactory_ok=false in catalog", file=sys.stderr); sys.exit(1)

    tasks = load_tasks(task_file=TASK)[:N_ITEMS]
    print(f"Backend : Token Factory ({base_url})")
    print(f"Model   : {MODEL}")
    print(f"Task    : {TASK}  ({N_ITEMS} items)\n")
    print("=" * 72)

    client = openai.OpenAI(base_url=base_url, api_key=api_key)

    total_score = 0.0
    for i, task in enumerate(tasks, 1):
        msgs = []
        if task.get("instruction"):
            msgs.append({"role": "system", "content": task["instruction"]})
        msgs.append({"role": "user", "content": task["input"]})

        t0   = time.time()
        resp = client.chat.completions.create(model=MODEL, messages=msgs, temperature=0)
        lat  = time.time() - t0

        raw        = resp.choices[0].message.content
        out_tokens = resp.usage.completion_tokens

        s, detail = scoring.score(raw, task)
        total_score += s

        print(f"[{i}/{N_ITEMS}] Q: {task['input']}")
        print(f"  expected : {task_expected(task)[:120]}")
        print(f"  answer   : {raw[:300]}")
        print(f"  score    : {s:.3f}  |  latency: {lat:.2f}s  |  out_tokens: {out_tokens}  |  {detail}")
        print()

    print("=" * 72)
    print(f"accuracy: {total_score / N_ITEMS:.1%}  ({int(total_score)}/{N_ITEMS} correct)")

if __name__ == "__main__":
    main()
