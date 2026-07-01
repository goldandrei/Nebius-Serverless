#!/usr/bin/env python3
"""
Smoke test: --backend local on 3 factual_qa items.
Shows raw model answer + embedding similarity score.
Run from repo root: uv run --with pyyaml,openai,boto3,requests python scripts/smoke_local.py
"""
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

import openai
from src.eval_runner import _make_embed, _load_jsonl, DATA_DIR
from src import scoring

LOCAL_BASE_URL = os.environ.get("LOCAL_BASE_URL", "http://localhost:8000/v1")
LOCAL_MODEL    = os.environ.get("LOCAL_MODEL",    "Qwen/Qwen2.5-0.5B-Instruct")
NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY")

# Pick 3 items: sky, photosynthesis, vaccines
SAMPLE_IDS = {1, 2, 5}

tasks = [t for t in _load_jsonl(DATA_DIR / "factual_qa.jsonl") if t["id"] in SAMPLE_IDS]

print(f"Local vLLM  : {LOCAL_BASE_URL}")
print(f"Local model : {LOCAL_MODEL}")
print(f"Embedding   : {'Qwen3-Embedding-8B via Token Factory' if NEBIUS_API_KEY else 'bag-of-words mock (no NEBIUS_API_KEY)'}")
print(f"Items       : {len(tasks)}\n")

embed  = _make_embed()
client = openai.OpenAI(base_url=LOCAL_BASE_URL, api_key="local")

PASS_THRESHOLD = 0.50   # harness threshold; embedding cosine for correct answer typically 0.75+

for task in tasks:
    msgs = []
    if task.get("instruction"):
        msgs.append({"role": "system", "content": task["instruction"]})
    msgs.append({"role": "user", "content": task["input"]})

    t0   = time.time()
    resp = client.chat.completions.create(model=LOCAL_MODEL, messages=msgs, temperature=0)
    lat  = time.time() - t0
    raw  = resp.choices[0].message.content
    toks = resp.usage.completion_tokens

    emb_ref = embed(task["reference"])
    emb_ans = embed(raw)
    sim = scoring._cosine(emb_ans, emb_ref)

    correct = sim >= PASS_THRESHOLD
    verdict = "PASS" if correct else "FAIL"

    print(f"Q  : {task['input']}")
    print(f"REF: {task['reference']}")
    print(f"ANS: {raw}")
    print(f"SIM: {sim:.4f}  tokens={toks}  lat={lat:.2f}s  [{verdict}]")
    print()
