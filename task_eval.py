#!/usr/bin/env python3
"""
A real 'your task' eval, end to end.

Task: turn a plain-language request into a strict JSON assistant command
      {"intent": ..., "slots": {...}}.

This is exactly the kind of task NO public leaderboard measures, because the
intents, the slot schema, and "what counts as correct" are yours. Scoring is
PROGRAMMATIC with partial credit:

  - invalid JSON            -> item score 0   (hard gate)
  - else item score = 0.5 * intent_match + 0.5 * slot_F1

slot_F1 compares the predicted {key: value} slots to the gold slots; a slot
counts as correct only if BOTH key and (normalized) value match.

Models are mocked here (the sandbox can't reach real endpoints). To go live,
replace `mock_generate()` with a real OpenAI-compatible call to your Nebius
endpoint and keep everything else identical.
"""

import json
import random
from pathlib import Path

HERE = Path(__file__).parent
TASK_FILE = HERE / "tasks" / "assistant_commands.jsonl"


# ---- scoring -----------------------------------------------------------------
def norm(v):
    """Normalize a slot value for comparison."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower()
    # let "30" match 30
    try:
        return float(s)
    except ValueError:
        return s


def slot_f1(gold: dict, pred: dict):
    if not gold and not pred:
        return 1.0
    matches = sum(1 for k, gv in gold.items()
                  if k in pred and norm(pred[k]) == norm(gv))
    precision = matches / len(pred) if pred else 0.0
    recall = matches / len(gold) if gold else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_item(raw_output: str, gold: dict):
    """Returns (item_score, detail_dict)."""
    try:
        pred = json.loads(raw_output)
        assert isinstance(pred, dict)
    except (json.JSONDecodeError, AssertionError):
        return 0.0, {"valid_json": False, "intent_match": False, "slot_f1": 0.0}
    intent_match = pred.get("intent") == gold["intent"]
    f1 = slot_f1(gold.get("slots", {}), pred.get("slots", {}) or {})
    item = 0.5 * (1.0 if intent_match else 0.0) + 0.5 * f1
    return item, {"valid_json": True, "intent_match": intent_match,
                  "slot_f1": round(f1, 3)}


# ---- mock models (stand-ins for Nebius endpoints) ----------------------------
# Each model degrades the gold answer according to its skill profile.
MODELS = [
    {"id": "qwen2.5-0.5b", "p_bad_json": 0.25, "p_wrong_intent": 0.30, "p_slot_err": 0.45},
    {"id": "qwen2.5-1.5b", "p_bad_json": 0.05, "p_wrong_intent": 0.10, "p_slot_err": 0.22},
    {"id": "llama-3.2-3b", "p_bad_json": 0.00, "p_wrong_intent": 0.03, "p_slot_err": 0.10},
]
OTHER_INTENTS = ["set_reminder", "set_timer", "control_light",
                 "set_thermostat", "play_music", "weather_query"]


def mock_generate(spec, item, seed):
    """Produce a (possibly corrupted) JSON command string for one item."""
    rng = random.Random(seed)
    gold = item["gold"]

    if rng.random() < spec["p_bad_json"]:
        # emit something that won't parse as a clean object
        return "Sure! " + json.dumps(gold)[:-1]  # truncated -> invalid

    intent = gold["intent"]
    if rng.random() < spec["p_wrong_intent"]:
        intent = rng.choice([i for i in OTHER_INTENTS if i != gold["intent"]])

    slots = {}
    for k, v in gold["slots"].items():
        r = rng.random()
        if r < spec["p_slot_err"]:
            # either drop the slot or corrupt its value
            if rng.random() < 0.5:
                continue  # dropped
            slots[k] = (v + 1) if isinstance(v, (int, float)) and not isinstance(v, bool) \
                else str(v) + "_x"
        else:
            slots[k] = v
    return json.dumps({"intent": intent, "slots": slots})


# ---- run ---------------------------------------------------------------------
def load_tasks(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    tasks = load_tasks(TASK_FILE)
    agg = {m["id"]: {"score": 0.0, "intent_ok": 0, "f1": 0.0, "bad_json": 0}
           for m in MODELS}
    show_examples = []

    for item in tasks:
        ex = {"input": item["input"], "gold": item["gold"], "preds": {}}
        for m in MODELS:
            raw = mock_generate(m, item, seed=f"{m['id']}:{item['id']}")
            s, d = score_item(raw, item["gold"])
            agg[m["id"]]["score"] += s
            agg[m["id"]]["intent_ok"] += int(d["intent_match"])
            agg[m["id"]]["f1"] += d["slot_f1"]
            agg[m["id"]]["bad_json"] += int(not d["valid_json"])
            ex["preds"][m["id"]] = {"raw": raw, "score": round(s, 3), **d}
        if item["id"] in (4, 1):
            show_examples.append(ex)

    n = len(tasks)
    print(f"=== 'Your task' eval: NL -> assistant command  (n={n}, programmatic scoring) ===\n")
    hdr = f"{'model':<16}{'score':>8}{'intent_acc':>12}{'slot_F1':>9}{'bad_json':>10}"
    print(hdr); print("-" * len(hdr))
    leaderboard = []
    for m in MODELS:
        a = agg[m["id"]]
        row = {"model": m["id"], "score": round(a["score"]/n, 3),
               "intent_acc": round(a["intent_ok"]/n, 3),
               "slot_f1": round(a["f1"]/n, 3),
               "bad_json_rate": round(a["bad_json"]/n, 3)}
        leaderboard.append(row)
        print(f"{row['model']:<16}{row['score']:>8.3f}{row['intent_acc']*100:>11.0f}%"
              f"{row['slot_f1']:>9.3f}{row['bad_json_rate']*100:>9.0f}%")

    print("\n--- inspector: how each model did on two items ---")
    for ex in show_examples:
        print(f"\ninput: {ex['input']}")
        print(f"gold : {json.dumps(ex['gold'])}")
        for mid, p in ex["preds"].items():
            flag = "OK " if p["valid_json"] else "BAD-JSON"
            print(f"  {mid:<14} score={p['score']:<5} intent={'Y' if p['intent_match'] else 'N'}"
                  f" f1={p['slot_f1']:<5} {flag}  {p['raw']}")

    (HERE / "task_results.json").write_text(
        json.dumps({"leaderboard": leaderboard}, indent=2))
    print("\nwrote: task_results.json")


if __name__ == "__main__":
    main()
