import json
import os
import random
import re
import secrets
import statistics
import time
from pathlib import Path

import yaml

from src import scoring
from src.cost import cost_per_1k_tokens, endpoint_cost

import hashlib
import math

ROOT          = Path(__file__).resolve().parent.parent
DATA_DIR      = ROOT / "data"
CATALOG_FILE  = ROOT / "config" / "catalog.yaml"
SELECTION_FILE = ROOT / "config" / "models.yaml"

_OTHER_INTENTS = [
    "set_reminder", "set_timer", "control_light",
    "set_thermostat", "play_music", "weather_query",
]

_PROSE_GOOD = [
    "Here are the key points as requested:\n• First item\n• Second item\n• Third item",
    "Based on the instructions:\n1. Point one\n2. Point two\n3. Point three",
    "Following the format requested, here is the answer with the necessary detail.",
]
_PROSE_WEAK = [
    "Sure! Here is some information about that topic.",
    "I can help with that. The answer involves several considerations.",
    "That's a great question. Let me provide some thoughts on the matter.",
]


# ── embedding ─────────────────────────────────────────────────────────────────

_EMBED_DIM = 512

def _mock_embed(text: str) -> list:
    """Deterministic bag-of-words vector via hashing trick — no API needed."""
    tokens = re.findall(r'\w+', text.lower())
    vec = [0.0] * _EMBED_DIM
    for tok in tokens:
        idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % _EMBED_DIM
        vec[idx] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _make_embed(api_key: str = None, base_url: str = None, model: str = None):
    """Return an embed callable. Falls back to mock if no API key is configured."""
    api_key  = api_key  or os.environ.get("NEBIUS_API_KEY")
    base_url = base_url or os.environ.get("NEBIUS_BASE_URL",
                                          "https://api.tokenfactory.nebius.com/v1/")
    model    = model    or os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")

    if not api_key:
        return _mock_embed

    import openai
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def _embed(text: str) -> list:
        resp = client.embeddings.create(model=model, input=text)
        return resp.data[0].embedding

    return _embed


# ── task loading ──────────────────────────────────────────────────────────────

def load_tasks(data_dir: Path = DATA_DIR, task_file: str = None) -> list:
    if task_file:
        path = data_dir / f"{task_file}.jsonl"
        return _load_jsonl(path)
    tasks = []
    for f in sorted(data_dir.glob("*.jsonl")):
        tasks.extend(_load_jsonl(f))
    return tasks


def _load_jsonl(path: Path) -> list:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def task_expected(task: dict) -> str:
    if "gold" in task:
        return json.dumps(task["gold"])
    if "reference" in task:
        return str(task["reference"])
    if "rubric" in task:
        return task["rubric"]
    return ""


_SCORER_LABELS = {
    "programmatic":    "Programmatic · code-based exact/numeric/json scoring",
    "reference_match": "Reference match · token F1 lexical similarity",
    "llm_judge":       "LLM judge · rubric-based scoring",
}

def available_tasks(data_dir: Path = DATA_DIR) -> list:
    _META = {
        "assistant_commands": {
            "name": "NL → JSON Command",
            "description": "Convert plain-language requests to structured JSON commands",
        },
        "factual_qa": {
            "name": "Factual Q&A",
            "description": "Short answers to factual questions",
        },
        "instruction_following": {
            "name": "Instruction Following",
            "description": "Open-ended responses evaluated against a rubric",
        },
        "custom": {
            "name": "Custom dataset",
            "description": "Assembled from your inputs via the dashboard builder",
        },
    }
    result = []
    for f in sorted(data_dir.glob("*.jsonl")):
        records = _load_jsonl(f)
        scorer  = records[0].get("scorer", "programmatic") if records else "programmatic"
        meta    = _META.get(f.stem, {"name": f.stem, "description": ""})
        result.append({
            **meta,
            "id":           f.stem,
            "scorer":       scorer,
            "scorer_label": _SCORER_LABELS.get(scorer, scorer),
            "n":            len(records),
        })
    return result


# ── model config ──────────────────────────────────────────────────────────────

def _short_id(hf_id: str) -> str:
    return hf_id.split("/")[-1].lower()


def load_catalog(catalog_file: Path = CATALOG_FILE) -> dict:
    with open(catalog_file, encoding="utf-8") as f:
        return {m["id"]: m for m in yaml.safe_load(f)["models"]}


def load_config(
    selection_file: Path = SELECTION_FILE,
    catalog_file: Path = CATALOG_FILE,
) -> list:
    catalog = load_catalog(catalog_file)
    with open(selection_file, encoding="utf-8") as f:
        selected_ids = yaml.safe_load(f)["compare"]

    missing = [mid for mid in selected_ids if mid not in catalog]
    if missing:
        raise ValueError(
            "Model(s) not found in catalog:\n" + "\n".join(f"  {m}" for m in missing)
        )
    result = []
    for mid in selected_ids:
        m = dict(catalog[mid])
        m["mock_id"] = _short_id(mid)
        result.append(m)
    return result


def list_catalog(catalog_file: Path = CATALOG_FILE) -> None:
    catalog = load_catalog(catalog_file)
    gated = {True: " [gated]", False: "", None: ""}
    tf    = {True: " [TF]",    False: "",  None: ""}
    print(f"{'ID':<52} {'preset':<16} {'rate_hr':>8}  notes")
    print("-" * 90)
    for m in catalog.values():
        print(
            f"{m['id']:<52} {m['preset']:<16} ${m['rate_hr']:>5.2f}/hr"
            f"{gated.get(m.get('gated'), '')}{tf.get(m.get('tokenfactory_ok'), '')}"
        )


# ── mock endpoint ─────────────────────────────────────────────────────────────

class MockEndpoint:
    def __init__(self, model_spec: dict):
        self.m = model_spec["mock"]

    def generate(self, task_item: dict, seed: str) -> dict:
        rng   = random.Random(seed)
        scorer = task_item.get("scorer", "programmatic")
        lat   = max(0.03, rng.gauss(self.m["lat_mean"], self.m["lat_sd"]))

        if scorer == "reference_match":
            return self._gen_factual(task_item, rng, lat)
        elif scorer == "llm_judge":
            return self._gen_prose(task_item, rng, lat)
        else:
            return self._gen_json(task_item, rng, lat)

    def _gen_factual(self, task_item: dict, rng: random.Random, lat: float) -> dict:
        ref   = str(task_item.get("reference", ""))
        wrong = task_item.get("wrong", [])
        if rng.random() < self.m.get("p_ref_err", 0.3):
            raw = rng.choice(wrong) if wrong else ref + "?"
        else:
            raw = ref
        return {"raw": raw, "latency_s": lat, "out_tokens": rng.randint(4, 12)}

    def _gen_prose(self, task_item: dict, rng: random.Random, lat: float) -> dict:
        jmean = self.m.get("judge_score_mean", 0.7)
        pool  = _PROSE_GOOD if jmean >= 0.65 else _PROSE_WEAK
        raw   = rng.choice(pool)
        return {"raw": raw, "latency_s": lat * 1.8, "out_tokens": rng.randint(30, 100)}

    def _gen_json(self, task_item: dict, rng: random.Random, lat: float) -> dict:
        gold    = task_item["gold"]
        compare = task_item.get("compare", "json_fields" if isinstance(gold, dict) else "exact")

        if not isinstance(gold, dict):
            p_err = self.m.get("p_wrong_intent", 0.15)
            if rng.random() < p_err:
                raw = str(gold) + "_wrong"
            else:
                raw = str(gold)
            return {"raw": raw, "latency_s": lat, "out_tokens": rng.randint(3, 10)}

        if rng.random() < self.m["p_bad_json"]:
            return {"raw": "Sure! " + json.dumps(gold)[:-1],
                    "latency_s": lat, "out_tokens": rng.randint(10, 30)}

        pred = {}
        for k, v in gold.items():
            if k == "intent":
                pred[k] = gold[k]
                if rng.random() < self.m["p_wrong_intent"]:
                    pred[k] = rng.choice([i for i in _OTHER_INTENTS if i != gold["intent"]])
            elif rng.random() < self.m["p_slot_err"]:
                if rng.random() < 0.5:
                    continue
                pred[k] = (v + 1) if isinstance(v, (int, float)) and not isinstance(v, bool) \
                    else str(v) + "_x"
            else:
                pred[k] = v

        return {"raw": json.dumps(pred), "latency_s": lat, "out_tokens": rng.randint(10, 40)}


# ── main eval loop ────────────────────────────────────────────────────────────

def run(backend: str = "mock", task_file: str = None) -> dict:
    """Run the eval harness.

    backend: "mock"         — fully offline, deterministic fake responses
             "tokenfactory" — Token Factory hosted API (NEBIUS_API_KEY + NEBIUS_BASE_URL)
             "endpoint"     — self-hosted Nebius Serverless Endpoints (creates/deletes GPU VMs)
    """
    tasks  = load_tasks(task_file=task_file)
    models = load_config()

    first_scorer = tasks[0].get("scorer", "programmatic") if tasks else "programmatic"
    task_meta    = {t["id"]: t for t in available_tasks()}
    task_label   = task_meta.get(task_file, {}).get("name", task_file or "mixed") if task_file else "mixed"

    if backend == "mock":
        return _run_mock(tasks, models, task_label, first_scorer, task_file)
    elif backend == "tokenfactory":
        return _run_tokenfactory(tasks, models, task_label, first_scorer, task_file)
    elif backend == "local":
        return _run_local(tasks, task_label, first_scorer, task_file)
    elif backend == "endpoint":
        return _run_endpoint(tasks, models, task_label, first_scorer, task_file)
    else:
        raise ValueError(f"Unknown backend {backend!r}; choose mock, tokenfactory, local, or endpoint")


def _build_leaderboard(models, per_model, mid_key, n_tasks):
    leaderboard = []
    for m in models:
        mid = m[mid_key]
        d   = per_model[mid]
        lat_sorted = sorted(d["lat"])
        p95 = lat_sorted[min(n_tasks - 1, int(round(0.95 * (n_tasks - 1))))]
        leaderboard.append({
            "model":                   mid,
            "preset":                  f"{m['preset']} / {m['instance_type']}",
            "accuracy":                round(d["score"] / n_tasks, 4),
            "correct":                 d["correct"],
            "n":                       n_tasks,
            "mean_latency_s":          round(statistics.mean(d["lat"]), 3),
            "p95_latency_s":           round(p95, 3),
            "cost_per_1k_tokens_usd":  round(statistics.mean(d["cost_per_1k"]), 5),
            "total_run_cost_usd":      round(sum(d["req_cost"]), 5),
        })
    leaderboard.sort(key=lambda r: r["accuracy"], reverse=True)
    return leaderboard


def _run_mock(tasks, models, task_label, first_scorer, task_file):
    mid_key   = "mock_id"
    per_model = {m[mid_key]: {"lat": [], "cost_per_1k": [], "req_cost": [],
                               "score": 0.0, "correct": 0} for m in models}
    endpoints = {m[mid_key]: MockEndpoint(m) for m in models}
    embed     = _make_embed()

    samples = []
    for task in tasks:
        scorer_name = task.get("scorer", "programmatic")
        row = {"q": task["input"], "expected": task_expected(task),
               "scorer": scorer_name, "answers": {}}

        for m in models:
            mid = m[mid_key]
            out = endpoints[mid].generate(task, seed=f"{mid}:{task['id']}")
            raw, lat, out_tokens = out["raw"], out["latency_s"], out["out_tokens"]

            if scorer_name == "llm_judge":
                jmean  = m["mock"].get("judge_score_mean", 0.70)
                jsd    = m["mock"].get("judge_score_sd",   0.10)
                s_rng  = random.Random(f"judge:{mid}:{task['id']}")
                s      = round(max(0.0, min(1.0, s_rng.gauss(jmean, jsd))), 4)
                detail = {"judge_score": round(s, 3)}
            else:
                s, detail = scoring.score(raw, task, embed=embed)

            req_cost = m["rate_hr"] / 3600.0 * lat
            c1k      = cost_per_1k_tokens(req_cost, out_tokens)

            per_model[mid]["lat"].append(lat)
            per_model[mid]["cost_per_1k"].append(c1k)
            per_model[mid]["req_cost"].append(req_cost)
            per_model[mid]["score"]   += s
            per_model[mid]["correct"] += int(s >= 0.5)

            row["answers"][mid] = {
                "text":      raw,
                "correct":   s >= 0.5,
                "score":     round(s, 3),
                "latency_s": round(lat, 3),
                **detail,
            }
        samples.append(row)

    n = len(tasks)
    return {
        "meta": {
            "mode":      "local-dry-run",
            "task":      task_file or "mixed",
            "task_name": task_label,
            "scorer":    first_scorer,
            "n_models":  len(models),
            "n_tasks":   n,
            "benchmark": task_label,
        },
        "leaderboard": _build_leaderboard(models, per_model, mid_key, n),
        "samples":     samples,
    }


def _run_tokenfactory(tasks, models, task_label, first_scorer, task_file):
    """Token Factory hosted API — no Serverless endpoints or jobs created."""
    import openai

    base_url = os.environ["NEBIUS_BASE_URL"]
    api_key  = os.environ["NEBIUS_API_KEY"]

    tf_models = [m for m in models if m.get("tokenfactory_ok")]
    if not tf_models:
        raise ValueError(
            "No tokenfactory_ok models in current selection. "
            "Add a model with tokenfactory_ok: true to config/models.yaml "
            "(e.g. meta-llama/Llama-3.3-70B-Instruct)."
        )

    client  = openai.OpenAI(base_url=base_url, api_key=api_key)
    mid_key = "id"
    embed   = _make_embed()

    per_model = {m[mid_key]: {"lat": [], "out_tokens": [], "score": 0.0, "correct": 0}
                 for m in tf_models}

    samples = []
    for task in tasks:
        scorer_name = task.get("scorer", "programmatic")
        row = {"q": task["input"], "expected": task_expected(task),
               "scorer": scorer_name, "answers": {}}

        for m in tf_models:
            mid  = m[mid_key]
            msgs = []
            if task.get("instruction"):
                msgs.append({"role": "system", "content": task["instruction"]})
            msgs.append({"role": "user", "content": task["input"]})

            t0   = time.time()
            resp = client.chat.completions.create(
                model=mid, messages=msgs, temperature=0
            )
            lat        = time.time() - t0
            raw        = resp.choices[0].message.content
            out_tokens = resp.usage.completion_tokens

            s, detail = scoring.score(raw, task, embed=embed)

            per_model[mid]["lat"].append(lat)
            per_model[mid]["out_tokens"].append(out_tokens)
            per_model[mid]["score"]   += s
            per_model[mid]["correct"] += int(s >= 0.5)

            row["answers"][mid] = {
                "text":       raw,
                "correct":    s >= 0.5,
                "score":      round(s, 3),
                "latency_s":  round(lat, 3),
                "out_tokens": out_tokens,
                **detail,
            }
        samples.append(row)

    n = len(tasks)
    leaderboard = []
    for m in tf_models:
        mid = m[mid_key]
        d   = per_model[mid]
        lat_sorted = sorted(d["lat"])
        p95 = lat_sorted[min(n - 1, int(round(0.95 * (n - 1))))]
        leaderboard.append({
            "model":                  mid,
            "preset":                 f"{m['preset']} / {m['instance_type']}",
            "accuracy":               round(d["score"] / n, 4),
            "correct":                d["correct"],
            "n":                      n,
            "mean_latency_s":         round(statistics.mean(d["lat"]), 3),
            "p95_latency_s":          round(p95, 3),
            "total_out_tokens":       sum(d["out_tokens"]),
            # TF bills per token; per-GPU-hour serving cost does not apply
            "cost_per_1k_tokens_usd": None,
            "total_run_cost_usd":     None,
        })
    leaderboard.sort(key=lambda r: r["accuracy"], reverse=True)

    return {
        "meta": {
            "mode":      "tokenfactory",
            "task":      task_file or "mixed",
            "task_name": task_label,
            "scorer":    first_scorer,
            "n_models":  len(tf_models),
            "n_tasks":   n,
            "benchmark": task_label,
        },
        "leaderboard": leaderboard,
        "samples":     samples,
    }


def _run_local(tasks, task_label, first_scorer, task_file):
    """Local vLLM server — OpenAI-compatible, no endpoint creation, no cost tracking.

    Env vars:
      LOCAL_BASE_URL  (default http://localhost:8000/v1)
      LOCAL_MODEL     (default Qwen/Qwen2.5-0.5B-Instruct)

    Pass threshold: embedding cosine ≥ 0.70 is a correct answer for reference_match tasks
    (the harness marks correct when score ≥ 0.5, which is conservative enough for cosine).
    """
    import openai

    base_url = os.environ.get("LOCAL_BASE_URL", "http://localhost:8000/v1")
    model_id = os.environ.get("LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    embed    = _make_embed()

    client    = openai.OpenAI(base_url=base_url, api_key="local")
    mid_key   = "id"
    per_model = {model_id: {"lat": [], "out_tokens": [], "score": 0.0, "correct": 0}}

    samples = []
    for task in tasks:
        scorer_name = task.get("scorer", "programmatic")
        row = {"q": task["input"], "expected": task_expected(task),
               "scorer": scorer_name, "answers": {}}

        msgs = []
        if task.get("instruction"):
            msgs.append({"role": "system", "content": task["instruction"]})
        msgs.append({"role": "user", "content": task["input"]})

        t0         = time.time()
        resp       = client.chat.completions.create(model=model_id, messages=msgs, temperature=0)
        lat        = time.time() - t0
        raw        = resp.choices[0].message.content
        out_tokens = resp.usage.completion_tokens

        s, detail = scoring.score(raw, task, embed=embed)

        per_model[model_id]["lat"].append(lat)
        per_model[model_id]["out_tokens"].append(out_tokens)
        per_model[model_id]["score"]   += s
        per_model[model_id]["correct"] += int(s >= 0.5)

        row["answers"][model_id] = {
            "text":       raw,
            "correct":    s >= 0.5,
            "score":      round(s, 3),
            "latency_s":  round(lat, 3),
            "out_tokens": out_tokens,
            **detail,
        }
        samples.append(row)

    n = len(tasks)
    d = per_model[model_id]
    lat_sorted = sorted(d["lat"])
    p95        = lat_sorted[min(n - 1, int(round(0.95 * (n - 1))))]
    leaderboard = [{
        "model":                  model_id,
        "preset":                 "local / vLLM",
        "accuracy":               round(d["score"] / n, 4),
        "correct":                d["correct"],
        "n":                      n,
        "mean_latency_s":         round(statistics.mean(d["lat"]), 3),
        "p95_latency_s":          round(p95, 3),
        "total_out_tokens":       sum(d["out_tokens"]),
        "cost_per_1k_tokens_usd": None,
        "total_run_cost_usd":     None,
    }]

    return {
        "meta": {
            "mode":      "local",
            "task":      task_file or "mixed",
            "task_name": task_label,
            "scorer":    first_scorer,
            "n_models":  1,
            "n_tasks":   n,
            "benchmark": task_label,
        },
        "leaderboard": leaderboard,
        "samples":     samples,
    }


def _run_endpoint(tasks, models, task_label, first_scorer, task_file):
    """Self-hosted Nebius Serverless Endpoints — creates and deletes GPU VMs per model."""
    import openai
    from src import orchestrator, storage

    mid_key    = "id"
    subnet_id  = os.environ["NEBIUS_SUBNET_ID"]
    auth_token = secrets.token_hex(32)
    embed      = _make_embed()

    per_model = {m[mid_key]: {"lat": [], "cost_per_1k": [], "req_cost": [],
                               "score": 0.0, "correct": 0} for m in models}
    answers_by_task = {task["id"]: {} for task in tasks}
    scorer_by_task  = {task["id"]: task.get("scorer", "programmatic") for task in tasks}

    for m in models:
        mid = m[mid_key]
        print(f"\n[{mid}] creating endpoint...", flush=True)
        endpoint_id = orchestrator.create_endpoint(
            m["id"], m["preset"], m["instance_type"], subnet_id, auth_token
        )
        t_up = time.time()

        try:
            base_url = orchestrator.wait_ready(endpoint_id, auth_token)
            client   = openai.OpenAI(
                base_url=f"{base_url}/v1", api_key=auth_token
            )

            for task in tasks:
                msgs = []
                if task.get("instruction"):
                    msgs.append({"role": "system", "content": task["instruction"]})
                msgs.append({"role": "user", "content": task["input"]})

                t0   = time.time()
                resp = client.chat.completions.create(
                    model=m["id"], messages=msgs, temperature=0
                )
                lat        = time.time() - t0
                raw        = resp.choices[0].message.content
                out_tokens = resp.usage.completion_tokens

                s, detail = scoring.score(raw, task, embed=embed)

                per_model[mid]["lat"].append(lat)
                per_model[mid]["score"]   += s
                per_model[mid]["correct"] += int(s >= 0.5)

                answers_by_task[task["id"]][mid] = {
                    "text":      raw,
                    "correct":   s >= 0.5,
                    "score":     round(s, 3),
                    "latency_s": round(lat, 3),
                    "out_tokens": out_tokens,
                    **detail,
                }
                per_model[mid].setdefault("out_tokens_total", 0)
                per_model[mid]["out_tokens_total"] += out_tokens

        finally:
            orchestrator.delete_endpoint(endpoint_id)

        t_down        = time.time()
        serving_cost  = endpoint_cost(t_up, t_down, m["rate_hr"])
        total_tokens  = per_model[mid].get("out_tokens_total", 1)
        c1k           = cost_per_1k_tokens(serving_cost, total_tokens)

        per_model[mid]["req_cost"]    = [serving_cost]
        per_model[mid]["cost_per_1k"] = [c1k]

        print(f"  {mid}: serving cost ${serving_cost:.4f}, "
              f"${c1k:.5f}/1k tokens", flush=True)

    samples = []
    for task in tasks:
        samples.append({
            "q":        task["input"],
            "expected": task_expected(task),
            "scorer":   scorer_by_task[task["id"]],
            "answers":  answers_by_task[task["id"]],
        })

    n       = len(tasks)
    results = {
        "meta": {
            "mode":      "endpoint",
            "task":      task_file or "mixed",
            "task_name": task_label,
            "scorer":    first_scorer,
            "n_models":  len(models),
            "n_tasks":   n,
            "benchmark": task_label,
        },
        "leaderboard": _build_leaderboard(models, per_model, mid_key, n),
        "samples":     samples,
    }

    if os.environ.get("STORAGE_BUCKET") and os.environ.get("STORAGE_ENDPOINT"):
        try:
            storage.put_results(results)
        except Exception as e:
            print(f"  storage upload failed (non-fatal): {e}")

    return results
