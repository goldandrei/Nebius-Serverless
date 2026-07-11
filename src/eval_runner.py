import json
import os
import secrets
import statistics
import time
from pathlib import Path

import yaml

from src import scoring

ROOT          = Path(__file__).resolve().parent.parent
DATA_DIR      = ROOT / "data"
CATALOG_FILE  = ROOT / "config" / "catalog.yaml"
SELECTION_FILE = ROOT / "config" / "models.yaml"

JUDGE_MODEL        = os.environ.get("JUDGE_MODEL", "deepseek-ai/DeepSeek-V4-Pro")
MAX_OUTPUT_TOKENS  = int(os.environ.get("MAX_OUTPUT_TOKENS", "2048"))  # bounds contestant output; reasoning models that exceed it get truncated, not hung


class JudgeError(Exception):
    """Raised when the LLM-judge API call fails after one retry."""


class JudgeClient:
    """Wraps a Token Factory chat model as an LLM judge; tracks usage for cost accounting.

    Shares the same base_url/api_key construction as the tokenfactory backend so judge
    calls bill against the same Nebius account, but is a separate client instance since
    a judge may be needed even when comparing only self-hosted (endpoint) models.
    """

    def __init__(self, client, model: str = JUDGE_MODEL):
        self.client     = client
        self.model      = model
        self.in_tokens  = 0
        self.out_tokens = 0
        self.calls      = 0
        self.errors     = 0

    def chat(self, prompt: str, temperature: float = 0) -> str:
        last_err = None
        for _ in range(2):  # one retry
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                self.calls      += 1
                self.in_tokens  += resp.usage.prompt_tokens
                self.out_tokens += resp.usage.completion_tokens
                return resp.choices[0].message.content
            except Exception as e:
                last_err = e
        raise JudgeError(f"judge call failed after retry: {last_err}") from last_err


def _validate_llm_judge_rubrics(tasks: list) -> None:
    for t in tasks:
        if t.get("scorer") == "llm_judge" and not str(t.get("rubric", "")).strip():
            raise ValueError(
                f"Task {t.get('id', '?')} has scorer=llm_judge but an empty rubric — "
                "a judge with no rubric is meaningless."
            )


# ── embedding ─────────────────────────────────────────────────────────────────

def _make_embed(api_key: str = None, base_url: str = None, model: str = None):
    api_key  = api_key  or os.environ.get("NEBIUS_API_KEY")
    base_url = base_url or os.environ.get("NEBIUS_BASE_URL",
                                          "https://api.tokenfactory.nebius.com/v1/")
    model    = model    or os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")

    if not api_key:
        raise RuntimeError(
            "NEBIUS_API_KEY is required for embedding scoring. "
            "Set it in .env or the environment."
        )

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
    "reference_match": "Reference match · embedding similarity",
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
    return [dict(catalog[mid]) for mid in selected_ids]


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


# ── main eval loop ────────────────────────────────────────────────────────────

def run(task_file: str = None, progress_cb=None, prices: dict = None) -> dict:
    """Run the eval harness with automatic per-model routing.

    Each model is routed by its catalog ``basis`` field:
      basis: self-hosted → Nebius Serverless Endpoint (creates/tears down a GPU VM per model)
      basis: hosted      → Token Factory API (no GPU infrastructure managed)

    A single comparison can mix both backends; results are merged into one leaderboard.
    progress_cb: optional callable(model, model_idx, n_models, item_idx, n_items)
    prices:      optional in-memory price dict populated at server startup
    """
    tasks  = load_tasks(task_file=task_file)
    models = load_config()

    _validate_llm_judge_rubrics(tasks)

    first_scorer = tasks[0].get("scorer", "programmatic") if tasks else "programmatic"
    task_meta    = {t["id"]: t for t in available_tasks()}
    task_label   = task_meta.get(task_file, {}).get("name", task_file or "mixed") if task_file else "mixed"

    effective_prices = prices if prices is not None else _load_prices_from_file()

    # Route each model to its backend by basis field
    ep_models = [m for m in models if m.get("basis", "self-hosted") == "self-hosted"]
    tf_models = [m for m in models if m.get("basis", "self-hosted") == "hosted"]

    routing = {
        "endpoint":     [m["id"] for m in ep_models],
        "tokenfactory": [m["id"] for m in tf_models],
    }
    print(
        f"  routing: endpoint={routing['endpoint']}  tokenfactory={routing['tokenfactory']}",
        flush=True,
    )

    # Lazy judge client — only pay for it when a selected task actually needs grading.
    judge_client = None
    if any(t.get("scorer") == "llm_judge" for t in tasks):
        import openai
        judge_client = JudgeClient(openai.OpenAI(
            base_url=os.environ["NEBIUS_BASE_URL"], api_key=os.environ["NEBIUS_API_KEY"],
        ))

    results_tf = None
    results_ep = None

    if tf_models:
        results_tf = _run_tokenfactory(
            tasks, tf_models, task_label, first_scorer, task_file,
            progress_cb, effective_prices, judge_client=judge_client,
        )
    if ep_models:
        results_ep = _run_endpoint(tasks, ep_models, task_label, first_scorer, task_file,
                                   progress_cb=progress_cb, judge_client=judge_client)

    # Merge leaderboards — sort by accuracy descending
    leaderboard: list = []
    if results_tf:
        leaderboard.extend(results_tf["leaderboard"])
    if results_ep:
        leaderboard.extend(results_ep["leaderboard"])
    leaderboard.sort(key=lambda r: r["accuracy"], reverse=True)

    # Merge samples — same question may have answers from both backends
    samples_map: dict = {}
    for result in filter(None, [results_tf, results_ep]):
        for s in result.get("samples", []):
            key = s["q"]
            if key not in samples_map:
                samples_map[key] = {k: v for k, v in s.items() if k != "answers"}
                samples_map[key]["answers"] = {}
            samples_map[key]["answers"].update(s.get("answers", {}))

    judge_meta = {
        "judge_model":      None,
        "judge_calls":      0,
        "judge_in_tokens":  0,
        "judge_out_tokens": 0,
        "judge_cost_usd":   None,
        "judge_errors":     0,
    }
    if judge_client is not None:
        p     = effective_prices.get(JUDGE_MODEL, {})
        p_in  = p.get("price_in_per_1m")
        p_out = p.get("price_out_per_1m")
        if p_in is not None and p_out is not None:
            judge_cost = (judge_client.in_tokens * p_in + judge_client.out_tokens * p_out) / 1_000_000
        else:
            judge_cost = None
            print(f"  [judge] TODO: no price for {JUDGE_MODEL} in prices.yaml — judge_cost_usd will be null",
                  flush=True)
        judge_meta = {
            "judge_model":      JUDGE_MODEL,
            "judge_calls":      judge_client.calls,
            "judge_in_tokens":  judge_client.in_tokens,
            "judge_out_tokens": judge_client.out_tokens,
            "judge_cost_usd":   round(judge_cost, 6) if judge_cost is not None else None,
            "judge_errors":     judge_client.errors,
        }

    return {
        "meta": {
            "mode":       "auto",
            "task":       task_file or "mixed",
            "task_name":  task_label,
            "scorer":     first_scorer,
            "n_models":   len(leaderboard),
            "n_tasks":    len(tasks),
            "benchmark":  task_label,
            "routing":    routing,
            **judge_meta,
        },
        "leaderboard": leaderboard,
        "samples":     list(samples_map.values()),
    }


def _build_leaderboard(models, per_model, mid_key, n_tasks):
    leaderboard = []
    for m in models:
        mid = m[mid_key]
        d   = per_model[mid]
        lat_sorted = sorted(d["lat"])
        p95 = lat_sorted[min(n_tasks - 1, int(round(0.95 * (n_tasks - 1))))]
        gpu   = m.get("endpoint_gpu_type", m.get("preset", ""))
        inst  = m.get("instance_type", f"{m.get('endpoint_gpu_count', 1)}×GPU")
        score_n = d.get("score_n", n_tasks)
        row = {
            "model":                   mid,
            "preset":                  f"{gpu} / {inst}",
            "cost_basis":              "self-hosted",
            "accuracy":                round(d["score"] / score_n, 4) if score_n else 0.0,
            "correct":                 d["correct"],
            "n":                       n_tasks,
            "mean_latency_s":          round(statistics.mean(d["lat"]), 3),
            "p95_latency_s":           round(p95, 3),
            "total_in_tokens":         d.get("in_tokens_total", 0),
            "total_out_tokens":        d.get("out_tokens_total", 0),
            "rate_hr":                 d.get("rate_hr"),
            "cost_per_1k_tokens_usd":  round(statistics.mean(d["cost_per_1k"]), 5),
            "total_run_cost_usd":      round(sum(d["req_cost"]), 5),
        }
        # Include deploy/eval cost split when timestamps were recorded
        if "deploy_cost" in d:
            row.update({
                "t_created":          d["t_created"],
                "t_ready":            d["t_ready"],
                "t_eval_done":        d["t_eval_done"],
                "inference_s":        round(d.get("inference_s", 0), 3),
                "deploy_cost_usd":    round(d["deploy_cost"], 6),
                "eval_cost_usd":      round(d["eval_cost"], 6),
            })
        leaderboard.append(row)
    leaderboard.sort(key=lambda r: r["accuracy"], reverse=True)
    return leaderboard


def _load_prices_from_file(prices_file: Path = None) -> dict:
    """Read prices.yaml and return the tokenfactory section. Returns {} on any failure."""
    f = prices_file or (ROOT / "config" / "prices.yaml")
    if not f.exists():
        return {}
    try:
        with open(f, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data.get("tokenfactory", {}) or {}
    except Exception:
        return {}


def _run_tokenfactory(tasks, models, task_label, first_scorer, task_file,
                      progress_cb=None, prices: dict = None, judge_client=None):
    """Nebius Token Factory hosted API — no Serverless endpoints or jobs created."""
    import openai

    base_url = os.environ["NEBIUS_BASE_URL"]
    api_key  = os.environ["NEBIUS_API_KEY"]

    if not models:
        raise ValueError("_run_tokenfactory called with no models")

    client  = openai.OpenAI(base_url=base_url, api_key=api_key)
    mid_key = "id"
    embed   = _make_embed()
    prices  = prices or {}

    per_model = {m[mid_key]: {"lat": [], "in_tokens": [], "out_tokens": [],
                              "score": 0.0, "score_n": 0, "correct": 0}
                 for m in models}

    n = len(tasks)
    samples = []
    for j, task in enumerate(tasks):
        scorer_name = task.get("scorer", "programmatic")
        row = {"q": task["input"], "expected": task_expected(task),
               "scorer": scorer_name, "compare": task.get("compare"),
               "metric": task.get("metric"), "answers": {}}

        for i, m in enumerate(models):
            mid  = m[mid_key]
            if progress_cb:
                progress_cb(model=mid, model_idx=i, n_models=len(models),
                            item_idx=j + 1, n_items=n)
            msgs = []
            if task.get("instruction"):
                msgs.append({"role": "system", "content": task["instruction"]})
            msgs.append({"role": "user", "content": task["input"]})

            t0   = time.time()
            resp = client.chat.completions.create(
                model=mid, messages=msgs, temperature=0, max_tokens=MAX_OUTPUT_TOKENS
            )
            lat        = time.time() - t0
            raw        = resp.choices[0].message.content
            in_tokens  = resp.usage.prompt_tokens
            out_tokens = resp.usage.completion_tokens

            s, detail = scoring.score(raw, task, embed=embed, judge_client=judge_client)

            per_model[mid]["lat"].append(lat)
            per_model[mid]["in_tokens"].append(in_tokens)
            per_model[mid]["out_tokens"].append(out_tokens)
            if s is not None:
                per_model[mid]["score"]   += s
                per_model[mid]["score_n"] += 1
                per_model[mid]["correct"] += int(s >= 0.5)

            if scorer_name == "llm_judge" and mid == JUDGE_MODEL:
                detail = {**detail, "self_graded": True}

            row["answers"][mid] = {
                "text":       raw,
                "correct":    s is not None and s >= 0.5,
                "score":      round(s, 3) if s is not None else None,
                "latency_s":  round(lat, 3),
                "in_tokens":  in_tokens,
                "out_tokens": out_tokens,
                **detail,
            }
        samples.append(row)

    leaderboard = []
    for m in models:
        mid = m[mid_key]
        d   = per_model[mid]
        lat_sorted = sorted(d["lat"])
        p95 = lat_sorted[min(n - 1, int(round(0.95 * (n - 1))))]

        total_in  = sum(d["in_tokens"])
        total_out = sum(d["out_tokens"])
        total_tok = total_in + total_out
        p = prices.get(mid, {})
        p_in  = p.get("price_in_per_1m")
        p_out = p.get("price_out_per_1m")
        if p_in is not None and p_out is not None and total_tok > 0:
            run_cost = (total_in * p_in + total_out * p_out) / 1_000_000
            c1k      = run_cost / total_tok * 1000
        else:
            run_cost = None
            c1k      = None

        score_n = d["score_n"]
        leaderboard.append({
            "model":                  mid,
            "preset":                 f"{m['preset']} / {m['instance_type']}",
            "cost_basis":             "hosted",
            "accuracy":               round(d["score"] / score_n, 4) if score_n else 0.0,
            "correct":                d["correct"],
            "n":                      n,
            "mean_latency_s":         round(statistics.mean(d["lat"]), 3),
            "p95_latency_s":          round(p95, 3),
            "total_in_tokens":        total_in,
            "total_out_tokens":       total_out,
            "price_in_per_1m":        p_in,
            "price_out_per_1m":       p_out,
            "cost_per_1k_tokens_usd": round(c1k, 6) if c1k is not None else None,
            "total_run_cost_usd":     round(run_cost, 6) if run_cost is not None else None,
        })
    leaderboard.sort(key=lambda r: r["accuracy"], reverse=True)

    return {
        "meta": {
            "mode":      "tokenfactory",
            "task":      task_file or "mixed",
            "task_name": task_label,
            "scorer":    first_scorer,
            "n_models":  len(models),
            "n_tasks":   n,
            "benchmark": task_label,
        },
        "leaderboard": leaderboard,
        "samples":     samples,
    }


def _run_endpoint(tasks, models, task_label, first_scorer, task_file, progress_cb=None,
                  judge_client=None):
    """Nebius Serverless AI Endpoints — creates and deletes GPU VMs per model."""
    import openai
    from src import orchestrator, storage

    mid_key   = "id"
    hf_token  = os.environ.get("HF_TOKEN", "").strip()
    # Real HF tokens are "hf_" + ≥20 alphanumeric chars; shorter values are placeholders.
    hf_valid  = hf_token.startswith("hf_") and len(hf_token) > 20

    # Gated pre-check: fail fast before spending any GPU time or API credits.
    for m in models:
        if m.get("gated") and not hf_valid:
            short = m[mid_key].split("/")[-1]
            hf_id = m[mid_key]
            if hf_token:
                msg = (
                    f"⚠ {short} is a gated model but HF_TOKEN looks like a placeholder "
                    f"('{hf_token[:12]}…'). Replace it with a real token from "
                    f"huggingface.co/settings/tokens. Also accept the model license at "
                    f"huggingface.co/{hf_id} before deploying."
                )
            else:
                msg = (
                    f"⚠ {short} is a gated model. To self-host it you must: "
                    f"(1) accept its license on huggingface.co/{hf_id}, and "
                    f"(2) set HF_TOKEN in your .env. "
                    f"Or choose an ungated model, or use the hosted Token Factory version if available."
                )
            raise RuntimeError(msg)

    embed    = _make_embed()

    per_model = {m[mid_key]: {"lat": [], "cost_per_1k": [], "req_cost": [],
                               "score": 0.0, "score_n": 0, "correct": 0,
                               "in_tokens_total": 0, "out_tokens_total": 0} for m in models}
    answers_by_task = {task["id"]: {} for task in tasks}
    scorer_by_task  = {task["id"]: task.get("scorer", "programmatic") for task in tasks}

    for model_idx, m in enumerate(models):
        mid                  = m[mid_key]
        platform             = m.get("preset", "gpu-l40s-a")
        preset               = m.get("instance_type", "1gpu-8vcpu-32gb")
        tensor_parallel_size  = m.get("tensor_parallel_size", 1)
        max_model_len         = m.get("max_model_len", 8192)
        load_timeout_s        = m.get("load_timeout_s", 480)
        provision_timeout_s   = m.get("provision_timeout_s", 900)
        extra_vllm_args       = m.get("extra_vllm_args", "")
        rate_hr               = m.get("rate_hr", 1.82)
        auth_token            = secrets.token_hex(32)

        tp_note = f", TP={tensor_parallel_size}" if tensor_parallel_size > 1 else ""
        ea_note = f", {extra_vllm_args}" if extra_vllm_args else ""
        print(f"\n[{mid}] creating endpoint ({platform}/{preset}{tp_note}, ctx={max_model_len}{ea_note})...",
              flush=True)
        if progress_cb:
            progress_cb(ep_model=mid, ep_state="PROVISIONING", ep_elapsed_s=0,
                        n_items=0, item_idx=0)
        t_created = time.time()
        try:
            endpoint_id = orchestrator.create_endpoint(mid, platform, preset, auth_token,
                                                        tensor_parallel_size=tensor_parallel_size,
                                                        max_model_len=max_model_len,
                                                        extra_vllm_args=extra_vllm_args)
        except RuntimeError as e:
            short = mid.split("/")[-1]
            raise RuntimeError(f"Endpoint deployment failed for {short}: {e}") from None

        # Wrap progress_cb to also inject ep_model so the dashboard knows which model
        def _ep_progress_cb(**kw):
            if progress_cb:
                progress_cb(ep_model=mid, **kw)

        try:
            base_url = orchestrator.wait_ready(endpoint_id, auth_token,
                                               progress_cb=_ep_progress_cb,
                                               load_timeout_s=load_timeout_s,
                                               provision_timeout_s=provision_timeout_s)
            t_ready  = time.time()

            client = openai.OpenAI(
                base_url=f"{base_url}/v1", api_key=auth_token
            )

            n_tasks = len(tasks)
            n_models_total = len(models)
            if progress_cb:
                progress_cb(ep_model=mid, ep_state="evaluating",
                            ep_elapsed_s=time.time() - t_created,
                            n_items=n_tasks, item_idx=0,
                            n_models=n_models_total, model=mid, model_idx=model_idx)

            for task_idx, task in enumerate(tasks):
                msgs = []
                if task.get("instruction"):
                    msgs.append({"role": "system", "content": task["instruction"]})
                msgs.append({"role": "user", "content": task["input"]})

                t0   = time.time()
                resp = client.chat.completions.create(
                    model=mid, messages=msgs, temperature=0, max_tokens=MAX_OUTPUT_TOKENS
                )
                lat        = time.time() - t0
                raw        = resp.choices[0].message.content
                in_tokens  = resp.usage.prompt_tokens
                out_tokens = resp.usage.completion_tokens

                s, detail = scoring.score(raw, task, embed=embed, judge_client=judge_client)

                per_model[mid]["lat"].append(lat)
                if s is not None:
                    per_model[mid]["score"]   += s
                    per_model[mid]["score_n"] += 1
                    per_model[mid]["correct"] += int(s >= 0.5)
                per_model[mid]["in_tokens_total"]  += in_tokens
                per_model[mid]["out_tokens_total"] += out_tokens

                if scorer_by_task[task["id"]] == "llm_judge" and mid == JUDGE_MODEL:
                    detail = {**detail, "self_graded": True}

                answers_by_task[task["id"]][mid] = {
                    "text":       raw,
                    "correct":    s is not None and s >= 0.5,
                    "score":      round(s, 3) if s is not None else None,
                    "latency_s":  round(lat, 3),
                    "in_tokens":  in_tokens,
                    "out_tokens": out_tokens,
                    **detail,
                }

                if progress_cb:
                    progress_cb(ep_model=mid, ep_state="evaluating",
                                ep_elapsed_s=time.time() - t_created,
                                n_items=n_tasks, item_idx=task_idx + 1,
                                n_models=n_models_total, model=mid, model_idx=model_idx)

        finally:
            if progress_cb:
                progress_cb(ep_model=mid, ep_state="deleting",
                            ep_elapsed_s=time.time() - t_created,
                            n_items=0, item_idx=0)
            try:
                orchestrator.delete_endpoint(endpoint_id)
            except Exception as _del_err:
                print(f"    [warn] delete {endpoint_id} failed: {_del_err}", flush=True)

        t_eval_done  = time.time()

        # inference_s = sum of per-item vLLM request times only.
        # t_eval_done - t_ready includes embedding-scoring API calls after each
        # inference call, inflating the apparent GPU-time cost. Use inference_s
        # for eval_cost and $/1K tok so those numbers reflect model serving, not
        # scoring overhead. total_cost = full uptime billed (honest billing total).
        inference_s  = sum(per_model[mid]["lat"])
        deploy_cost  = (t_ready - t_created) / 3600 * rate_hr
        eval_cost    = inference_s / 3600 * rate_hr
        total_cost   = (t_eval_done - t_created) / 3600 * rate_hr
        total_out    = per_model[mid]["out_tokens_total"] or 1
        c1k          = eval_cost / total_out * 1000

        per_model[mid].update({
            "rate_hr":      rate_hr,
            "t_created":    t_created,
            "t_ready":      t_ready,
            "t_eval_done":  t_eval_done,
            "inference_s":  inference_s,
            "deploy_cost":  deploy_cost,
            "eval_cost":    eval_cost,
            "total_cost":   total_cost,
            "req_cost":     [total_cost],
            "cost_per_1k":  [c1k],
        })

        deploy_min = (t_ready - t_created) / 60
        print(
            f"  {mid}: deploy {deploy_min:.1f} min (${deploy_cost:.4f})  "
            f"inference {inference_s:.1f}s (${eval_cost:.6f})  "
            f"total ${total_cost:.4f}  ${c1k:.5f}/1k tok (inference-only)",
            flush=True,
        )

    samples = []
    for task in tasks:
        samples.append({
            "q":        task["input"],
            "expected": task_expected(task),
            "scorer":   scorer_by_task[task["id"]],
            "compare":  task.get("compare"),
            "metric":   task.get("metric"),
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
