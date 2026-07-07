import json
import re


# ── helpers ───────────────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    return re.sub(r"[.,!?;:\s]+$", "", str(s).strip().lower())


def extract_number(s: str):
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def field_f1(gold: dict, pred: dict) -> float:
    """F1 over key:value pairs — a key counts only if present AND normalized values equal."""
    if not gold and not pred:
        return 1.0
    matches = sum(
        1 for k, gv in gold.items()
        if k in pred and normalize(str(pred[k])) == normalize(str(gv))
    )
    precision = matches / len(pred) if pred else 0.0
    recall    = matches / len(gold) if gold else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def token_f1(a: str, b: str) -> float:
    """F1 over word sets."""
    a_toks = set(a.split())
    b_toks = set(b.split())
    if not a_toks and not b_toks:
        return 1.0
    common = len(a_toks & b_toks)
    if common == 0:
        return 0.0
    p = common / len(a_toks)
    r = common / len(b_toks)
    return 2 * p * r / (p + r)


def _strip_think(text: str) -> str:
    """Remove <think>…</think> blocks emitted by reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text: str) -> str:
    """Strip think blocks + fences, return the last {...} span.

    Using the last block (not first-to-last greedy) means a reasoning model
    that writes example JSON inside its think block, then emits the real
    answer afterward, still scores on the correct final object.
    """
    text = _strip_think(text)
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    # Find all {...} blocks (handles one level of nesting, enough for flat schemas)
    blocks = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if blocks:
        return blocks[-1]
    # Fallback: greedy span (last resort — avoids returning bare text)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return m.group() if m else text


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


# ── scorers ───────────────────────────────────────────────────────────────────

def score_programmatic(answer: str, gold, compare: str = "exact") -> float:
    answer = _strip_think(answer)
    a = normalize(answer)
    if compare == "exact":
        return 1.0 if a == normalize(str(gold)) else 0.0
    if compare == "numeric":
        num = extract_number(answer)
        try:
            return 1.0 if num is not None and num == float(gold) else 0.0
        except (TypeError, ValueError):
            return 0.0
    if compare == "json_fields":
        try:
            pred = json.loads(_extract_json(answer))
        except Exception:
            return 0.0
        return field_f1(gold, pred)
    return 0.0


def score_reference(answer: str, reference: str, metric: str = "lexical",
                    threshold: float = 0.6, embed=None) -> float:
    answer = _strip_think(answer)
    if metric == "lexical":
        return token_f1(normalize(answer), normalize(reference))
    # "embedding" (optional/stretch — needs embed callable)
    return _cosine(embed(answer), embed(reference))


def score_judge(instruction: str, input_text: str, answer: str, rubric: str,
                judge_client, scale: int = 5) -> tuple:
    """Grade one answer against a rubric. Returns (score_0_1, reason, error).

    On success: (float in [0,1], reason string, None).
    On failure (judge call/parse/validation error): (None, None, error string) —
    the caller must exclude this item from the model's mean, not count it as 0.
    """
    answer = _strip_think(answer)
    if judge_client is None:
        raise RuntimeError("llm_judge scoring requires a judge_client")
    judge_prompt = (
        "You are grading an answer. Be strict and consistent.\n"
        f"Task: {instruction}\n"
        f"Input: {input_text}\n"
        f"Answer: {answer}\n"
        f"Rubric: {rubric}\n"
        f'Reply ONLY as JSON: {{"score": <1-{scale}>, "reason": "..."}}'
    )
    try:
        verdict   = judge_client.chat(judge_prompt, temperature=0)
        obj       = json.loads(_extract_json(verdict))
        raw_score = obj["score"]
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise ValueError(f"non-numeric score: {raw_score!r}")
        if not (1 <= raw_score <= scale):
            raise ValueError(f"score {raw_score} out of range [1,{scale}]")
        reason = str(obj.get("reason", ""))
    except Exception as e:
        judge_client.errors += 1
        return None, None, f"{type(e).__name__}: {e}"
    norm = 1.0 if scale <= 1 else (raw_score - 1) / (scale - 1)
    return max(0.0, min(1.0, norm)), reason, None


# ── harness dispatcher ────────────────────────────────────────────────────────

def score_item(record: dict, answer: str, embed=None, judge_client=None) -> tuple:
    """Returns (score_or_None, reason_or_None, error_or_None)."""
    s = record["scorer"]
    if s == "programmatic":
        return score_programmatic(answer, record["gold"], record.get("compare", "exact")), None, None
    if s == "reference_match":
        return score_reference(answer, record["reference"],
                               record.get("metric", "lexical"),
                               record.get("threshold", 0.6), embed), None, None
    if s == "llm_judge":
        return score_judge(record["instruction"], record["input"], answer,
                           record["rubric"], judge_client, record.get("scale", 5))
    return 0.0, None, None


# ── backward-compat wrapper (returns (score, detail) for eval_runner) ─────────

def score(raw_output: str, task_item: dict, scorer_name: str = None, embed=None,
         judge_client=None) -> tuple:
    name             = scorer_name or task_item.get("scorer", "programmatic")
    record           = {**task_item, "scorer": name}
    s, reason, error = score_item(record, raw_output, embed=embed, judge_client=judge_client)
    detail           = _make_detail(name, raw_output, task_item, s, reason, error)
    return (round(s, 4) if s is not None else None), detail


def _make_detail(scorer: str, answer: str, task_item: dict, s, reason: str = None,
                 error: str = None) -> dict:
    if scorer == "programmatic":
        compare = task_item.get("compare", "exact")
        if compare == "json_fields":
            try:
                json.loads(_extract_json(answer))
                return {"valid_json": True, "field_f1": round(s, 3)}
            except Exception:
                return {"valid_json": False, "field_f1": 0.0}
        return {"match": s == 1.0}
    if scorer == "reference_match":
        return {"sim": round(s, 3)}
    if scorer == "llm_judge":
        if error:
            return {"judge_error": error}
        return {"judge_score": round(s, 3), "reason": reason}
    return {}
