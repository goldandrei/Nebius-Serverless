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
                judge_client, scale: int = 5) -> float:
    answer = _strip_think(answer)
    if judge_client is None:
        return 0.5  # stub: wire up a real judge endpoint in Phase 2+
    judge_prompt = (
        "You are grading an answer. Be strict and consistent.\n"
        f"Task: {instruction}\n"
        f"Input: {input_text}\n"
        f"Answer: {answer}\n"
        f"Rubric: {rubric}\n"
        f'Reply ONLY as JSON: {{"score": <1-{scale}>, "reason": "..."}}'
    )
    verdict = judge_client.chat(judge_prompt, temperature=0)
    obj = json.loads(_extract_json(verdict))
    return max(0.0, min(1.0, obj["score"] / scale))


# ── harness dispatcher ────────────────────────────────────────────────────────

def score_item(record: dict, answer: str, embed=None, judge_client=None) -> float:
    s = record["scorer"]
    if s == "programmatic":
        return score_programmatic(answer, record["gold"], record.get("compare", "exact"))
    if s == "reference_match":
        return score_reference(answer, record["reference"],
                               record.get("metric", "lexical"),
                               record.get("threshold", 0.6), embed)
    if s == "llm_judge":
        return score_judge(record["instruction"], record["input"], answer,
                           record["rubric"], judge_client, record.get("scale", 5))
    return 0.0


# ── backward-compat wrapper (returns (score, detail) for eval_runner) ─────────

def score(raw_output: str, task_item: dict, scorer_name: str = None, embed=None) -> tuple:
    name   = scorer_name or task_item.get("scorer", "programmatic")
    record = {**task_item, "scorer": name}
    s      = score_item(record, raw_output, embed=embed)
    return round(s, 4), _make_detail(name, raw_output, task_item, s)


def _make_detail(scorer: str, answer: str, task_item: dict, s: float) -> dict:
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
        return {"judge_score": round(s, 3), "note": "llm_judge stub"}
    return {}
