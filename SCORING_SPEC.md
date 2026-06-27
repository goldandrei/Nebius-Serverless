# Scoring Spec — three evaluation methods

> Reference for building `src/scoring.py`. The **user explicitly picks one method**
> and provides plain pieces (an instruction, a list of inputs, and answers/rubric).
> The tool assembles the internal JSONL — the user never writes JSONL by hand.

## Shared contract (applies to all three)

- The prompt sent to a model = `instruction` + one `input`. The user writes the
  instruction **once**; the inputs are the varying part.
- Every scorer takes the model's returned `answer` and returns a **float in [0.0, 1.0]**.
- Per model, the leaderboard number = **mean score over all items**.
- The tool builds each internal record from the user's uploaded pieces by zipping
  `inputs` with `answers` (or attaching the single shared `rubric`) and stamping the
  chosen `scorer` + options.

**Build priority:** `programmatic` and `reference_match (lexical)` are **core** — pure
code, no extra model. `reference_match (embedding)` and `llm_judge` need a second model
running and are **optional/stretch**.

---

## Method 1 — Programmatic

**Explain to the user:** "Use this when each answer has one correct value you can check
exactly — a label, a number, or a structured object like JSON. The tool checks the
model's answer against your expected answer with code. Instant, free, exact."

**Small example:**
- Instruction: *"Classify the sentiment as positive, negative, or neutral. Reply with one word."*
- Input: *"The food was cold and the service was slow."*
- Expected answer: *"negative"*
- Model replies `"Negative."` → normalized → matches → **score 1.0**. Replies `"positive"` → **0.0**.

**What the user provides:**
- one task **instruction**
- a list of **inputs**
- a list of **expected answers** (aligned 1:1 with inputs)
- *(optional)* compare mode: `exact` (default) · `numeric` · `json_fields`

**Internal record the tool assembles:**
```json
{"id": 1, "instruction": "...", "input": "The food was cold...", "gold": "negative", "scorer": "programmatic", "compare": "exact"}
```

**Evaluation logic to build** (runs after the model returns `answer`):
```python
def score_programmatic(answer, gold, compare="exact"):
    a = normalize(answer)                 # strip, lowercase, drop trailing punctuation
    if compare == "exact":
        return 1.0 if a == normalize(str(gold)) else 0.0
    if compare == "numeric":
        return 1.0 if extract_number(answer) == float(gold) else 0.0
    if compare == "json_fields":
        try:
            pred = json.loads(answer)     # invalid JSON -> 0.0
        except Exception:
            return 0.0
        return field_f1(gold, pred)       # partial credit: matching key:value pairs
```
Helpers: `normalize()`, `extract_number()` (regex first number), `field_f1(gold, pred)`
(a key counts only if key present AND normalized values equal; F1 of matched pairs).

---

## Method 2 — Reference-match

**Explain to the user:** "Use this when there's a right answer but it can be worded many
ways — a question's answer, a short summary. The tool checks how *close* the model's
answer is to your reference answer."

**Small example:**
- Instruction: *"Answer the question in one sentence."*
- Input: *"What's the capital of Australia?"*
- Reference answer: *"The capital of Australia is Canberra."*
- Model replies `"Canberra is Australia's capital."` → high overlap → **pass / ~0.8**.

**What the user provides:**
- one task **instruction**
- a list of **inputs**
- a list of **reference answers** (aligned 1:1)
- *(optional)* `metric`: `lexical` (default, pure code) · `embedding` (needs a small model)
- *(optional)* `threshold` (default 0.6) if you want pass/fail instead of a continuous score

**Internal record the tool assembles:**
```json
{"id": 1, "instruction": "...", "input": "What's the capital of Australia?", "reference": "The capital of Australia is Canberra.", "scorer": "reference_match", "metric": "lexical", "threshold": 0.6}
```

**Evaluation logic to build:**
```python
def score_reference(answer, reference, metric="lexical", threshold=0.6, embed=None):
    if metric == "lexical":
        sim = token_f1(normalize(answer), normalize(reference))   # word-set overlap, no model
    else:  # "embedding" (optional/stretch)
        sim = cosine(embed(answer), embed(reference))             # needs embedding model
    return sim                       # continuous 0..1
    # for pass/fail instead: return 1.0 if sim >= threshold else 0.0
```
`token_f1` = F1 over the two word sets. `embed` is an injected callable (an embedding
endpoint / Token Factory) — only wired up if `metric == "embedding"`.

---

## Method 3 — LLM-judge  *(optional/stretch — needs a judge model)*

**Explain to the user:** "Use this when there's no single right answer — writing quality,
tone, helpfulness. A second 'judge' model reads each answer and grades it against your
rubric."

**Small example:**
- Instruction: *"Write a friendly one-line apology for a late delivery."*
- Input: *"Order #123 arrived 3 days late."*
- Rubric: *"Score 1-5: apologizes (2), friendly tone (2), one concise line (1)."*
- Model replies `"So sorry your order was late — we'll make it right!"` → judge returns 5 → **1.0**.

**What the user provides:**
- one task **instruction**
- a list of **inputs**
- one **rubric** (shared across all inputs)
- which model **judges** (default: a capable model on Nebius / Token Factory)
- *(no gold answers needed)*

**Internal record the tool assembles:**
```json
{"id": 1, "instruction": "...", "input": "Order #123 arrived 3 days late.", "rubric": "Score 1-5: apologizes (2)...", "scorer": "llm_judge", "judge_model": "...", "scale": 5}
```

**Evaluation logic to build** (runs after the model-under-test returns `answer`):
```python
def score_judge(instruction, input, answer, rubric, judge_client, scale=5):
    judge_prompt = (
        "You are grading an answer. Be strict and consistent.\n"
        f"Task: {instruction}\n"
        f"Input: {input}\n"
        f"Answer: {answer}\n"
        f"Rubric: {rubric}\n"
        f'Reply ONLY as JSON: {{"score": <1-{scale}>, "reason": "..."}}'
    )
    verdict = judge_client.chat(judge_prompt, temperature=0)   # one call per item
    obj = json.loads(extract_json(verdict))                    # robust to extra text
    return max(0.0, min(1.0, obj["score"] / scale))            # normalize to 0..1
```
`extract_json` pulls the first `{...}` block in case the judge adds prose. Use
`temperature=0` for repeatability. This is the only scorer that costs a model call per item.

---

## How the harness uses this

```python
def score_item(record, answer, embed=None, judge_client=None):
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
```

The `--mock` local pipeline exercises all of this with fake `answer`s, so it can be
built and tested offline before any Nebius endpoint exists.
