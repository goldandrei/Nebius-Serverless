#!/usr/bin/env python3
"""
Recompute results.json from cached run data applying all three honesty fixes:

  Fix 1 — Inference-only endpoint cost: use per-item latency sum, not wall-clock.
  Fix 2 — Real Token Factory prices from prices.yaml.
  Fix 3 — Embedding-scorer caveat flag in meta (dashboard picks it up).

No GPU spend — reads the existing results/results.json and prices.yaml.
"""
import json
import sys
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

import yaml

# ── load inputs ───────────────────────────────────────────────────────────────
results_path = ROOT / "results" / "results.json"
prices_path  = ROOT / "config" / "prices.yaml"

results = json.loads(results_path.read_text(encoding="utf-8"))
prices  = yaml.safe_load(prices_path.read_text(encoding="utf-8"))
tf_prices = prices.get("tokenfactory", {})

catalog_raw = yaml.safe_load((ROOT / "config" / "catalog.yaml").read_text(encoding="utf-8"))
catalog = {m["id"]: m for m in catalog_raw["models"]}

print("=" * 60)
print("Recomputing results with honesty fixes")
print("=" * 60)

# ── rebuild leaderboard rows ──────────────────────────────────────────────────
new_lb = []

for row in results["leaderboard"]:
    mid   = row["model"]
    basis = row["cost_basis"]
    r     = dict(row)  # copy

    if basis == "self-hosted":
        # Fix 1: recompute eval_cost from sum of per-item inference latencies
        # (stored in samples.answers[mid].latency_s)
        inf_lats = []
        out_toks = []
        for s in results["samples"]:
            ans = s["answers"].get(mid)
            if ans:
                inf_lats.append(ans.get("latency_s", 0))
                out_toks.append(ans.get("out_tokens", 0))

        inference_s   = sum(inf_lats)
        total_out     = sum(out_toks) or 1
        rate_hr       = catalog.get(mid, {}).get("rate_hr", 1.55)

        t_created  = row["t_created"]
        t_ready    = row["t_ready"]
        t_eval_done = row["t_eval_done"]

        deploy_cost  = (t_ready - t_created) / 3600 * rate_hr
        eval_cost    = inference_s / 3600 * rate_hr          # inference-only
        total_cost   = (t_eval_done - t_created) / 3600 * rate_hr
        c1k          = eval_cost / total_out * 1000

        r["inference_s"]         = round(inference_s, 3)
        r["deploy_cost_usd"]     = round(deploy_cost, 6)
        r["eval_cost_usd"]       = round(eval_cost, 6)
        r["total_run_cost_usd"]  = round(total_cost, 6)
        r["cost_per_1k_tokens_usd"] = round(c1k, 6)

        old_c1k = row.get("cost_per_1k_tokens_usd", "?")
        print(f"\n[self-hosted] {mid}")
        print(f"  inference_s          = {inference_s:.3f}s  (sum of {len(inf_lats)} items)")
        print(f"  output tokens        = {total_out}")
        print(f"  deploy_cost          = ${deploy_cost:.4f}  (startup tax, unchanged)")
        print(f"  eval_cost  OLD       = ${row.get('eval_cost_usd', row.get('eval_cost', '?')):.4f}  (wall-clock incl. scoring)")
        print(f"  eval_cost  NEW       = ${eval_cost:.6f}  (inference-only)")
        print(f"  total_cost           = ${total_cost:.4f}  (full uptime, unchanged)")
        print(f"  $/1K tok   OLD       = ${old_c1k:.5f}")
        print(f"  $/1K tok   NEW       = ${c1k:.5f}  (inference-only)")

    elif basis == "hosted":
        # Fix 2: apply real per-token prices
        p = tf_prices.get(mid, {})
        p_in  = p.get("price_in_per_1m")
        p_out = p.get("price_out_per_1m")
        total_in  = row.get("total_in_tokens", 0)
        total_out = row.get("total_out_tokens", 0)
        total_tok = total_in + total_out

        print(f"\n[hosted]      {mid}")
        print(f"  price_in  = ${p_in}/1M  price_out = ${p_out}/1M")
        print(f"  tokens    = {total_in} in + {total_out} out = {total_tok} total")

        if p_in is not None and p_out is not None and total_tok > 0:
            run_cost = (total_in * p_in + total_out * p_out) / 1_000_000
            c1k      = run_cost / total_tok * 1000
            r["cost_per_1k_tokens_usd"] = round(c1k, 6)
            r["total_run_cost_usd"]     = round(run_cost, 6)
            print(f"  run_cost  = ${run_cost:.6f}")
            print(f"  $/1K tok  = ${c1k:.6f}")
        else:
            print(f"  prices missing for {mid} — cost columns stay empty")

    new_lb.append(r)

# Re-sort by accuracy
new_lb.sort(key=lambda x: x["accuracy"], reverse=True)

# ── rebuild meta ──────────────────────────────────────────────────────────────
new_meta = dict(results["meta"])
# Tag whether embedding scoring is in use so dashboard can show caveat
has_embedding = any(
    s.get("scorer") == "reference_match"
    and any(a.get("sim") is not None for a in s.get("answers", {}).values())
    for s in results.get("samples", [])
)
new_meta["embedding_scoring"] = has_embedding

# ── write corrected results ───────────────────────────────────────────────────
corrected = {
    "meta":        new_meta,
    "leaderboard": new_lb,
    "samples":     results["samples"],
}
results_path.write_text(json.dumps(corrected, indent=2, ensure_ascii=False), encoding="utf-8")

# ── print corrected leaderboard ───────────────────────────────────────────────
print()
print("=" * 60)
print("CORRECTED LEADERBOARD")
print("=" * 60)

hdr = f"  {'Model':<40} {'Basis':<12} {'Score':>6} {'✓':>4} {'MeanLat':>8} {'$/1Ktok':>10} {'RunCost':>10}"
print(hdr)
print("  " + "─" * (len(hdr) - 2))

best_c1k = min(
    (r["cost_per_1k_tokens_usd"] for r in new_lb if r.get("cost_per_1k_tokens_usd") is not None),
    default=None
)

for r in new_lb:
    name  = r["model"].split("/")[1][:38]
    basis = r["cost_basis"]
    acc   = f"{r['accuracy']*100:.1f}%"
    cor   = f"{r['correct']}/{r['n']}"
    lat   = f"{r['mean_latency_s']:.2f}s"
    c1k   = r.get("cost_per_1k_tokens_usd")
    cost  = r.get("total_run_cost_usd")
    c1k_s = f"${c1k:.5f}" if c1k is not None else "—"
    cost_s= f"${cost:.4f}" if cost is not None else "—"
    if c1k is not None and c1k == best_c1k:
        c1k_s += " ★"
    print(f"  {name:<40} {basis:<12} {acc:>6} {cor:>4} {lat:>8} {c1k_s:>12} {cost_s:>10}")

print()
print("  Notes on corrected numbers:")
for r in new_lb:
    if r["cost_basis"] == "self-hosted":
        inf_s = r.get("inference_s", 0)
        d_s   = r["t_ready"] - r["t_created"]
        t_s   = r["t_eval_done"] - r["t_created"]
        print(f"  {r['model'].split('/')[1]}:")
        print(f"    deploy  = {d_s/60:.1f} min  → ${r['deploy_cost_usd']:.4f} (startup tax, billed, one-time)")
        print(f"    infer   = {inf_s:.1f}s     → ${r['eval_cost_usd']:.6f} (pure vLLM serving)")
        print(f"    total   = {t_s/60:.1f} min  → ${r['total_run_cost_usd']:.4f} (full uptime billed)")
        print(f"    $/1K    = ${r['cost_per_1k_tokens_usd']:.5f} (inference-only, NOT including startup)")
    elif r["cost_basis"] == "hosted":
        p    = tf_prices.get(r["model"], {})
        t_in = r.get("total_in_tokens", 0)
        t_out= r.get("total_out_tokens", 0)
        print(f"  {r['model'].split('/')[1]}:")
        print(f"    in={t_in} × ${p.get('price_in_per_1m','?')}/1M  out={t_out} × ${p.get('price_out_per_1m','?')}/1M")
        if r.get("total_run_cost_usd") is not None:
            print(f"    total = ${r['total_run_cost_usd']:.6f}  $/1K = ${r['cost_per_1k_tokens_usd']:.6f}")

print()
print("  ⚠  Embedding-scorer caveat" if has_embedding else "  (no embedding scoring detected)")
if has_embedding:
    print("     Scores reflect cosine similarity to a SHORT reference sentence.")
    print("     Qwen3-32B generates chain-of-thought + long answers → lower sim.")
    print("     Accuracy ranking does NOT reliably reflect model quality here.")
    print("     Dashboard will show a warning when these results are loaded.")
print()
print(f"  Corrected results saved to results/results.json")
