#!/usr/bin/env python3
"""
Guided helper to refresh config/prices.yaml from the Nebius Token Factory pricing page.

Usage:
  uv run --with pyyaml,requests python scripts/update_prices.py

The script fetches the Token Factory model list to confirm which models are
available, then prompts you to enter the current per-token prices from
studio.nebius.ai/pricing (the pricing page is not machine-readable).
It writes the result back to config/prices.yaml with today's date.
"""
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import yaml


def get_tf_models():
    """List models currently available on Token Factory."""
    try:
        import openai
        api_key  = os.environ.get("NEBIUS_API_KEY")
        base_url = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
        if not api_key:
            print("  NEBIUS_API_KEY not set — skipping live model fetch")
            return []
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        return [m.id for m in client.models.list().data]
    except Exception as e:
        print(f"  Could not fetch model list: {e}")
        return []


def prompt_price(model_id: str, field: str, current) -> float | None:
    label = "input" if "in" in field else "output"
    cur_str = f"${current}" if current is not None else "not set"
    raw = input(f"  {model_id}\n    {label} $/1M tokens [{cur_str}]: ").strip()
    if not raw:
        return current
    try:
        return float(raw)
    except ValueError:
        print("    Invalid — keeping current value")
        return current


def main():
    prices_file = ROOT / "config" / "prices.yaml"
    if not prices_file.exists():
        print(f"prices.yaml not found at {prices_file}")
        sys.exit(1)

    with open(prices_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    print("Fetching live Token Factory model list...")
    live = set(get_tf_models())
    if live:
        print(f"  Live models: {sorted(live)}")
    print()
    print("Open https://tokenfactory.nebius.com/pricing in your browser")
    print("(studio.nebius.ai/pricing redirects there).")
    print("Note: the pricing page is a JS SPA — no public REST API exists for prices.")
    print("Enter the $/1M token rates below. Press Enter to keep the current value.\n")

    tf = data.get("tokenfactory", {})
    for model_id, rates in tf.items():
        status = "(live)" if model_id in live else "(not in live list)"
        print(f"  {model_id} {status}")
        rates["price_in_per_1m"]  = prompt_price(model_id, "price_in_per_1m",  rates.get("price_in_per_1m"))
        rates["price_out_per_1m"] = prompt_price(model_id, "price_out_per_1m", rates.get("price_out_per_1m"))
        print()

    data["prices_as_of"] = date.today().isoformat()
    with open(prices_file, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Saved {prices_file} (prices_as_of: {data['prices_as_of']})")


if __name__ == "__main__":
    main()
