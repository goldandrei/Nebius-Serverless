#!/usr/bin/env python3
"""
Offline dry-run: runs the full eval harness against mock endpoints and
writes results/results.json + patches dashboard/dashboard.html with live data.
"""
import json
import re
import sys
import webbrowser
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from src import eval_runner  # noqa: E402

RESULTS_PATH = ROOT / "results" / "results.json"
DASHBOARD_PATH = ROOT / "dashboard" / "dashboard.html"


def main():
    print("Running local mock eval...")
    results = eval_runner.run(mock=True)

    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  wrote {RESULTS_PATH.relative_to(ROOT)}")

    html = DASHBOARD_PATH.read_text(encoding="utf-8")
    patched = re.sub(
        r'(<script type="application/json" id="data">)(.*?)(</script>)',
        lambda m: m.group(1) + json.dumps(results) + m.group(3),
        html,
        flags=re.DOTALL,
    )
    DASHBOARD_PATH.write_text(patched, encoding="utf-8")
    print(f"  updated {DASHBOARD_PATH.relative_to(ROOT)}")

    print(f"\n=== EVAL COMPLETE ({results['meta']['mode']}) ===")
    print(f"benchmark : {results['meta']['benchmark']}")
    print(f"models    : {results['meta']['n_models']}   tasks: {results['meta']['n_tasks']}\n")
    hdr = f"{'model':<28}{'score':>8}{'mean_lat':>10}{'$/1k_tok':>11}{'run_cost':>11}"
    print(hdr)
    print("-" * len(hdr))
    for r in results["leaderboard"]:
        print(
            f"{r['model']:<28}{r['accuracy'] * 100:>7.1f}%"
            f"{r['mean_latency_s']:>9.2f}s"
            f"{'$' + format(r['cost_per_1k_tokens_usd'], '.4f'):>11}"
            f"{'$' + format(r['total_run_cost_usd'], '.4f'):>11}"
        )
    print(f"\nOpening {DASHBOARD_PATH.relative_to(ROOT)} ...")
    webbrowser.open(DASHBOARD_PATH.resolve().as_uri())


if __name__ == "__main__":
    main()
