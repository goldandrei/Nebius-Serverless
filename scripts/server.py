#!/usr/bin/env python3
"""
Local dev server for the eval dashboard.

  GET  /               → dashboard/dashboard.html
  GET  /results.json   → results/results.json  (404 if not yet run)
  GET  /api/catalog    → full model catalog as JSON
  GET  /api/selection  → current compare list as JSON
  GET  /api/tasks      → available benchmark tasks with metadata
  POST /api/run        → {models:[...], task:"..."} → run eval, return results JSON
"""
import json
import sys
import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env if present — must happen before any os.environ reads
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# ── pricing ───────────────────────────────────────────────────────────────────
# Loaded once at startup from config/prices.yaml; never fetched per-eval.
# The Nebius Token Factory pricing page (https://tokenfactory.nebius.com/pricing)
# is a pure JS SPA with no public pricing REST API — all pricing data is served
# from an internal IAM-authenticated billing service not reachable by API key.
# Therefore: prices are read from the local prices.yaml snapshot, which you
# update manually via `python scripts/update_prices.py`.
_PRICES_FILE = ROOT / "config" / "prices.yaml"
_prices_cache: dict = {}   # model_id -> {price_in_per_1m, price_out_per_1m}
_prices_as_of: str  = ""


def _load_prices_at_startup() -> None:
    """Read prices.yaml once and populate _prices_cache. Logs status clearly."""
    global _prices_cache, _prices_as_of
    import yaml

    if not _PRICES_FILE.exists():
        print("  [prices] prices.yaml not found — cost columns will be empty.")
        print(f"  [prices] Run: python scripts/update_prices.py")
        return

    try:
        data = yaml.safe_load(_PRICES_FILE.read_text(encoding="utf-8")) or {}
    except Exception as e:
        print(f"  [prices] Could not parse prices.yaml ({e}) — cost columns will be empty.")
        return

    _prices_as_of = data.get("prices_as_of", "unknown date")
    tf = data.get("tokenfactory") or {}

    populated, missing = [], []
    for model_id, rates in tf.items():
        p_in  = rates.get("price_in_per_1m")
        p_out = rates.get("price_out_per_1m")
        if p_in is not None and p_out is not None:
            _prices_cache[model_id] = {"price_in_per_1m": p_in, "price_out_per_1m": p_out}
            populated.append(model_id)
        else:
            missing.append(model_id)

    if populated:
        print(f"  [prices] Loaded from prices.yaml (as of {_prices_as_of}):")
        for mid in populated:
            r = _prices_cache[mid]
            print(f"           {mid}: in=${r['price_in_per_1m']}/1M  out=${r['price_out_per_1m']}/1M")
    if missing:
        print(f"  [prices] Missing rates for: {', '.join(missing)}")
        print(f"  [prices] Fill them in: python scripts/update_prices.py")
    if not populated and not missing:
        print(f"  [prices] prices.yaml is empty — cost columns will be empty.")
        print(f"  [prices] Run: python scripts/update_prices.py")


# Shared progress + run-guard state
_progress_lock = threading.Lock()
_progress: dict = {"running": False}
_run_lock = threading.Lock()   # prevents concurrent evals (one at a time)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            self._file(ROOT / "dashboard" / "dashboard.html", "text/html; charset=utf-8")
        elif p == "/results.json":
            f = ROOT / "results" / "results.json"
            self._file(f, "application/json") if f.exists() else self._json(None, 404)
        elif p == "/api/catalog":
            import yaml
            data = yaml.safe_load((ROOT / "config" / "catalog.yaml").read_text(encoding="utf-8"))
            self._json(data["models"])
        elif p == "/api/selection":
            import yaml
            data = yaml.safe_load((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
            self._json(data["compare"])
        elif p == "/api/tasks":
            from src.eval_runner import available_tasks
            self._json(available_tasks())
        elif p == "/api/progress":
            with _progress_lock:
                self._json(dict(_progress))
        elif p == "/api/routing":
            from urllib.parse import urlparse, parse_qs
            from src.eval_runner import load_catalog
            qs  = parse_qs(urlparse(self.path).query)
            ids = [m.strip() for m in qs.get("models", [""])[0].split(",") if m.strip()]
            cat = load_catalog()
            plan = {"endpoint": [], "tokenfactory": [], "unknown": []}
            for mid in ids:
                m = cat.get(mid)
                if not m:
                    plan["unknown"].append(mid)
                elif m.get("basis") == "hosted":
                    plan["tokenfactory"].append(mid)
                else:
                    plan["endpoint"].append(mid)
            self._json(plan)
        elif p == "/api/prices":
            self._json({
                "prices": _prices_cache,
                "prices_as_of": _prices_as_of,
                "source": "config/prices.yaml",
            })
        elif p.startswith("/api/tasks/"):
            task_id = p[len("/api/tasks/"):]
            from src.eval_runner import _load_jsonl, DATA_DIR
            f = DATA_DIR / f"{task_id}.jsonl"
            self._json(_load_jsonl(f) if f.exists() else [], 200 if f.exists() else 404)
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        if self.path == "/api/run":
            self._handle_run()
        elif self.path == "/api/build":
            self._handle_build()
        elif self.path.startswith("/api/upload"):
            self._handle_upload()
        elif self.path == "/api/test-progress":
            self._handle_test_progress()
        else:
            self.send_response(404); self.end_headers()

    def _handle_run(self):
        length   = int(self.headers.get("Content-Length", 0))
        body     = json.loads(self.rfile.read(length))
        selected = body.get("models", [])
        task     = body.get("task") or "assistant_commands"

        if not selected:
            self._json({"error": "No models selected"}, 400); return
        if len(selected) > 3:
            self._json({"error": "Maximum 3 models per comparison"}, 400); return

        # Reject if a run is already in progress
        if not _run_lock.acquire(blocking=False):
            self._json({"error": "A run is already in progress — wait for it to finish."}, 409)
            return

        lines = "# Updated by dashboard picker.\ncompare:\n" + \
                "".join(f"  - {m}\n" for m in selected)
        (ROOT / "config" / "models.yaml").write_text(lines, encoding="utf-8")

        with _progress_lock:
            _progress.update({"running": True, "done": False, "error": None,
                               "model": "", "model_idx": 0, "n_models": 0,
                               "item_idx": 0, "n_items": 0})

        def _progress_cb(**kw):
            with _progress_lock:
                _progress.update(kw)

        def _run_eval():
            try:
                from src import eval_runner
                results = eval_runner.run(task_file=task,
                                          progress_cb=_progress_cb, prices=_prices_cache)
                (ROOT / "results" / "results.json").write_text(
                    json.dumps(results, indent=2), encoding="utf-8"
                )
                with _progress_lock:
                    _progress.update({"running": False, "done": True, "error": None})
            except Exception as e:
                import traceback
                err = str(e)
                tr  = traceback.format_exc()
                print(f"\n[run error] {err}\n{tr}", flush=True)
                with _progress_lock:
                    # Clear ep_state/ep_model so the 800ms poller doesn't keep
                    # rendering the deploy spinner after the error is set.
                    _progress.update({"running": False, "done": True,
                                      "error": err, "trace": tr,
                                      "ep_state": None, "ep_model": None})
            finally:
                _run_lock.release()

        # Start eval in background thread; return immediately so HTTP doesn't timeout
        threading.Thread(target=_run_eval, daemon=True).start()
        self._json({"status": "started"})

    def _handle_build(self):
        """Assemble JSONL from plain pieces — user never writes JSONL directly."""
        length = int(self.headers.get("Content-Length", 0))
        body   = json.loads(self.rfile.read(length))

        method      = body.get("method", "programmatic")
        instruction = body.get("instruction", "").strip()
        inputs      = [i.strip() for i in body.get("inputs", []) if str(i).strip()]

        if not inputs:
            self._json({"error": "Provide at least one input."}, 400); return

        records = []
        if method == "programmatic":
            compare = body.get("compare", "exact")
            answers = [str(a).strip() for a in body.get("answers", [])]
            for idx, inp in enumerate(inputs, 1):
                gold = answers[idx - 1] if idx - 1 < len(answers) else ""
                if compare == "json_fields":
                    try: gold = json.loads(gold)
                    except json.JSONDecodeError: pass
                elif compare == "numeric":
                    try: gold = float(gold)
                    except ValueError: pass
                records.append({"id": idx, "instruction": instruction, "input": inp,
                                "gold": gold, "scorer": "programmatic", "compare": compare})

        elif method == "reference_match":
            metric     = body.get("metric", "lexical")
            references = [str(r).strip() for r in body.get("references", [])]
            for idx, inp in enumerate(inputs, 1):
                ref = references[idx - 1] if idx - 1 < len(references) else ""
                records.append({"id": idx, "instruction": instruction, "input": inp,
                                "reference": ref, "scorer": "reference_match", "metric": metric})

        elif method == "llm_judge":
            rubric = body.get("rubric", "").strip()
            if not rubric:
                self._json({"error": "Rubric is required for llm_judge scoring — "
                                     "a judge with no rubric is meaningless."}, 400)
                return
            scale  = int(body.get("scale", 5))
            for idx, inp in enumerate(inputs, 1):
                records.append({"id": idx, "instruction": instruction, "input": inp,
                                "rubric": rubric, "scorer": "llm_judge", "scale": scale})

        if not records:
            self._json({"error": "No records assembled."}, 400); return

        path = ROOT / "data" / "custom.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        self._json({"n": len(records), "task": "custom"})

    def _handle_upload(self):
        from urllib.parse import urlparse, parse_qs
        qs   = parse_qs(urlparse(self.path).query)
        task = qs.get("task", [""])[0]
        VALID = {"assistant_commands", "factual_qa", "instruction_following"}
        if task not in VALID:
            self._json({"error": f"Unknown task id. Must be one of: {', '.join(sorted(VALID))}"}, 400)
            return
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length).decode("utf-8")
        lines  = [l.strip() for l in body.splitlines() if l.strip()]
        try:
            records = [json.loads(l) for l in lines]
        except json.JSONDecodeError as e:
            self._json({"error": f"Invalid JSON: {e}"}, 400); return
        if not records:
            self._json({"error": "File has no valid rows"}, 400); return
        path = ROOT / "data" / f"{task}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        self._json({"n": len(records), "task": task})

    def _handle_test_progress(self):
        """Dev-only: POST a JSON dict to merge into _progress, for dashboard simulation.

        Examples:
          # Simulate PROVISIONING:
          curl -s -X POST http://localhost:7860/api/test-progress \
               -H "Content-Type: application/json" \
               -d '{"running":true,"done":false,"error":null,"ep_model":"DeepSeek-R1-Distill-Qwen-32B","ep_state":"PROVISIONING","ep_elapsed_s":12}'

          # Simulate loading (RUNNING but vLLM not yet ready):
          ... -d '{"ep_state":"loading","ep_elapsed_s":45}'

          # Simulate ERROR (proves dashboard stops spinner and shows error):
          ... -d '{"running":false,"done":true,"error":"Endpoint entered ERROR state — check server logs (vLLM crash: OOM, max_model_len, weight load failure)","ep_state":null,"ep_model":null}'
        """
        length = int(self.headers.get("Content-Length", 0))
        body   = json.loads(self.rfile.read(length))
        with _progress_lock:
            _progress.update(body)
        self._json({"ok": True, "progress": dict(_progress)})

    def _file(self, path: Path, content_type: str):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        print(f"  {fmt % args}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7860

    # ── tee all output to logs/server.log ────────────────────────────────────
    class _Tee:
        def __init__(self, *streams):
            self._s = streams
        def write(self, data):
            for s in self._s: s.write(data)
        def flush(self):
            for s in self._s: s.flush()
        def __getattr__(self, name):
            return getattr(self._s[0], name)

    _log_path = ROOT / "logs" / "server.log"
    _log_path.parent.mkdir(exist_ok=True)
    _log_fh   = open(_log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, _log_fh)
    sys.stderr = _Tee(sys.__stderr__, _log_fh)
    print(f"  [server] logging to {_log_path}")
    # ─────────────────────────────────────────────────────────────────────────

    # ── startup banner — shows git SHA so stale-process issues are obvious ────
    import subprocess as _sp
    _r = _sp.run(["git", "rev-parse", "--short", "HEAD"],
                  cwd=str(ROOT), capture_output=True, text=True)
    _sha = _r.stdout.strip() if _r.returncode == 0 else "unknown"
    print(f"  [server] git={_sha}  (restart server after every code change)")
    # Confirm the key runtime parameters so we can verify without deploying
    from src.orchestrator import create_endpoint as _ce, wait_ready as _wr
    import inspect as _ins
    _wr_sig = _ins.signature(_wr)
    _mml  = _ins.signature(_ce).parameters["max_model_len"].default
    _lto  = _wr_sig.parameters["load_timeout_s"].default
    _pto  = _wr_sig.parameters["provision_timeout_s"].default
    print(f"  [server] create_endpoint defaults: max_model_len={_mml}")
    print(f"  [server] wait_ready: PROVISIONING≤per-model ({_pto//60}m default, 40m for 4×GPU) "
          f"STARTING≤5m RUNNING≤per-model ({_lto//60}m default)")
    del _sp, _r, _sha, _ce, _wr, _ins, _wr_sig, _mml, _lto, _pto
    # ─────────────────────────────────────────────────────────────────────────

    print("\nLoading prices...")
    _load_prices_at_startup()
    print()

    httpd = ThreadingHTTPServer(("localhost", port), Handler)
    print(f"Dashboard -> http://localhost:{port}")
    print("Ctrl+C to stop.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
