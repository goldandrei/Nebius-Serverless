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
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env if present — must happen before any os.environ reads
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


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
        else:
            self.send_response(404); self.end_headers()

    def _handle_run(self):
        length   = int(self.headers.get("Content-Length", 0))
        body     = json.loads(self.rfile.read(length))
        selected = body.get("models", [])
        task     = body.get("task") or "assistant_commands"
        backend  = body.get("backend", "mock")

        if backend not in ("mock", "tokenfactory", "endpoint"):
            self._json({"error": f"Unknown backend {backend!r}"}, 400); return
        if not selected:
            self._json({"error": "No models selected"}, 400); return
        if len(selected) > 3:
            self._json({"error": "Maximum 3 models per comparison"}, 400); return

        lines = "# Updated by dashboard picker.\ncompare:\n" + \
                "".join(f"  - {m}\n" for m in selected)
        (ROOT / "config" / "models.yaml").write_text(lines, encoding="utf-8")

        try:
            from src import eval_runner
            results = eval_runner.run(backend=backend, task_file=task)
            (ROOT / "results" / "results.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8"
            )
            self._json(results)
        except Exception as e:
            import traceback
            self._json({"error": str(e), "trace": traceback.format_exc()}, 500)

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
    httpd = HTTPServer(("localhost", port), Handler)
    print(f"Dashboard -> http://localhost:{port}")
    print("Ctrl+C to stop.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
