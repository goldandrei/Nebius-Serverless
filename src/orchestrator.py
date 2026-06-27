"""Nebius endpoint lifecycle management."""
import json
import os
import subprocess
import time
from pathlib import Path

import requests

ROOT       = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "results" / ".endpoints.json"


def _slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower().replace(".", "-")[:30]


def _run(*args) -> dict:
    r = subprocess.run(
        ["nebius", *args, "--format", "json"],
        capture_output=True, text=True, check=True,
    )
    return json.loads(r.stdout)


def _extract_url(data: dict) -> str | None:
    status = data.get("status", {})
    # Try the known field paths; adjust if your CLI version differs
    for key in ("endpoint_url", "url", "address", "external_address"):
        if status.get(key):
            return status[key]
    return None


# ── single endpoint ───────────────────────────────────────────────────────────

def create_endpoint(model_id: str, preset: str, instance_type: str,
                    subnet_id: str, auth_token: str) -> str:
    """Create a vLLM endpoint and return its endpoint ID. Billing starts here."""
    name      = f"eval-{_slug(model_id)}"
    hf_token  = os.environ.get("HF_TOKEN", "")

    cmd = [
        "nebius", "ai", "endpoint", "create",
        "--name", name,
        "--image", "vllm/vllm-openai:latest",
        "--container-command", "python3 -m vllm.entrypoints.openai.api_server",
        "--args", f"--model {model_id} --host 0.0.0.0 --port 8000",
        "--platform", preset,
        "--preset", instance_type,
        "--public",
        "--container-port", "8000",
        "--shm-size", "16Gi",
        "--disk-size", "450Gi",
        "--auth", "token",
        "--token", auth_token,
        "--subnet-id", subnet_id,
        "--format", "json",
    ]
    if hf_token:
        cmd += ["--env", f"HF_TOKEN={hf_token}"]

    r    = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(r.stdout)
    return data["metadata"]["id"]


def get_url(endpoint_id: str) -> str | None:
    try:
        data = _run("ai", "endpoint", "get", "--id", endpoint_id)
        return _extract_url(data)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def wait_ready(endpoint_id: str, auth_token: str, timeout_s: int = 600) -> str:
    """Poll until endpoint is RUNNING and model weights are loaded. Returns base URL."""
    deadline = time.time() + timeout_s
    base_url = None

    print(f"    waiting for RUNNING state...", flush=True)
    while time.time() < deadline:
        url = get_url(endpoint_id)
        if url:
            base_url = url.rstrip("/")
            break
        time.sleep(15)

    if not base_url:
        raise TimeoutError(f"Endpoint {endpoint_id} never got a URL in {timeout_s}s")

    print(f"    endpoint at {base_url} — waiting for weights to load...", flush=True)
    while time.time() < deadline:
        try:
            r = requests.get(
                f"{base_url}/v1/models",
                headers={"Authorization": f"Bearer {auth_token}"},
                timeout=10,
            )
            if r.status_code == 200:
                print("    model ready.", flush=True)
                return base_url
        except requests.RequestException:
            pass
        time.sleep(20)

    raise TimeoutError(f"Model never responded at {base_url}/v1/models in {timeout_s}s")


def delete_endpoint(endpoint_id: str) -> None:
    """Delete endpoint. Billing stops here."""
    subprocess.run(
        ["nebius", "ai", "endpoint", "delete", "--id", endpoint_id, "--yes"],
        check=True,
    )
    print(f"    deleted {endpoint_id}", flush=True)


# ── multi-model helpers (make up / down) ──────────────────────────────────────

def _load_models() -> list:
    from src.eval_runner import load_config
    return load_config()


def create_all() -> None:
    """Create all endpoints from config/models.yaml and save state."""
    import secrets
    subnet_id  = os.environ["NEBIUS_SUBNET_ID"]
    auth_token = secrets.token_hex(32)
    models     = _load_models()
    state      = {"auth_token": auth_token, "endpoints": {}}

    for m in models:
        print(f"  creating endpoint for {m['id']}...")
        eid = create_endpoint(m["id"], m["preset"], m["instance_type"],
                              subnet_id, auth_token)
        state["endpoints"][m["id"]] = eid
        print(f"    id: {eid}")

    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"  state saved to {STATE_FILE.relative_to(ROOT)}")


def delete_all() -> None:
    """Delete all endpoints recorded in state file, or scan by name prefix."""
    if not STATE_FILE.exists():
        print("No state file — scanning for eval-* endpoints...")
        _delete_by_prefix("eval-")
        return

    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    for model_id, eid in state.get("endpoints", {}).items():
        print(f"  deleting {model_id} ({eid})...")
        try:
            delete_endpoint(eid)
        except subprocess.CalledProcessError as e:
            print(f"    warning: {e}")

    STATE_FILE.unlink(missing_ok=True)


def _delete_by_prefix(prefix: str) -> None:
    data = _run("ai", "endpoint", "list")
    for ep in data.get("items", []):
        name = ep.get("metadata", {}).get("name", "")
        if name.startswith(prefix):
            eid = ep["metadata"]["id"]
            print(f"  deleting {name} ({eid})...")
            try:
                delete_endpoint(eid)
            except subprocess.CalledProcessError as e:
                print(f"    warning: {e}")
