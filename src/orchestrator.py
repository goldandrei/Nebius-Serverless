"""Nebius Serverless AI Endpoint lifecycle via the nebius CLI.

The `nebius` binary (Linux/macOS only) is invoked:
  - directly when found in PATH (Linux CI, macOS dev)
  - via WSL on Windows: wsl bash -c "~/.nebius/bin/nebius ..."
"""
import json
import os
import secrets
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _nebius_cmd(args: list[str]) -> list[str]:
    """Build the OS-appropriate nebius command list."""
    if sys.platform == "win32":
        inner = 'export PATH="$HOME/.nebius/bin:$PATH"; nebius ' + shlex.join(args)
        return ["wsl", "bash", "-c", inner]
    return ["nebius", *args]


def _run(*args) -> dict:
    """Run `nebius <args> --format json` and return parsed output."""
    cmd = _nebius_cmd(list(args) + ["--format", "json"])
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    text = r.stdout.strip()
    return json.loads(text) if text and text not in ("{}", "{}") else {}


def _run_create_async(args: list[str]) -> str:
    """
    Run `nebius ai endpoint create --async` and return the endpoint ID.

    With --async the CLI always prints text to stdout regardless of --format:
      Token: <token>
      Endpoint ID: aiendpoint-e00...
    Parse the Endpoint ID line directly.
    """
    cmd = _nebius_cmd(args)
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    output = r.stdout + r.stderr
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Endpoint ID:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError(
        f"Could not find 'Endpoint ID:' in create output:\n{output}"
    )


def _project_id() -> str:
    pid = os.environ.get("NEBIUS_PROJECT_ID", "")
    if not pid:
        raise RuntimeError("NEBIUS_PROJECT_ID is not set")
    return pid


def _subnet_id() -> str:
    sid = os.environ.get("NEBIUS_SUBNET_ID", "")
    if not sid:
        raise RuntimeError("NEBIUS_SUBNET_ID is not set")
    return sid


def _slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower().replace(".", "-")[:28]


# ── single endpoint ────────────────────────────────────────────────────────────

def create_endpoint(model_id: str, platform: str, preset: str,
                    auth_token: str = None) -> str:
    """
    Create a Nebius Serverless AI Endpoint running vllm/vllm-openai.

    Returns endpoint_id. auth_token is passed in (or generated) so the caller
    already knows it before the endpoint is ready.

    Billing starts here.
    """
    if auth_token is None:
        auth_token = secrets.token_hex(32)

    hf_token = os.environ.get("HF_TOKEN", "")
    name = f"eval-{_slug(model_id)}"
    args = [
        "ai", "endpoint", "create",
        "--name", name,
        "--image", "vllm/vllm-openai:latest",
        "--container-command", "python3 -m vllm.entrypoints.openai.api_server",
        "--args", f"--model {model_id} --host 0.0.0.0 --port 8000",
        "--platform", platform,
        "--preset", preset,
        "--container-port", "8000",
        "--shm-size", "16Gi",
        "--disk-size", "450Gi",
        "--auth", "token",
        "--token", auth_token,
        "--subnet-id", _subnet_id(),
        "--parent-id", _project_id(),
        "--async",
    ]
    if hf_token:
        args += ["--env", f"HF_TOKEN={hf_token}"]

    endpoint_id = _run_create_async(args)
    print(f"    created {endpoint_id} ({platform}/{preset})", flush=True)
    return endpoint_id


def get_endpoint(endpoint_id: str) -> dict:
    """Return current endpoint spec+status as a dict."""
    return _run("ai", "endpoint", "get", "--id", endpoint_id)


def _extract_url(data: dict) -> str | None:
    """Extract the HTTPS base URL from an endpoint get/list response.

    public_endpoints is a list of strings in the CLI JSON output:
      "public_endpoints": ["https://port8000-<hash>.tunnel....nebius.cloud"]
    """
    eps = data.get("status", {}).get("public_endpoints", [])
    for ep in eps:
        url = ep if isinstance(ep, str) else ep.get("url", "")
        if url.startswith("https://"):
            return url.rstrip("/")
    return None


def wait_ready(endpoint_id: str, auth_token: str, timeout_s: int = 1800) -> str:
    """
    Poll until the endpoint is RUNNING and the model weights are loaded.
    Returns the HTTPS base URL (without /v1 suffix).
    """
    deadline = time.time() + timeout_s
    base_url = None

    print("    polling for RUNNING state...", flush=True)
    while time.time() < deadline:
        data = get_endpoint(endpoint_id)
        state = data.get("status", {}).get("state", "")
        print(f"    state={state}", flush=True)

        if state == "ERROR":
            raise RuntimeError(f"Endpoint {endpoint_id} entered ERROR state")

        url = _extract_url(data)
        if url and state == "RUNNING":
            base_url = url
            break

        time.sleep(15)

    if not base_url:
        raise TimeoutError(f"Endpoint {endpoint_id} not ready after {timeout_s}s")

    # Wait for the model to actually respond on /v1/models
    import requests
    print(f"    endpoint at {base_url} — waiting for model weights...", flush=True)
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
    _run("ai", "endpoint", "delete", "--id", endpoint_id)
    print(f"    deleted {endpoint_id}", flush=True)


def list_endpoints() -> list:
    """Return all Serverless AI Endpoints for the project."""
    data = _run("ai", "endpoint", "list", "--parent-id", _project_id())
    return data.get("items", [])
