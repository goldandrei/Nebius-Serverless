"""Nebius Serverless AI Endpoint lifecycle via the nebius CLI.

The `nebius` binary (Linux/macOS only) is invoked:
  - directly when found in PATH (Linux CI, macOS dev)
  - via WSL on Windows: wsl bash -c "~/.nebius/bin/nebius ..."
"""
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── secret redaction ──────────────────────────────────────────────────────────

_REDACT_PATTERNS = [
    (re.compile(r'--token\s+\S+'),        '--token [REDACTED]'),
    (re.compile(r'Token:\s+\S+'),         'Token: [REDACTED]'),
    (re.compile(r'HF_TOKEN=\S+'),         'HF_TOKEN=[REDACTED]'),
    (re.compile(r'\b[0-9a-f]{40,}\b'),    '[REDACTED]'),   # long hex secrets
]

def _redact(text: str) -> str:
    """Strip auth tokens and long secrets from any string before surfacing it."""
    for pattern, replacement in _REDACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ── stderr → human reason ─────────────────────────────────────────────────────

_REASON_PATTERNS = [
    (re.compile(r'no preset found with name = "([^"]+)"'),
     lambda m: f'GPU preset "{m.group(1)}" is not available — check that the '
               f'instance_type in catalog.yaml is valid for this platform.'),
    (re.compile(r'no platform found with name = "([^"]+)"'),
     lambda m: f'GPU platform "{m.group(1)}" is not available in your account or region.'),
    (re.compile(r'gated|access to the model|401|unauthorized', re.I),
     lambda _: 'Model access denied — model may be gated. Accept the license on '
               'huggingface.co and set HF_TOKEN in your .env.'),
    (re.compile(r'quota|limit exceeded', re.I),
     lambda _: 'GPU quota exceeded — check your project limits in the Nebius console.'),
    (re.compile(r'insufficient.{0,20}capacity|no.{0,10}capacity', re.I),
     lambda _: 'Insufficient GPU capacity — try again later or choose a different instance type.'),
    (re.compile(r'out.of.memory|OOM', re.I),
     lambda _: 'Out of GPU memory — the model is too large for this preset.'),
]

def _extract_reason(stderr: str) -> str:
    """Parse nebius CLI stderr and return a short, human-readable failure reason."""
    for pattern, formatter in _REASON_PATTERNS:
        m = pattern.search(stderr)
        if m:
            return formatter(m)
    # Pull first desc = ... fragment
    m = re.search(r'desc\s*=\s*([^\n\r]+)', stderr)
    if m:
        return m.group(1).strip()
    # Fall back to first substantive line
    skip = re.compile(r'^(Hint|This issue|Trace|  -)', re.I)
    for line in stderr.splitlines():
        line = re.sub(r'^Error:\s*rpc error:\s*code\s*=\s*\S+\s*desc\s*=\s*', '', line.strip())
        if line and not skip.match(line):
            return line
    return "Deployment failed — see server logs for details."


def _nebius_cmd(args: list[str]) -> list[str]:
    """Build the OS-appropriate nebius command list."""
    if sys.platform == "win32":
        inner = 'export PATH="$HOME/.nebius/bin:$PATH"; nebius ' + shlex.join(args)
        return ["wsl", "bash", "-c", inner]
    return ["nebius", *args]


def _run(*args) -> dict:
    """Run `nebius <args> --format json` and return parsed output."""
    cmd = _nebius_cmd(list(args) + ["--format", "json"])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "") + (e.stdout or "")
        print(f"    [nebius] {_redact(stderr)}", flush=True)   # full detail server-side
        raise RuntimeError(_extract_reason(stderr)) from None
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
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "") + (e.stdout or "")
        print(f"    [nebius] {_redact(stderr)}", flush=True)   # full detail server-side
        raise RuntimeError(_extract_reason(stderr)) from None
    output = r.stdout + r.stderr
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Endpoint ID:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError(
        f"Could not find 'Endpoint ID:' in create output:\n{_redact(output)}"
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
                    auth_token: str = None,
                    tensor_parallel_size: int = 1,
                    max_model_len: int = 8192) -> str:
    """
    Create a Nebius Serverless AI Endpoint running vllm/vllm-openai.

    Returns endpoint_id. auth_token is passed in (or generated) so the caller
    already knows it before the endpoint is ready.

    Billing starts here.
    """
    if auth_token is None:
        auth_token = secrets.token_hex(32)

    hf_token  = os.environ.get("HF_TOKEN", "").strip()
    # Only forward a real token — placeholder values cause spurious HF auth warnings.
    hf_valid  = hf_token.startswith("hf_") and len(hf_token) > 20

    name = f"eval-{_slug(model_id)}"
    vllm_args = (
        f"--model {model_id} --host 0.0.0.0 --port 8000"
        f" --max-model-len {max_model_len}"
    )
    if tensor_parallel_size > 1:
        vllm_args += f" --tensor-parallel-size {tensor_parallel_size}"
    args = [
        "ai", "endpoint", "create",
        "--name", name,
        "--image", "vllm/vllm-openai:latest",
        "--container-command", "python3 -m vllm.entrypoints.openai.api_server",
        "--args", vllm_args,
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
    if hf_valid:
        args += ["--env", f"HF_TOKEN={hf_token}"]

    # Print the command that will be sent to the CLI — token redacted — so the
    # server log is the authoritative record of what actually ran.
    display = " ".join(f'"{a}"' if " " in str(a) else str(a) for a in args)
    print(f"    [cmd] {_redact(display)}", flush=True)

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


def wait_ready(endpoint_id: str, auth_token: str, progress_cb=None,
               load_timeout_s: int = 480,
               provision_timeout_s: int = 900) -> str:
    """
    Poll until the endpoint is RUNNING and the model weights are loaded.
    Returns the HTTPS base URL (without /v1 suffix).

    Per-stage budgets:
      PROVISIONING  provision_timeout_s  — GPU allocation queue (no billing)
      STARTING                 5 min     — image pull + VM boot
      RUNNING         load_timeout_s     — vLLM weight download + load (billing GPU)
      Hard cap        dynamic            — PROVISIONING + STARTING + load_timeout_s + 5 min

    provision_timeout_s: 15 min default; use 25 min for multi-GPU presets (harder to fulfill).
    load_timeout_s: 8 min default; set higher for large models (e.g. 600s for 32B, 900s for 70B+).

    progress_cb(**kw): called each poll with ep_state and ep_elapsed_s.
    """
    import requests

    STAGE_BUDGETS: dict[str, int] = {
        "PROVISIONING": provision_timeout_s, # no billing; multi-GPU needs more time
        "STARTING":     300,                 # 5 min — image pull / VM boot
        "RUNNING":      load_timeout_s,      # per-model — vLLM weight download + load
    }
    HARD_CAP = provision_timeout_s + 300 + load_timeout_s + 300  # 5-min margin

    t_start  = time.time()
    deadline = t_start + HARD_CAP
    stage_t0: dict[str, float] = {}  # state → wall time first observed
    base_url = None

    def _note_stage(state: str) -> None:
        if state and state not in stage_t0:
            stage_t0[state] = time.time()

    def _check_stage_budget(state: str) -> None:
        budget = STAGE_BUDGETS.get(state)
        if not budget or state not in stage_t0:
            return
        spent = time.time() - stage_t0[state]
        if spent <= budget:
            return
        if state == "PROVISIONING":
            raise RuntimeError(
                f"Stuck in PROVISIONING for {spent/60:.1f} min — "
                f"GPU capacity unavailable; try again later or choose a different instance type."
            )
        raise RuntimeError(
            f"Stage {state} exceeded {budget//60} min budget "
            f"({spent/60:.1f} min elapsed) — see server logs for details."
        )

    def _check_error(data: dict) -> None:
        status = data.get("status", {})
        if status.get("state") != "ERROR":
            return
        detail = status.get("message") or status.get("error") or ""
        for op in status.get("reconciling_operations", []):
            op_s = (op.get("status") or {})
            detail = detail or op_s.get("message") or op_s.get("error") or ""
        raise RuntimeError(
            "Endpoint entered ERROR state"
            + (f" — {detail}" if detail else
               " — check server logs (vLLM crash: OOM, max_model_len, weight load failure)")
        )

    # ── Phase 1: wait for PROVISIONING → STARTING → RUNNING ──────────────────
    print("    polling for RUNNING state...", flush=True)
    while time.time() < deadline:
        data    = get_endpoint(endpoint_id)
        state   = data.get("status", {}).get("state", "")
        elapsed = time.time() - t_start
        print(f"    state={state}  elapsed={elapsed:.0f}s", flush=True)

        if progress_cb:
            progress_cb(ep_state=state, ep_elapsed_s=elapsed)

        _check_error(data)
        _note_stage(state)
        _check_stage_budget(state)

        url = _extract_url(data)
        if url and state == "RUNNING":
            base_url = url
            break

        time.sleep(15)

    if not base_url:
        raise TimeoutError(
            f"Endpoint {endpoint_id} did not reach RUNNING within {HARD_CAP//60} min"
        )

    # ── Phase 2: wait for vLLM to respond on /v1/models ──────────────────────
    # Re-check endpoint state on EVERY poll — vLLM can crash and flip to ERROR
    # within seconds of RUNNING, and we must not hold the billing GPU for 8 min.
    _note_stage("RUNNING")  # in case phase 1 never recorded it
    print(f"    endpoint at {base_url} — waiting for model weights...", flush=True)
    while time.time() < deadline:
        elapsed = time.time() - t_start

        data = get_endpoint(endpoint_id)
        _check_error(data)
        _check_stage_budget("RUNNING")

        if progress_cb:
            progress_cb(ep_state="loading", ep_elapsed_s=elapsed)

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

    raise TimeoutError(
        f"Endpoint {endpoint_id} not ready within {HARD_CAP//60} min overall"
    )


def delete_endpoint(endpoint_id: str) -> None:
    """Delete endpoint. Billing stops here."""
    _run("ai", "endpoint", "delete", "--id", endpoint_id)
    print(f"    deleted {endpoint_id}", flush=True)


def list_endpoints() -> list:
    """Return all Serverless AI Endpoints for the project."""
    data = _run("ai", "endpoint", "list", "--parent-id", _project_id())
    return data.get("items", [])
