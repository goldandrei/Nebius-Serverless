"""Nebius dedicated endpoint lifecycle — REST API (no CLI required)."""
import os
import time
from pathlib import Path

import requests

API_BASE   = "https://api.tokenfactory.nebius.com"
ROOT       = Path(__file__).resolve().parent.parent


def _api_key() -> str:
    key = os.environ.get("NEBIUS_API_KEY")
    if not key:
        raise RuntimeError("NEBIUS_API_KEY is not set")
    return key


def _headers() -> dict:
    return {"Authorization": f"Bearer {_api_key()}"}


def _slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower().replace(".", "-")[:30]


# ── single endpoint ────────────────────────────────────────────────────────────

def create_endpoint(model_name: str, flavor_name: str, gpu_type: str,
                    gpu_count: int, region: str) -> tuple[str, str]:
    """POST /v0/dedicated_endpoints. Returns (endpoint_id, routing_key). Billing starts here."""
    r = requests.post(
        f"{API_BASE}/v0/dedicated_endpoints",
        headers=_headers(),
        json={
            "name":        f"eval-{_slug(model_name)}",
            "model_name":  model_name,
            "flavor_name": flavor_name,
            "gpu_type":    gpu_type,
            "region":      region,
            "gpu_count":   gpu_count,
            "scaling":     {"min_replicas": 1, "max_replicas": 1},
        },
        timeout=30,
    )
    r.raise_for_status()
    ep = r.json()["endpoint"]
    print(f"    created endpoint {ep['id']} (routing_key={ep['routing_key']})", flush=True)
    return ep["id"], ep["routing_key"]


def get_endpoint(endpoint_id: str) -> dict:
    """GET /v0/dedicated_endpoints — filter by id (no per-id GET exists)."""
    all_eps = list_endpoints()
    for ep in all_eps:
        if ep.get("id") == endpoint_id:
            return ep
    raise LookupError(f"Endpoint {endpoint_id} not found in list")


def wait_ready(endpoint_id: str, timeout_s: int = 900) -> str:
    """Poll GET /v0/dedicated_endpoints/{id} until running+ready. Returns inference base URL."""
    deadline = time.time() + timeout_s
    print("    waiting for endpoint to be ready...", flush=True)
    while time.time() < deadline:
        ep  = get_endpoint(endpoint_id)
        dep = ep.get("deployment", {})
        status    = dep.get("status")
        readiness = dep.get("readiness")
        ready     = dep.get("ready_replicas", 0)
        print(f"    [{status}] readiness={readiness} ready_replicas={ready}", flush=True)
        if status == "running" and readiness == "ready" and ready > 0:
            return f"{API_BASE}/v1"
        if status == "error":
            raise RuntimeError(f"Endpoint {endpoint_id} entered error state")
        time.sleep(20)
    raise TimeoutError(f"Endpoint {endpoint_id} not ready after {timeout_s}s")


def delete_endpoint(endpoint_id: str) -> None:
    """DELETE /v0/dedicated_endpoints/{id}. Billing stops here."""
    r = requests.delete(
        f"{API_BASE}/v0/dedicated_endpoints/{endpoint_id}",
        headers=_headers(),
        timeout=30,
    )
    r.raise_for_status()
    print(f"    deleted {endpoint_id}", flush=True)


def list_endpoints() -> list:
    """Return all current endpoints for this account."""
    r = requests.get(
        f"{API_BASE}/v0/dedicated_endpoints",
        headers=_headers(),
        timeout=15,
    )
    r.raise_for_status()
    return r.json().get("data", [])
