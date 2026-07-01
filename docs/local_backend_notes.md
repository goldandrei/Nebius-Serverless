# Local vLLM Backend — Environment Notes

## What it is

`--backend local` points the eval harness at a self-hosted vLLM server via
`LOCAL_BASE_URL` (default `http://localhost:8000/v1`) and `LOCAL_MODEL`.
The code is implemented in `src/eval_runner._run_local` and works identically
to `--backend tokenfactory` (OpenAI-compatible chat completions, wall-clock
latency, completion token count, cost fields left null).

## Why it didn't run in this dev environment

Attempted on: Windows 11 + WSL2 Ubuntu 24.04 + RTX 5070 Ti (Blackwell, sm_120)

vLLM 0.24.0 hit a cascade of incompatibilities:

| Attempt | Error |
|---|---|
| Default launch | SIGSEGV in model-arch inspection subprocess |
| + `CUDA_HOME` set (bundled nvcc) | FlashInfer JIT `nvcc` lookup resolved; new crash: fast tokenizer unpickling error |
| + `--tokenizer-mode slow` | `_deepcopy_tuple()` arity mismatch (Python 3.12.3 copy module ABI change) |
| Python 3.11 venv + correct `lib/` path | SIGSEGV on main process startup (Blackwell sm_120 kernel load) |

Root causes: Blackwell (sm_120) CUDA kernel compatibility with vLLM 0.24.0's
bundled FlashInfer/FlashAttention, combined with the CUDA toolkit being absent
from WSL2 (only the Windows driver stub `libcuda.so` is present, not `nvcc`).

## How to make local vLLM work

Requires a native Linux host (or WSL2 with a full CUDA toolkit install):

```bash
# Install CUDA toolkit matching your driver (e.g. CUDA 13.x for driver 610+)
sudo apt-get install -y cuda-toolkit-13-0   # or whatever version matches

# Then start the server:
bash scripts/start_vllm_local.sh
```

`scripts/start_vllm_local.sh` serves `Qwen/Qwen2.5-0.5B-Instruct` with dev
flags: `--gpu-memory-utilization 0.40 --max-model-len 2048 --max-num-seqs 8
--enforce-eager --port 8000`.

## Development workflow used instead

- **Iteration / scoring logic**: `--backend mock` (fully offline, deterministic)
- **Real-model validation**: `--backend tokenfactory` (Nebius Token Factory API,
  no GPU provisioning needed, requires `NEBIUS_API_KEY`)
- **Full cost+latency run**: `--backend endpoint` (creates Nebius Serverless
  Endpoints, real GPU-hour billing)
