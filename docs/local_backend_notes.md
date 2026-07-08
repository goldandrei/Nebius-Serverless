# Local and mock backends removed — project runs on Nebius Serverless endpoints, with Token Factory as the judge/hosted layer.

## Why local vLLM was removed

The `--backend local` and `--backend mock` backends have been removed. The project
now supports exactly two backends: `tokenfactory` (Nebius AI Studio hosted API) and
`endpoint` (Nebius Serverless GPU endpoints). The local backend was attempted but
never reliably worked due to the environment issues documented below.

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

## Development workflow

- **Real-model validation**: `--backend tokenfactory` (Nebius Token Factory API,
  no GPU provisioning needed, requires `NEBIUS_API_KEY`)
- **Full cost+latency run**: `--backend endpoint` (creates Nebius Serverless
  Endpoints, real GPU-hour billing)
