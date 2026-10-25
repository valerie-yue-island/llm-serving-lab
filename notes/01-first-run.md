# 01 · First run: Nano-vLLM baseline benchmark

> `llm-serving-lab` · 2026-10-24 · Phase 1, day 1

## Setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 (24 GB VRAM) · RunPod on-demand · $0.89/hr |
| Environment | PyTorch 2.8.0 + CUDA 12.8 (driver supports up to 13.2) · flash-attn · nano-vllm 0.2.0 |
| Model | Qwen3-0.6B (bf16, ~1.2 GB of weights) |
| Workload | `bench.py`: 256 requests, input and output lengths each sampled randomly between 100 and 1024 tokens |

## Results

| Metric | RTX 4090 (mine) | RTX 4070 Laptop 8 GB (README) |
|---|---|---|
| Output tokens | 133,966 | 133,966 |
| Time | 22.57 s | 93.41 s |
| **Throughput** | **5,936 tok/s** | 1,434 tok/s |
| Peak GPU memory | 22,244 / 24,564 MiB (~90%) | — |
| Max GPU utilization | 90% | — |

Same token count → same workload, so the comparison is fair: **~4.1× faster**.

## What the numbers mean

- **~4× throughput:** decode is memory-bandwidth-bound, and the 4090 has roughly 4× the bandwidth of a laptop 4070. Its 24 GB (vs 8 GB) also leaves far more room for KV cache, so more requests fit in each batch and the weights are read once for a bigger batch.
- **~90% memory used with a 1.2 GB model:** the engine pre-allocates a KV cache pool at startup and splits it into blocks for PagedAttention.
  Rough split: weights ~1.2 GB + activations/overhead ~1 GB + **KV pool ~20 GB** ≈ 180–190K tokens at 112 KB/token ≈ 11–12K blocks of 16 tokens. This matches my Phase 0 estimate of ~200K tokens of KV capacity on a 24 GB GPU.
- **90% GPU utilization:** the remaining ~10% is mostly time between steps, when the CPU scheduler prepares the next batch.
- `bench.py` measures **offline throughput** only. Latency (TTFT / TPOT) needs a server-style test, planned for later.
- The `Prefill=2tok/s, Decode=407tok/s` progress line is the 1-request warm-up, not the benchmark.

## Setup notes (gotchas)

- `huggingface-hub` 2.x removed `huggingface-cli` → use `hf download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B`
- Install `flash-attn` before Nano-vLLM; check that `pip install -e .` doesn't replace the preinstalled torch.
- `Ctrl+Z` only pauses a job (`kill %1` to end it); use `Ctrl+C` to stop.
- Log GPU memory during a run:
  ```bash
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -l 1 > gpu.log &
  MON=$!; python bench.py; kill $MON
  ```
- RTX 3090 ($0.50/hr) was out of capacity; the pod sat in "Waiting for availability". Switched to RTX 4090.
- Terminate pods (not just stop) when done.

## Next

- Read Nano-vLLM's `engine.py` (no GPU needed) and find where the KV block pool is allocated.
- Later: prefix caching on vs off with a shared-prefix workload; latency (TTFT / TPOT); Mini-SGLang on the same workload.
