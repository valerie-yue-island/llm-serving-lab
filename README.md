# llm-serving-lab

Reading and benchmarking two minimal LLM inference engines — [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) and [Mini-SGLang](https://github.com/sgl-project/mini-sglang) — on the same model (Qwen3-0.6B) and the same workloads, to understand *why* one is faster than the other.

Written from a data engineer's point of view: latency percentiles, throughput, GPU memory, and cost per 1M tokens — not just a single tokens/s number.

## Plan

- [ ] **Phase 0** — Concepts: prefill vs decode, KV cache, PagedAttention, continuous batching, RadixAttention → [`notes/00-concepts.md`](notes/00-concepts.md)
- [ ] **Phase 1** — Run Nano-vLLM on a cloud GPU, benchmark, read the engine / scheduler / block manager / model runner
- [ ] **Phase 2** — Shared-prefix workload: prefix caching on vs off
- [ ] **Phase 3** — Same workload on Mini-SGLang; compare RadixAttention and overlap scheduling; final report
