# 00 · Concepts: How a decoder-only LLM runs at inference time

> Phase 0 of `llm-serving-lab` · 2026-10-17 (Q3–Q5 added 2026-10-21)
> Goal: understand enough of the transformer and the inference process to read Nano-vLLM and Mini-SGLang next week.
> Reference model: **Qwen3-0.6B** ([config.json](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json))

---

## 1. The big picture

```
text → tokenizer → token IDs → embedding lookup → vectors (hidden_size = 1024)
                                      ↓
                ┌──────────── Block 1 ─────────────┐
                │  Attention  (mix info ACROSS tokens) │
                │  MLP        (process EACH token)     │
                └──────────────────────────────────┘
                                      ↓
                              Block 2 … Block 28
                                      ↓
              final RMSNorm → LM head → next-token probabilities
                                      ↓
              append the new token, repeat  (autoregressive generation)
```

**A misconception I corrected:** attention is not "one layer followed by many feed-forward layers". Every block has its own attention and its own MLP, alternating. Qwen3-0.6B = 28 attention layers + 28 MLP layers.

Each block (pre-norm, with residual connections):

$$x \leftarrow x + \text{Attention}(\text{RMSNorm}(x))$$
$$x \leftarrow x + \text{MLP}(\text{RMSNorm}(x))$$

| Term | Meaning |
|---|---|
| **Token** | A piece of text produced by the tokenizer (often a subword), identified by a token ID. Not a vector. |
| **Embedding / hidden state** | The vector for a token at input / while it flows through the blocks |
| **Parameters ≈ weights** | The ~0.6B numbers learned in training. Fixed at inference. ~1.2 GB in bf16 |
| **Hyperparameters** | Chosen by humans (layers, heads…), stored in `config.json` |
| **MLP = FFN = feed-forward layer** | Same sub-layer. Qwen uses SwiGLU (`gate_proj`, `up_proj`, `down_proj`) |
| **Decoder-only** | Each token can only see earlier tokens (causal). GPT, Llama, Qwen, Claude |

## 2. Attention as a soft database lookup

| Step | My first intuition | More precise | Database analogy |
|---|---|---|---|
| $Q = XW_Q$ | the vector becomes a query ✅ | "what am I looking for?" | the WHERE condition |
| $K = XW_K$ | the answer to the query | a **label** each token advertises — not the answer itself | the index |
| $V = XW_V$ | (missed it) | the **content** handed over if attended to | the row data |
| $Q \cdot K$ | relevance ✅ | scaled by $\sqrt{d_k}$ for stability | match score per row |
| softmax | attention pattern ✅ | weights that sum to 1 | how much of each row to take |
| weighted sum of V | (missed it) | **Q·K decides what proportion of each V to take**; result is added back to the token | blended result rows |

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

Unlike a real database, this is a **soft lookup**: every row contributes, in proportion to relevance.

**Example.** "The cat sat because **it** was tired." When `it` attends: cat ≈ 80%, sat ≈ 8%, because ≈ 7%, The ≈ 5%, was/tired = 0% (masked). So `it` mostly absorbs `cat`'s value → the model "knows" what `it` refers to. *(illustrative numbers)*

**Causal mask.** $M_{ij} = 0$ if $j \le i$, else $-\infty$. Later tokens never affect earlier tokens, in every layer.

**Multi-head.** 16 query heads, each with its own projections, computed in parallel and concatenated. Heads are independent, so they can be split across GPUs → **tensor parallelism**.

### 2.1 What every number means (the "28-floor building")

| Number | Config name | What it is | Picture |
|---|---|---|---|
| 1024 | `hidden_size` | How many numbers describe one token (its row width) | each token is a row with 1,024 columns |
| 28 | `num_hidden_layers` | How many blocks the token passes through | a 28-floor building; the token visits every floor |
| 16 | `num_attention_heads` | Query heads per layer | 16 "askers" on each floor |
| 8 | `num_key_value_heads` | Key/value heads per layer | 8 "filing cabinets" on each floor |
| 128 | `head_dim` | How many numbers each head gets | each asker's / cabinet's card has 128 slots |

### 2.2 Shapes through one attention layer

```
token x: 1 × 1024
  × W_Q (1024 × 2048) → q: 2048 numbers → cut into 16 heads × 128
  × W_K (1024 × 1024) → k: 1024 numbers →   8 heads × 128   (stored in KV cache)
  × W_V (1024 × 1024) → v: 1024 numbers →   8 heads × 128   (stored in KV cache)
each of the 16 heads does attention → 16 outputs × 128 = 2048 (concat)
  × W_O (2048 → 1024) → back to 1024 → added to x (residual)
```

A projection like W_Q is a `SELECT` that computes 2,048 new columns, each a weighted sum of the token's 1,024 original columns. **16 × 128 = 2048 is the *output width* of W_Q**, not W_Q itself (W_Q is 1024 × 2048 ≈ 2.1M numbers per layer).

**W_O (`o_proj`)** does two things: brings the concatenated 2048 numbers back to 1024 so they can be added to x, and blends the 16 heads' results (the "chief editor" merging 16 detectives' reports).

| Code name | Symbol | Shape | Role |
|---|---|---|---|
| `q_proj` | W_Q | 1024 → 2048 (16 × 128) | queries |
| `k_proj` | W_K | 1024 → 1024 (8 × 128) | keys |
| `v_proj` | W_V | 1024 → 1024 (8 × 128) | values |
| `o_proj` | W_O | 2048 → 1024 | merge heads, restore width |

## 3. The five Phase 0 questions

### Q1. Prefill vs decode — which is compute-bound, which is memory-bandwidth-bound?

Example request: 1,000-token prompt, 200 output tokens.

| | Prefill | Decode |
|---|---|---|
| What happens | All 1,000 prompt tokens go through the model in **one forward pass**, in parallel. K/V for all of them are written to the cache. Produces token #1. | **One token per forward pass**, 200 passes. Each step: compute the new token's Q/K/V → append K/V to the cache → attend against all cached K/V. |
| Bottleneck | **Compute-bound** — large matrix multiplies saturate the GPU | **Memory-bandwidth-bound** — to produce 1 token, the GPU reads all weights (~1.2 GB) + the whole KV cache |
| Metric | **TTFT** (time to first token) | **TPOT / ITL** (time per output token) |
| DE analogy | one bulk load of 1,000 rows | row-by-row inserts, each needing a full table scan |

**Worked example.** Prompt: `I love machine learning` (4 tokens). In prefill, Q/K/V for all 4 positions are computed at once ($Q = XW_Q$ with $X$ of shape 4 × 1024), but the causal mask still limits what each position sees:

| Token | Can attend to |
|---|---|
| I | I |
| love | I, love |
| machine | I, love, machine |
| learning | I, love, machine, learning |

Positions are processed in parallel, but **layers still run one after another** (layer 2 needs layer 1's output).

**Decode inputs one new token per step — but that token attends to the whole history:**

$$\text{Attention}(Q_t, K_{\le t}, V_{\le t}) = \text{softmax}\left(\frac{Q_t K_{\le t}^\top}{\sqrt{d_k}}\right) V_{\le t}$$

**Why prefill parallelizes and decode doesn't:** the prompt is fully known, so every position can be computed at once. In decode, token $t+1$ doesn't exist until token $t$ is predicted — a sequential dependency *within one request*. Decode can still be parallel *across requests* (batching) or sped up with speculative decoding.

**One decode step, inside each layer:**

| Step | What happens | Uses weights? | Shared by batching? |
|---|---|---|---|
| ① | new token × W_Q, W_K, W_V → q, k, v | yes | **yes** — weights read once for the whole batch |
| ② | append k, v to this request's KV cache | no | — |
| ③ | q · all cached K → softmax → weighted sum of cached V | **no** — reads KV cache only | **no** — each request reads its own cache |
| ④ | × W_O, then MLP | yes | **yes** |

The KV cache is never multiplied by weights; only the **new token's vector** is.

**Metrics.** TTFT includes queueing + prefill + producing the first token. TPOT is the average time per later token: 50 tokens/s ≈ 20 ms/token — which says nothing about TTFT.

**Optimization directions differ by phase:**

| Prefill (compute-heavy) | Decode (memory-heavy) |
|---|---|
| Process long prompts efficiently; maximize compute throughput | Manage KV cache memory (PagedAttention, prefix caching) |
| FlashAttention (faster attention, less memory traffic) | Continuous batching for concurrency |
| | Quantization; speculative decoding |

### Q2. Why do we need a KV cache, and how big is it for Qwen3-0.6B?

**Why.** In decode, every new token's query must be compared with the keys — and read the values — of *all* previous tokens. Because of the causal mask, earlier tokens' K and V never change. So we compute them once and keep them in GPU memory instead of recomputing them every step (which would make total work grow as $O(n^2)$). It's **memoization**: memory traded for compute.

**Why only K and V, not Q?** Q is "me querying others" — once a token has been processed, its query is never used again. K and V are "others querying me" — every future token needs them.

**Size.** Stored per token, per layer, per KV head:

$$\text{KV bytes per token} = 2 \times L \times H_{kv} \times d_{head} \times b$$

Qwen3-0.6B: $L = 28$, $H_{kv} = 8$, $d_{head} = 128$, $b = 2$ (bf16)

- Per token: 2 × 28 × 8 × 128 × 2 = 114,688 bytes = **112 KB**
- One 4,096-token request: 114,688 × 4,096 = **448 MB**
- Max concurrent 4,096-token requests on a 24 GB GPU (minus ~1.2 GB weights): 22.8 GB ÷ 448 MB ≈ **50** (theoretical; activations and framework overhead make it lower)

> A 4K-token request's KV cache (448 MB) is over a third of the whole model (1.2 GB). Three such requests already outweigh the model. **In serving, GPU memory is eaten by KV cache, not by weights.**

**Common mistake I made:** I first wrote 28 × 8 × 128 × 2 bytes — that counts only K. The leading 2 is for K **and** V.

**Why `hidden_size` (1024) isn't in the formula:** the cache stores the *outputs* K = x·W_K and V = x·W_V, not the token vector x. Their width is (KV heads × head_dim). For Qwen3-0.6B that happens to also be 8 × 128 = 1024, but it's a coincidence — with 16 KV heads it would be 2048 while hidden_size stays 1024. **KV cache stores activations, not weights.**

| | Weights | KV cache |
|---|---|---|
| Origin | learned in training | computed from your input at inference |
| Changes? | read-only | grows with every token |
| Shared? | one copy for all requests | one copy per request |
| DE analogy | compiled program | per-session state |

### Q3. What problem does PagedAttention solve?

Contiguous, max-length KV allocation wastes **60–80%** of KV memory. PagedAttention splits the KV cache into **fixed-size blocks** (16 tokens in vLLM) allocated **on demand**, with a per-request **block table** mapping logical blocks to non-contiguous physical blocks. Waste is limited to each request's last, partially filled block (**< 4%** overall), so far more requests fit in memory.

| Waste before PagedAttention | What it is |
|---|---|
| Reservation | memory held for tokens that haven't been generated yet |
| Internal fragmentation | reserved for 4,096 tokens, request only uses 600 |
| External fragmentation | gaps between differently sized reservations — too small to fit a new request |

The idea is OS paging applied to the KV cache:

| Operating system | PagedAttention | DE analogy |
|---|---|---|
| process | request (sequence) | a file |
| page | KV block | HDFS block |
| byte | **token** | a row |
| page table | **block table** | NameNode metadata |
| physical memory | KV cache in GPU memory | data on DataNodes |

**My sizing** (22.8 GB for KV, 112 KB/token, max 4,096, average request 600 tokens, block = 16 tokens):

| | Reserve max length | PagedAttention |
|---|---|---|
| Memory per request | 4,096 × 112 KB = 448 MB | ⌈600/16⌉ = 38 blocks = 608 slots ≈ 66.5 MB |
| Waste | (4,096 − 600) / 4,096 ≈ **85%** | 8 empty slots in the last block ≈ **1.3%** |
| Concurrent requests | ≈ **52** | ≈ **351** (≈ 6.75×) |

**Why round up to 38 blocks?** A block is the smallest unit of allocation. 37 blocks hold 592 tokens; the last 8 still need a home, so a whole 38th block is allocated and half of it stays empty. Waste per request is always under one block.

**Block size is a trade-off.** Bigger blocks → shorter block tables and more efficient kernels, but more tail waste and coarser prefix sharing. vLLM uses 16 tokens; SGLang originally used 1-token pages.

**Copy-on-write.** When several outputs are sampled from one prompt, they share the prompt's physical blocks (reference counted). A block is copied only when one sequence needs to write into a shared block. (OS: `fork()`; DE: BigQuery table clones / git branches.)

**Mistakes I made:** mapped OS *byte* → byte (it's *token*) and *page table* → KV cache (it's the *block table*); described the block table as "pretending to reserve max length" — there is no reservation at all, allocation is on demand.

### Q4. How does continuous batching differ from static batching?

**Static batching** runs a fixed group until *every* request in it finishes. Slots of finished requests sit idle and queued requests wait, so the batch runs at the pace of its longest request. **Continuous batching** (iteration-level scheduling, from Orca) re-schedules at **every decode step**: when a request emits EOS, the scheduler admits a waiting request into the freed slot immediately.

**My hand simulation** — 2 slots; output lengths A = 2, B = 8, C = 3, D = 5:

```
Static:      slot1  A A – – – – – – | C C C – –
             slot2  B B B B B B B B | D D D D D      → 13 steps, 8 idle slot-steps
Continuous:  slot1  A A C C C D D D D D
             slot2  B B B B B B B B – –              → 10 steps, 2 idle slot-steps
```

| | Static | Continuous |
|---|---|---|
| Decode steps | 13 | **10** |
| Slot utilization | 18/26 ≈ 69% | 18/20 = **90%** |
| C starts / finishes | step 9 / 11 | **step 3 / 5** |

**Trade-off:** admitting a new request requires a prefill, which briefly stalls running decodes (TTFT vs TPOT; TGI exposes this as `waiting_served_ratio`). Requests joining and leaving at every step also make KV memory allocation dynamic — which is why continuous batching needs PagedAttention.

**How they fit together:** continuous batching keeps every step full; PagedAttention lets more requests fit in memory so each step can be bigger. In Anyscale's benchmark at QPS = 4, only vLLM (both techniques) kept p50 latency flat (~6 s); continuous batching alone rose to ~33 s.

**Mistake I made:** said static batching mainly wastes *memory*. It mainly wastes **GPU compute** (idle slots) and **queueing time**.

### Q5. What computation does prefix caching / RadixAttention save?

It eliminates **redundant prefill** of shared prefixes — system prompts, tool definitions, multi-turn history — and **duplicate KV storage**. A request whose tokens match a cached prefix **exactly from the first token** reuses that KV instead of recomputing it. Correctness follows from the causal mask: a prefix's K/V depend only on the prefix itself.

Example: 50 requests sharing a 2,000-token system prompt → ~10.7 GB of KV without sharing vs ~219 MB with sharing, and that prefix disappears from every request's prefill (lower TTFT).

**Prefix caching** is the technique; **RadixAttention** is SGLang's implementation of it.

| | vLLM (Automatic Prefix Caching) | SGLang (RadixAttention) |
|---|---|---|
| Lookup structure | hash table | radix tree over token sequences |
| Key | hash of a full block's tokens + the hash of everything before it | path from the root = the prefix |
| Match granularity | full blocks only (16 tokens) | any token (1-token pages) |
| Eviction | LRU over unused blocks | LRU over leaf nodes not used by any running request |
| Scheduling | — | **cache-aware**: admit requests with the longest cached prefix first |

The prefix must be part of a block's key because the same 16 tokens after a different prefix produce **different** K/V. **Matching granularity follows page size:** with 16-token pages you can't share 10 tokens of a page.

**Cache-aware scheduling** skips redundant prefill and keeps hot prefixes from being evicted, at some cost in fairness. It reduces *how much* work is done, not kernel efficiency (that depends on page size / memory layout). DE analogy: Spark's locality-aware scheduling.

**Mistake I made:** framed vLLM vs SGLang as "PagedAttention vs radix tree". Both page KV memory; the real difference is **how they find a reusable prefix** (hash table vs radix tree).

## 4. The one idea that ties serving together

Decode reads all the weights on every step. Running 64 requests together reads the weights once and produces 64 tokens — that's **batching**. But each request carries its own KV cache, so a bigger batch needs more memory.

> **KV cache memory management → max batch size → throughput → cost per token**

PagedAttention, continuous batching, and RadixAttention all work on the first link of this chain. That's what Phases 1–3 will measure.

**What exactly is batched?** The *new token vectors* of different requests, stacked into one matrix: `[64 × 1024] × W` instead of 64 separate `[1 × 1024] × W`. Same math, but the weights are read from GPU memory once instead of 64 times. DE analogy: weights are a dimension table every request joins against — batching scans it once. Each request's KV cache is like its own history table — it can't be shared.

| | Prefill | Decode |
|---|---|---|
| One request alone | already many tokens → big matrices | 1 token → tiny matrices |
| GPU state | compute already saturated | waiting on memory; compute idle |
| Batching gain | modest (fills the GPU when prompts are short) | **large** (amortizes the weight reads) |

Batching amortizes **weight reads**, not **KV cache reads** — and every extra request in the batch adds its own KV cache. So max batch size is capped by KV cache memory.

## 5. Why 16 Q heads but only 8 KV heads? (Grouped-Query Attention)

| | 16 KV heads (= Q heads) | 8 KV heads (Qwen3-0.6B) |
|---|---|---|
| KV per token | 2 × 28 × 16 × 128 × 2 = 224 KB | 112 KB |
| One 4,096-token request | 896 MB | 448 MB |
| Max concurrent on 24 GB | ≈ 25 | ≈ 50 |

Halving KV heads halves the KV cache → **2× more concurrent requests** (or 2× longer context), and each decode step reads half as much KV data → faster decode.

Q heads don't matter for the cache — queries are never stored, they only change the size of W_Q.

This design is **Grouped-Query Attention (GQA)**: the 16 query heads are split into 8 groups, and each pair of query heads shares one K/V head.

| Variant | KV heads | KV cache | Trade-off |
|---|---|---|---|
| MHA (multi-head) | = Q heads (16) | largest | most expressive |
| **GQA (grouped-query)** | between (8) | medium | small quality cost, big memory saving |
| MQA (multi-query) | 1 | smallest | biggest risk to quality |

**Cost:** query heads that share a K/V head can't each have their own "labels" and "content", so the model is slightly less expressive. GQA was adopted because the quality loss is small and the serving win is large.

## 6. Four sentences to remember

1. **Prefill** processes the whole prompt in parallel and builds the KV cache.
2. **Decode** generates one new token per step.
3. **Attention** lets the new token look at the entire allowed history, not just the previous token.
4. **KV cache** stores past keys and values so they're never recomputed.

`Prompt → Prefill → KV cache → Decode → new token → append to KV cache → Decode → …`

## References

- 3Blue1Brown, *Attention in transformers, visually explained* (Deep Learning ch. 6)
- Jay Alammar, [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017)
- Kwon et al., [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) (2023)
- Zheng et al., [SGLang / RadixAttention](https://arxiv.org/abs/2312.07104) (2023)
- vLLM blog, [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) (2023)
- Anyscale, [How continuous batching enables 23x throughput in LLM inference](https://www.anyscale.com/blog/continuous-batching-llm-inference) (2023)
- LMSYS, [Fast and Expressive LLM Inference with RadixAttention and SGLang](https://lmsys.org/blog/2024-01-17-sglang/) (2024)
