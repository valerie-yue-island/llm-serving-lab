# 00 · Concepts: How a decoder-only LLM runs at inference time

> Phase 0 of `llm-serving-lab` · 2026-10-17
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

## 3. The five Phase 0 questions

### Q1. Prefill vs decode — which is compute-bound, which is memory-bandwidth-bound?

Example request: 1,000-token prompt, 200 output tokens.

| | Prefill | Decode |
|---|---|---|
| What happens | All 1,000 prompt tokens go through the model in **one forward pass**, in parallel. K/V for all of them are written to the cache. Produces token #1. | **One token per forward pass**, 200 passes. Each step: compute the new token's Q/K/V → append K/V to the cache → attend against all cached K/V. |
| Bottleneck | **Compute-bound** — large matrix multiplies saturate the GPU | **Memory-bandwidth-bound** — to produce 1 token, the GPU reads all weights (~1.2 GB) + the whole KV cache |
| Metric | **TTFT** (time to first token) | **TPOT / ITL** (time per output token) |
| DE analogy | one bulk load of 1,000 rows | row-by-row inserts, each needing a full table scan |

### Q2. Why do we need a KV cache, and how big is it for Qwen3-0.6B?

**Why.** In decode, every new token's query must be compared with the keys — and read the values — of *all* previous tokens. Because of the causal mask, earlier tokens' K and V never change. So we compute them once and keep them in GPU memory instead of recomputing them every step (which would make total work grow as $O(n^2)$). It's **memoization**: memory traded for compute.

**Why only K and V, not Q?** Q is "me querying others" — once a token has been processed, its query is never used again. K and V are "others querying me" — every future token needs them.

**Size.** Stored per token, per layer, per KV head:

$$\text{KV bytes per token} = 2 \times L \times H_{kv} \times d_{head} \times b$$

Qwen3-0.6B: $L = 28$, $H_{kv} = 8$, $d_{head} = 128$, $b = 2$ (bf16)

- Per token: **TODO** KB
- One 4,096-token request: **TODO** MB
- Max concurrent 4,096-token requests on a 24 GB GPU (minus ~1.2 GB weights): **TODO**

| | Weights | KV cache |
|---|---|---|
| Origin | learned in training | computed from your input at inference |
| Changes? | read-only | grows with every token |
| Shared? | one copy for all requests | one copy per request |
| DE analogy | compiled program | per-session state |

### Q3. What problem does PagedAttention solve?

**TODO** — read vLLM paper sections 1–3.

### Q4. Continuous batching vs static batching?

**TODO**

### Q5. What computation does prefix caching / RadixAttention save?

**TODO** — read SGLang paper, RadixAttention section.

## 4. The one idea that ties serving together

Decode reads all the weights on every step. Running 64 requests together reads the weights once and produces 64 tokens — that's **batching**. But each request carries its own KV cache, so a bigger batch needs more memory.

> **KV cache memory management → max batch size → throughput → cost per token**

PagedAttention, continuous batching, and RadixAttention all work on the first link of this chain. That's what Phases 1–3 will measure.

## 5. Open question

Qwen3-0.6B has 16 query heads but only 8 KV heads. Why, and what does it do to KV cache size? **TODO**

## References

- 3Blue1Brown, *Attention in transformers, visually explained* (Deep Learning ch. 6)
- Jay Alammar, [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017)
- Kwon et al., [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) (2023)
- Zheng et al., [SGLang / RadixAttention](https://arxiv.org/abs/2312.07104) (2023)
