#!/usr/bin/env bash
# Reproduce the Nano-vLLM baseline benchmark on a fresh GPU pod.
#
# Tested 2026-10-24 on RunPod: RTX 4090 (24 GB), template "PyTorch 2.8.0 · CUDA 12.8.1 · Ubuntu 24.04".
# Result that run: 133,966 tokens in 22.57 s = 5,936 tok/s; peak memory 22,244 / 24,564 MiB; max util 90%.
#
# Usage (on the pod):
#   bash run_baseline.sh            # setup + benchmark
#   SKIP_SETUP=1 bash run_baseline.sh   # benchmark only (already installed)
#
# Needs an Ampere-or-newer NVIDIA GPU (flash-attn does not support T4 / V100).

set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
MODEL_DIR="$HOME/huggingface/Qwen3-0.6B"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="$WORKDIR/results/$RUN_ID"
mkdir -p "$OUT_DIR"

log() { echo -e "\n=== $* ==="; }

# ---------------------------------------------------------------- 1. environment
log "Environment"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$OUT_DIR/gpu.txt"
TORCH_BEFORE="$(python -c 'import torch; print(torch.__version__)')"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, '| cuda', torch.version.cuda)"

if [[ "${SKIP_SETUP:-0}" != "1" ]]; then
  # ------------------------------------------------------------- 2. code
  log "Clone Nano-vLLM"
  cd "$WORKDIR"
  [[ -d nano-vllm ]] || git clone https://github.com/GeeeekExplorer/nano-vllm.git
  cd nano-vllm

  # ------------------------------------------------------------- 3. install
  log "Install flash-attn (should download a prebuilt wheel in 1-2 min; if it compiles for long, Ctrl+C)"
  pip install flash-attn --no-build-isolation

  log "Install Nano-vLLM"
  pip install -e .

  TORCH_AFTER="$(python -c 'import torch; print(torch.__version__)')"
  if [[ "$TORCH_BEFORE" != "$TORCH_AFTER" ]]; then
    echo "WARNING: torch changed from $TORCH_BEFORE to $TORCH_AFTER during install. Stop and check before benchmarking."
    exit 1
  fi

  # ------------------------------------------------------------- 4. model
  log "Download Qwen3-0.6B (huggingface-hub 2.x uses 'hf', not 'huggingface-cli')"
  hf download Qwen/Qwen3-0.6B --local-dir "$MODEL_DIR"
fi

cd "$WORKDIR/nano-vllm"

# ---------------------------------------------------------------- 5. record versions
log "Versions"
{
  echo "nano-vllm commit: $(git rev-parse --short HEAD)"
  pip list 2>/dev/null | grep -iE "^(torch|flash|flash_attn|transformers|nano)" || true
} | tee "$OUT_DIR/versions.txt"

# ---------------------------------------------------------------- 6. benchmark + GPU monitor
log "Benchmark (256 requests, random input/output lengths 100-1024)"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -l 1 > "$OUT_DIR/gpu.log" &
MON=$!
trap 'kill $MON 2>/dev/null || true' EXIT

python bench.py 2>&1 | tee "$OUT_DIR/bench.log"

kill $MON 2>/dev/null || true

# ---------------------------------------------------------------- 7. summary
PEAK_MEM="$(sort -t, -k1 -n "$OUT_DIR/gpu.log" | tail -1 | cut -d, -f1 | tr -d ' ')"
MAX_UTIL="$(sort -t, -k2 -n "$OUT_DIR/gpu.log" | tail -1 | cut -d, -f2 | tr -d ' ')"
RESULT="$(grep -E '^Total:' "$OUT_DIR/bench.log" | tail -1 || true)"

log "Summary"
{
  echo "Run:             $RUN_ID"
  echo "GPU:             $(cat "$OUT_DIR/gpu.txt")"
  echo "Result:          $RESULT"
  echo "Peak memory:     ${PEAK_MEM} MiB"
  echo "Max utilization: ${MAX_UTIL} %"
} | tee "$OUT_DIR/summary.txt"

echo -e "\nSaved to $OUT_DIR — copy summary.txt (and versions.txt) off the pod, then TERMINATE it."
