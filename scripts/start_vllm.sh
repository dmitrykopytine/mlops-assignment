#!/usr/bin/env bash
#
# Start vLLM with your chosen configuration.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

set -euo pipefail

# Load HF_TOKEN (and any other vars) from .env if present.
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

# Runtime FP8 weight quantization toggle. On-the-fly quantizes the BF16
# checkpoint at load time (no separate download); the served model name stays
# unchanged, so the agent's VLLM_MODEL needs no edit. Halves weight bandwidth
# per decode step -> lower ITL on the MoE.
#   QUANT=fp8 ./scripts/start_vllm.sh   # enable
#   ./scripts/start_vllm.sh             # default: BF16 (no quantization)
QUANT="${QUANT:-}"

quant_args=()
if [[ -n "$QUANT" ]]; then
    quant_args+=(--quantization "$QUANT")
fi

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    ${quant_args[@]+"${quant_args[@]}"}
