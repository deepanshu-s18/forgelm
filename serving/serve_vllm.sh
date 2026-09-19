#!/usr/bin/env bash
# serve_vllm.sh — Start ForgeLM as OpenAI-compatible endpoint via vLLM
#
# Usage:
#   bash serving/serve_vllm.sh                        # defaults
#   MODEL=checkpoints/stage2_sft/final bash ...       # after SFT only
#   PORT=8001 bash ...                                 # custom port
#
# Requires:
#   pip install vllm>=0.5
#   GPU with >=10GB VRAM for 1.5B bfloat16
#
# For CPU-only (slow, dev only):
#   MODEL=checkpoints/stage3_dpo/final bash serving/serve_vllm.sh --cpu

set -euo pipefail

MODEL="${MODEL:-checkpoints/stage3_dpo/final}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-4096}"
DTYPE="${DTYPE:-bfloat16}"

if [ ! -d "$MODEL" ]; then
  echo "ERROR: model directory not found: $MODEL" >&2
  echo "Run training stages first, or set MODEL= to a HuggingFace model ID" >&2
  exit 1
fi

echo "Starting ForgeLM server"
echo "  model:   $MODEL"
echo "  port:    $PORT"
echo "  max_len: $MAX_LEN"
echo ""

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --dtype "$DTYPE" \
  --served-model-name forgelm \
  "$@"
