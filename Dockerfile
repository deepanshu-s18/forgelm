# ForgeLM CPU Inference Image
# Size target: < 1.5GB (no GPU deps, no training deps)
# Usage:
#   docker build -t forgelm .
#   docker run -p 8000:8000 \
#     -v $(pwd)/checkpoints/stage3_dpo/final:/model \
#     forgelm --model /model --port 8000

FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only inference deps (no torch/cuda)
COPY pyproject.toml .
COPY shared_schemas/ shared_schemas/
COPY forgelm/__init__.py forgelm/__init__.py
COPY forgelm/eval/ forgelm/eval/
COPY forgelm/serving/ forgelm/serving/

# Core + serving deps only (no train extras)
RUN pip install --no-cache-dir \
    numpy>=1.24 pandas>=2.0 pydantic>=2.5 structlog>=23.0 \
    openai>=1.30 vllm>=0.5 scipy>=1.11

# Install the package itself
RUN pip install --no-cache-dir -e "." --no-deps

EXPOSE 8000

ENTRYPOINT ["python", "-m", "forgelm.serving.alphaforge_adapter"]
CMD ["--model", "/model", "--port", "8000"]
