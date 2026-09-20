# Standalone end-of-turn runtime. Build from the repository root:
#   docker build -t eot-runtime:local -f docker/turn_runtime/Dockerfile .
FROM python:3.12-slim

WORKDIR /app

# libsndfile is required by the soundfile wheel at import time.
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv using the official standalone binary.
COPY --from=ghcr.io/astral-sh/uv:0.11.1 /uv /uvx /bin/

# Workspace descriptors first so dependency layers cache independently of source.
COPY pyproject.toml uv.lock /app/
COPY packages/turn_runtime/pyproject.toml /app/packages/turn_runtime/pyproject.toml
COPY packages/turn_ui/pyproject.toml /app/packages/turn_ui/pyproject.toml
COPY packages/ml_core/pyproject.toml /app/packages/ml_core/pyproject.toml

# Whisper-tiny + a tiny MLP; CPU wheels are enough for a standalone image.
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_TORCH_BACKEND=cpu

RUN uv sync --frozen --no-dev --package turn-runtime --no-install-workspace --no-cache

COPY packages/turn_runtime/src /app/packages/turn_runtime/src
RUN uv sync --frozen --no-dev --package turn-runtime --no-cache --compile-bytecode

# Bake the frozen encoder and the current best pause head. `current` is a
# symlink on the host; Docker copies the resolved files.
COPY models/whisper-tiny /models/whisper-tiny
COPY models/pause_head/current/best.pt /models/pause_head/current/best.pt
COPY models/pause_head/current/config.json /models/pause_head/current/config.json

ENV PATH="/app/.venv/bin:$PATH"
ENV HOST=0.0.0.0
ENV PORT=8766
ENV WHISPER_MODEL_DIR=/models/whisper-tiny
ENV PAUSE_HEAD_DIR=/models/pause_head
ENV CORS_ALLOWED_ORIGINS=*

EXPOSE 8766

ENTRYPOINT ["turn-runtime-serve"]
