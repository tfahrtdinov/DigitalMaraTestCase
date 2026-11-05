# Two training images from uv.lock, entry point `python -m <module under src/>`:
#   lgbm    LightGBM steps          docker compose run --rm train lgbm.run_hpo solar
#   neural  LSTM and MLP with torch docker compose run --rm train-neural neural.run ...
# torch is the CPU build (see [tool.uv.sources] in pyproject.toml).
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project

ENV PATH=/app/.venv/bin:$PATH PYTHONPATH=/app/src GIT_PYTHON_REFRESH=quiet
ENTRYPOINT ["python", "-m"]

FROM base AS lgbm
COPY src ./src
CMD ["lgbm.run_hpo", "--help"]

FROM base AS neural
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project --extra neural
COPY src ./src
CMD ["neural.run", "--help"]
