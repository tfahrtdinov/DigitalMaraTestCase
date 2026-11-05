# LightGBM training environment from uv.lock; the entry point is a module under src/.
#   docker compose run --rm train lgbm.run_hpo solar --trials 30
# The `neural` extra (torch) is left out: on Linux it pulls the CUDA stack, several GB.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project
COPY src ./src

ENV PATH=/app/.venv/bin:$PATH PYTHONPATH=/app/src GIT_PYTHON_REFRESH=quiet
ENTRYPOINT ["python", "-m"]
CMD ["lgbm.run_hpo", "--help"]
