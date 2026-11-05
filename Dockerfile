# One image for every service; docker-compose.yml picks the entry point per service.
FROM python:3.13-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY task1_mcp_server ./task1_mcp_server
COPY task2_mcp_gateway ./task2_mcp_gateway
COPY task3_llm_guardrail ./task3_llm_guardrail
COPY task4_model_router ./task4_model_router
# Non-root; /data holds the router's SQLite file (mounted as a volume).
RUN useradd --system --uid 10001 app && mkdir /data && chown app /data
USER app
EXPOSE 8400
CMD ["uvicorn", "--factory", "task4_model_router.app:app_factory", "--host", "0.0.0.0", "--port", "8400"]
