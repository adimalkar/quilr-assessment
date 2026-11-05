.PHONY: install test lint fmt demo demo-md bench serve-task2 serve-task3 serve-task4 serve-all mcp-stdio inspector up down

install:            ## uv sync (Python 3.12+)
	uv sync
test:               ## full test suite (real processes + sockets)
	uv run pytest
lint:               ## ruff lint + format check
	uv run ruff check . && uv run ruff format --check .
fmt:
	uv run ruff format . && uv run ruff check --fix .
demo:               ## narrated walkthrough of all four tasks
	uv run python scripts/demo.py
demo-md:            ## regenerate docs/DEMO.md from a real run
	uv run python scripts/demo.py --markdown
bench:              ## latency / throughput numbers quoted in the README
	uv run python scripts/bench.py
serve-task2 serve-task3 serve-task4 serve-all:  ## run a stack on fixed ports + print curls
	uv run python scripts/serve.py $(subst serve-,,$@)
mcp-stdio:          ## Task 1 MCP server on stdio
	uv run python -m task1_mcp_server
inspector:          ## Task 1 in the MCP Inspector UI
	npx @modelcontextprotocol/inspector uv run python -m task1_mcp_server
up:                 ## everything in containers (needs GATEWAY_JWT_SECRET)
	docker compose up --build
down:
	docker compose down
help:
	@grep -E "^[a-z0-9 -]+:.*##" Makefile | sed "s/:.*##/ —/"
