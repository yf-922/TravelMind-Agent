# Production Readiness Notes

This repository includes a reproducible local deployment path for demos and interviews.

## Start with Docker

1. Copy `.env.example` to `.env.local` and configure an LLM provider and keys.
2. Run `docker compose up --build`.
3. Open `http://localhost:8765`.

The compose stack persists SQLite data in a named volume and runs Redis for external API caching. The API container exposes a Docker health check backed by `/api/health`.

## Operational endpoints

- `GET /api/health`: checks API liveness, SQLite availability, and Redis readiness.
- `GET /api/metrics`: dependency-free Prometheus counters for HTTP traffic plus Agent run/node outcomes and latency.
- `GET /api/agent-runs/{run_id}`: owner-only execution status, attempts, slowest nodes, degraded services, observed parallel overlap, and request-attributed LLM usage.
- Optional platform prototypes: deterministic model routing and local Agent Card/MCP manifest/A2A envelope contracts; remote protocol transport is not enabled by default.
- Every response contains `X-Request-ID`; clients may provide one to correlate a request across logs and SSE calls.
- Every planning stream emits a `run` event and returns `X-Agent-Run-ID` for trace lookup.
- `PLAN_TIMEOUT_SECONDS` bounds each run (default 180 seconds); timeout and client disconnect outcomes are tracked.
- Docker disables Chroma embedding downloads by default and uses source-labelled lexical retrieval. Set `SEMANTIC_MEMORY_ENABLED=1` and `TRAVEL_KNOWLEDGE_ENABLED=1` after prewarming the model; semantic-memory lookup still runs off the event loop and degrades after `MEMORY_LOOKUP_TIMEOUT_SECONDS` (default 2.5 seconds).

The in-memory trace store is capped at 200 recent runs. Metrics and traces intentionally exclude user prompts, chain-of-thought, model output, API keys, and itinerary content. Parallel overlap is calculated from observed node spans and is not presented as a dependency-graph critical path. Per-run LLM usage is request-context attribution; `usage_exact_ratio` distinguishes provider usage from estimates. For a real deployment, export these counters to the service's existing metrics system and add a durable trace store before enabling multi-instance horizontal scaling.

## CI

GitHub Actions compiles all Python sources, validates the 30-case evaluation manifest without network access, and runs the complete deterministic test suite without requiring external LLM or map API keys. Online-provider evaluation remains opt-in, requires explicit numeric call budgets plus `--allow-external-calls`, and should run separately with repository secrets. `python scripts/validate_eval_assets.py` validates the manifest for free; add `--require-fixtures` when generated fixture completeness must be a hard gate.
