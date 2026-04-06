# Ada Runtime

Hyperdimensional Computing runtime for vector-based knowledge models. Exposes MCP and REST endpoints for natural language and GQL queries, data ingestion via listeners, and model management.

Clean port of `glyphh-runtime` with no runtime-level authentication or metering — MCP and listener endpoints still require Bearer tokens.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Start the server
python main.py
```

Or with Docker:

```bash
docker compose up
```

## Architecture

```
ada/              # Core SDK — HDC encoding, GQL, similarity, memory
api/              # FastAPI routes (health, tokens, listeners, org-scoped)
domains/          # Business logic (auth, MCP, models, query, listeners)
infrastructure/   # Config (Pydantic settings), database (SQLAlchemy async)
shared/           # Auth middleware, exceptions, utilities
migrations/       # Alembic schema migrations
```

## Endpoints

- `GET /health` — liveness probe
- `POST /{org_id}/{model_id}/mcp` — MCP tool calls (nl_query, gql_query, etc.)
- `POST /{org_id}/{model_id}/listener` — async data ingestion
- `POST /{org_id}/tokens` — API token management

## Configuration

All settings via environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | SQLite (local) | PostgreSQL + pgvector for production |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8002` | Server port |
| `JWT_SECRET_KEY` | — | Shared secret for local JWT validation |
| `ENABLE_DOCS` | `false` | Enable /docs and /redoc |

## License

MIT
