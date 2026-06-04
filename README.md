# Ada

**A hyperdimensional cognitive substrate that gives an LLM persistent,
queryable memory beyond any context window.**

Ada stores facts as 2048-dim bipolar vectors and answers *compositional*
queries over them — aggregation, distribution, multi-role intersection,
versioned drift — in well under a second, with full provenance, on corpora
far larger than any prompt can hold. It runs on-device (SQLite, no embedding
API) or against Postgres + pgvector for scale, and exposes itself to any
LLM over MCP.

This README is about **running and using** Ada. For the empirical argument
(why aggregation queries beat RAG and full-context LLMs at scale), see the
paper: [`benchmark/paper.md`](benchmark/paper.md).

---

## How it works

Four pieces, each independently inspectable:

1. **HDC substrate** — 2048-dim bipolar vectors. Words are encoded by
   positional character n-grams (deterministic, no embedding model).
   `bind` composes role–filler frames; `bundle` aggregates. `ada/core/`,
   `ada/memory/`.

2. **Universal schema** — a fixed 7-layer × 33-role lattice every fact maps
   into (entity, perceptual, spatial, temporal, relational, quantitative,
   epistemic). Counting "facts where color = blue" is a one-line scan; "top
   5 cities" is `Counter().most_common(5)`. `ada/cognitive/universal.py`.

3. **Versioned memory** — write a fact under a `key` and it becomes v1, v2,
   v3… with delta vectors between versions. You can query its full
   `history` and its `drift` trajectory over time. `ada/memory/thought_space.py`.

4. **Storage** — SQLite by default (`~/.ada/ada.db`, zero setup). Point
   `DATABASE_URL` at Postgres and the runtime switches to the **pgvector**
   backend: vectors stored once, reconstructed on boot (no re-encoding),
   HNSW cosine recall at scale.

The substrate runs fully without an LLM. An Anthropic API key is optional —
it enables the build-time enricher (`tell`) and Claude as the answer renderer
(`ask`).

---

## Install & start

### Quickest: `make`

```bash
make install        # .venv + editable install (+ dev extras), seeds .env
# add ANTHROPIC_API_KEY to .env  (optional — substrate works without it)
make dev            # server with hot-reload on http://localhost:8002
```

Other targets: `make repl` (interactive shell), `make serve` (no reload),
`make test`, `make fmt`, `make clean`. Run `make` alone to list them.

### The interactive REPL

```bash
make repl           # or, after install:  ada
```

```
  ada> aurora                        # load the 77-fact demo corpus
  ada> ask what hull does aurora use?
  → trimaran                         (grounded in: "The Aurora boat uses a trimaran hull design.")
  ada> tell key=chris.location Chris just moved to Brooklyn.
  absorbed v2 ✓
  ada> history chris.location
  v1: San Francisco  →  v2: Brooklyn
  ada> drift aurora.hull
  consecutive sims: [+0.812, +0.829]  drift rate: +0.306
  ada> stats
  thoughts: 77   versioned keys: 12   foundation primitives: 122
```

Type `help` in the shell for the full command list.

---

## Production: Docker

The image is storage-agnostic — `DATABASE_URL` decides the backend. Unset, Ada
runs on in-process SQLite. Point it at Postgres and the runtime flips to the
**pgvector** backend, enabling the vector extension and running the alembic
migrations (incl. the `ada_thoughts` cortex column + HNSW index) on first boot.

### Recommended: compose (runtime + pgvector)

```bash
cp .env.example .env          # set ANTHROPIC_API_KEY — DB is wired by compose
docker compose up -d --build
curl localhost:8002/health
```

The bundled `db` service is `pgvector/pgvector:pg16`; the runtime waits for it
to be healthy, connects via `DATABASE_URL=postgresql://ada:ada@db:5432/ada`,
and persists vectors to the `pgdata` named volume across restarts and upgrades.

### Single container (SQLite, no Postgres)

```bash
docker build -t ada .
docker run -d -p 8002:8002 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -v ada-data:/root/.ada \
  ada
```

Leaving `DATABASE_URL` unset keeps SQLite at `~/.ada/ada.db`; mount a volume
there to persist the substrate.

---

## Using it from an LLM (MCP)

Ada exposes a single MCP endpoint at **`/mcp`**. Start the server, then point
any MCP client at `http://localhost:8002/mcp`.

### The tools

| Tool | What it does |
|---|---|
| `think(input, top_k)` | Pure associative recall — surfaces the thoughts that resonate. No answer, just context. |
| `ask(question)` | Targeted retrieval. Returns one fact + confidence, or "I don't know." |
| `tell(text, key?)` | Natural-language ingestion. With `key`, becomes a new version of that key. |
| `tell_raw(facts, key?)` | Programmatic ingestion of a universal-schema slot fill. **No LLM in the path.** |
| `recall(query, top_k)` | Deterministic HDC cosine search. Top matches + similarity + versioned key. |
| `history(key)` | Full version chain for a key, chronological. |
| `drift(key)` | How a key's representation shifted: consecutive sims, drift rate. |
| `predict(key, steps)` | Project a key forward via beam search over its temporal deltas. |
| `stats()` | Substrate vital signs: thoughts, versioned keys, foundation primitives. |
| `create_token(...)` | Mint an API token for this MCP server (returned once). |

`domains/mcp/server.py`.

### Connecting clients

```bash
ada setup claude            # auto-configure Claude Code
ada setup                   # print config snippets for Claude Desktop / Cursor / any MCP client
```

For authenticated access, mint a token:

```bash
ada token create --name my-client --permissions read,write
ada token list
ada token revoke <id-or-prefix>
```

### HTTP endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness probe |
| `POST /mcp` | MCP tool calls (Streamable HTTP) |
| `POST /{org_id}/tokens` | API token management |

---

## Configuration

All values have defaults — the server boots with an empty `.env`. The `.env`
is read by both the server and the `ada` REPL; secrets can also be stored
encrypted in `~/.ada/vault` (`ada setup key`).

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Enables the `tell` enricher + Claude as `ask` renderer |
| `ADA_MODEL` | `claude-haiku-4-5-20251001` | LLM model for enrichment / rendering |
| `DATABASE_URL` | SQLite at `~/.ada/ada.db` | `postgresql://…` switches to the pgvector backend |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8002` | Server port |
| `JWT_SECRET_KEY` | — | JWT validation secret (token auth) |
| `ENABLE_DOCS` | `false` | Enable `/docs` and `/redoc` |

For the code-intelligence capability (`capabilities/code/`), the client reads
`ADAL_RUNTIME_URL`, `ADAL_ORG_ID`, `ADAL_MODEL_ID`, and `ADAL_TOKEN`.

---

## Architecture map

```
ada/
  core/                HDC primitives (Vector; bind, bundle, cosine)
  cognitive/
    universal.py       7-layer × 33-role universal schema
    surface.py         think / ask cognitive surfaces
    loop.py            cognitive loop
  memory/
    atom.py            2048-dim bipolar atoms
    binding.py         role–filler frames
    thought_space.py   tell_raw / absorb / recall + versioning
    foundation.py      pre-encoded primitive warm-start (~56ms boot)
  temporal/            version deltas, drift statistics, beam-search predictor
  market/              glyph-native HDC market forecaster
  tui/                 CLI / REPL  (the `ada` command)
  llm/                 LLM backends (API, MLX, llama-cpp)

domains/
  brain/               central think() pipeline
  mcp/server.py        MCP tools: think / ask / tell / tell_raw / recall /
                       history / drift / predict / stats / create_token

api/                   FastAPI HTTP routes (health, tokens)
infrastructure/        config + database (SQLite / pgvector)
shared/                auth, exceptions, middleware

curriculum/            YAML concept curricula (NSM primes → composed concepts)
benchmark/             paper.md + the at-scale benchmarks
scripts/               ada_repl.py and standalone demos
```

---

## The paper

[`benchmark/paper.md`](benchmark/paper.md) makes the empirical case: at corpus
sizes that exceed an LLM context window — the regime any production memory
system enters within months — Ada answers compositional queries (counts,
distributions, multi-role intersections) that a full-context LLM can't see and
that top-K RAG, by construction, can't aggregate.

---

## License

See [`LICENSE`](LICENSE) — Glyphh AI Community License. Copyright © 2026
Glyphh AI LLC.
