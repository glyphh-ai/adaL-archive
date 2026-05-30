# Ada

**A cognitive substrate for LLMs.** Persistent memory with versioned beliefs,
schema that grows as facts arrive, and a self-describing audit trail.

> *"Why not just RAG?"* — see [`benchmark/paper.md`](benchmark/paper.md). Short
> version: at small scale a full-context LLM beats Ada on retrieval (96% vs
> 72% on our hard test). Ada wins on three things RAG/KG can't do at any
> scale: **versioned-belief drift**, **substrate self-description with a
> markdown changelog**, and **cognitive continuity across model versions**.

## What it is

Ada is a hierarchical hyperdimensional-computing (HDC) substrate wrapped
in four layers:

1. **The HDC substrate** — 2048-dim bipolar vectors. Atoms encode words via
   positional character n-grams (no embedding model needed). Bindings
   compose role-filler frames; bundling aggregates. Standard SPA-style
   algebra. (`ada/memory/`)
2. **LLM-at-build-time enrichment** — one Claude Haiku call per fact,
   cached. Yields a structured `(subject, predicate, object, topic)`
   tuple, bound into the glyph. Recall gets a +1.0 boost on predicate
   match — the lift that makes "Alice's age" stop confusing with
   "Alice's color." (`ada/encoder/llm_enricher.py`)
3. **Versioned-concept memory** — `absorb(text, key=...)` makes the
   write a new version of `key`. The `TemporalEncoder` computes a delta
   vector to the prior version. Queryable via `history(key)` and
   `drift(key)`. Thought IDs follow `key@iso_timestamp#v{N}`. (`ada/temporal/`,
   `ada/memory/thought_space.py`)
4. **Schema evolution + audit** — `SchemaTracker` watches every absorb. On
   the first sight of a new predicate or topic it (a) appends a
   human-readable entry to `docs/schema_history.md` and (b) absorbs a
   meta-glyph back into the substrate so Ada can recall her own schema.
   (`ada/skill/schema_evolution.py`)

The MCP layer (`domains/mcp/server.py`) exposes three tools an LLM can
call: `think`, `ask`, and `write`.

## Try it in the REPL

```bash
# Install
pip install numpy pyyaml pydantic pydantic-settings transformers torch anthropic

# (Optional but recommended) — Haiku-backed enrichment + Claude renderer:
export ANTHROPIC_API_KEY=sk-ant-...

# Without an API key, Ada falls back to a heuristic enricher + Qwen3-0.6B
# local renderer. Download Qwen3 first:
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen3-0.6B', local_dir='/tmp/qwen3-0.6b', \
  allow_patterns=['*.json','*.safetensors','tokenizer*','*.txt'])"

# Launch the REPL
PYTHONPATH=. python scripts/ada_repl.py
```

Inside the REPL:

```
  ada> aurora
  loaded 77/77 Aurora facts.

  ada> ask what hull does aurora use?
  The Aurora boat currently uses a trimaran hull design.
  grounded in: The Aurora boat uses a trimaran hull design.

  ada> history aurora.hull
  v1: The Aurora boat uses a single hull design.
  v2: The Aurora boat uses a catamaran hull design.
  v3: The Aurora boat uses a trimaran hull design.

  ada> drift aurora.hull
  versions: 3
  consecutive sims: [+0.812, +0.829]
  first→last sim  : +0.694
  drift rate      : +0.306

  ada> think project aurora
  6 thoughts activated
    [+1.69]  Project Aurora partnered with Sony for image sensors.
    [+1.65]  Project Aurora targets the recreational sailing market.
    [+1.55]  Project Aurora's budget was increased to 350,000 dollars.
    ...

  ada> schema
  predicates known: 41
  topics known    : 37

  ada> ask what is Aurora's CTO's name?
  I don't know.  (conf=0.18)

  ada> write key=chris.location Chris just moved to Brooklyn.
  absorbed v2: chris.location@20260530T053012Z#v2

  ada> ask where does chris live?
  Chris lives in Brooklyn.
```

## Try the benchmarks

```bash
# Held-out generalization (95% / 100% on facts the enricher never parsed)
PYTHONPATH=. python scripts/heldout_test.py

# The hard test — 77 facts, 25 queries across six categories
PYTHONPATH=. python scripts/hard_test.py

# Baseline: same 25 queries answered by Claude with all 77 facts in context
ANTHROPIC_API_KEY=... PYTHONPATH=. python benchmark/llm_only_baseline.py

# Demos of the individual primitives
PYTHONPATH=. python scripts/versioning_demo.py
PYTHONPATH=. python scripts/schema_demo.py
PYTHONPATH=. python scripts/think_vs_ask.py
PYTHONPATH=. python scripts/multihop_demo.py
```

## Architecture map

```
ada/
  core/                HDC primitives (Vector, ops: bind, bundle, cosine)
  encoder/             encoders (character, projection, intent)
    llm_enricher.py    LLM-at-build-time fact enrichment ← NEW
  memory/
    atom.py            2048-dim bipolar atoms
    binding.py         role-filler frames
    thought_space.py   absorb / recall + versioning ← UPDATED
    glyph_dream.py     REM + slow-wave dream loops
    thought_glyph.py   5-layer hierarchical encoder
  temporal/
    delta.py           compute / apply version-delta vectors
    predictor.py       drift statistics, beam search
  cognitive/
    loop.py            9-stage cognitive loop
    generate.py        LLM-as-renderer for substrate context
    surface.py         think + ask cognitive surfaces ← NEW
  skill/
    schema_evolution.py SchemaTracker ← NEW
  tui/                 CLI / REPL
  llm/                 local LLM backends (MLX, llama-cpp, API)
domains/
  brain/               the central think() pipeline
  mcp/                 MCP server exposing think / ask / write ← UPDATED
  query/, models/, nl_query/, listeners/, capabilities/
api/                   FastAPI routes
infrastructure/        config + db
shared/                auth, exceptions, utilities

benchmark/
  paper.md             the writeup ← NEW
  aurora_versioned.py  77-fact, 25-query benchmark (run via hard_test.py)
  llm_only_baseline.py same questions, LLM with full corpus

scripts/
  ada_repl.py          interactive REPL ← NEW
  hard_test.py         Aurora benchmark
  heldout_test.py      generalization test
  versioning_demo.py
  schema_demo.py
  think_vs_ask.py
  multihop_demo.py
  generate_demo.py
  generate_demo_100.py

docs/
  schema_history.md    auto-generated schema changelog
```

## Endpoints (HTTP / MCP)

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness probe |
| `POST /{org_id}/{model_id}/mcp` | MCP tool calls: `think`, `ask`, `write` |
| `POST /{org_id}/{model_id}/listener` | async data ingestion |
| `POST /{org_id}/tokens` | API token management |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Enables Haiku-backed enricher + Claude renderer |
| `ADA_MODEL` | `claude-haiku-4-5-20251001` | LLM model for enrichment / rendering |
| `DATABASE_URL` | SQLite | PostgreSQL + pgvector for production |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8002` | Server port |
| `JWT_SECRET_KEY` | — | JWT validation secret |
| `ENABLE_DOCS` | `false` | Enable /docs and /redoc |

## The paper

Read [`benchmark/paper.md`](benchmark/paper.md) for the architectural
contribution and benchmark results. The honest summary: Ada is not the
right tool when an LLM's context window can hold your whole corpus. It is
the right tool when (a) it can't, (b) you need versioned trajectories on
beliefs, (c) you need a substrate that survives model swaps, or (d) you
need an auditable record of every structural concept the system learned.

## License

MIT
