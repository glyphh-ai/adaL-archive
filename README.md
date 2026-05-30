# Ada

**A hyperdimensional cognitive substrate that answers compositional queries
on corpora exceeding any LLM context window.**

At 100,000 facts / 3.6M tokens (18× Claude Haiku/Opus, 3.6× Sonnet),
Ada answers five compositional queries correctly in under 200 ms each
with full provenance. A full-context LLM scores 0/5 because the corpus
doesn't fit in any prompt. Naive RAG scores 0/5 because top-K retrieval
is the wrong shape for aggregation. Read the paper:
[`benchmark/paper.md`](benchmark/paper.md).

> *"Why not just RAG?"* — RAG handles **retrieval**. Ada handles
> **aggregation, distribution, and multi-role intersection** — query
> shapes RAG structurally cannot answer because the answer is not a
> single fact in the substrate, it's a *property of the whole substrate*.

## The headline result

| | Score | Latency | Why |
|---|---|---|---|
| **Ada (substrate)** | **5/5 ✓** | 102–177 ms | Universal-schema role scan; aggregation is `Counter().most_common()` |
| **LLM-only (full context)** | **0/5 ✗** | N/A | Corpus is 3.6–18× every public Claude window — bytes don't fit |
| **LLM + naive RAG** (top-50) | **0/5 ✗** | ~1 s | Top-K can't contain a *count*; aggregation isn't a retrieval shape |

Reproducible from this repo: `ANTHROPIC_API_KEY=... PYTHONPATH=. python curriculum/at-scale/bench_rag.py --n 5000` (~30 minutes; details in [`curriculum/at-scale/README.md`](curriculum/at-scale/README.md)).

## What it actually is

Four pieces, each independently inspectable:

1. **HDC substrate** — 2048-dim bipolar vectors. Atoms encode words via
   positional character n-grams (no embedding model). Bindings compose
   role-filler frames; bundling aggregates. Standard SPA-style algebra.
   `ada/memory/`, `ada/core/`.

2. **Universal schema** — a fixed 7-layer × 33-role lattice that every
   fact maps into. No schema sprawl. Counting matches on
   `perceptual.color = blue` is a one-line dict scan; top-5 cities is
   `Counter().most_common(5)`. `ada/cognitive/universal.py`.

3. **Programmatic ingestion** — `space.tell_raw(facts, key)` takes
   pre-structured universal-schema slots and writes a glyph with **zero
   LLM calls**. The path used by the curriculum loaders and the
   at-scale benchmark. Versioned: same `key` becomes v2, v3, ...,
   with `TemporalEncoder` delta vectors and `drift(key)` trajectories.
   `ada/memory/thought_space.py`.

4. **Curriculum** — YAML files that declare every concept as a
   universal-schema fill plus `composed_of:` references to earlier
   concepts. Layer-0 is the 63 NSM semantic primes (Wierzbicka). Layer-1
   is 59 concepts (Berlin & Kay colors, numbers, spatial, logical,
   temporal) each citing its layer-0 primes. The composition lattice
   is recursive: any concept can be traced back to its primes.
   `curriculum/`.

## MCP surface

Four tools an LLM (Claude, Claude Code, any MCP-capable model) can call:

| Tool | What it does |
|---|---|
| `think(input)` | Pure associative recall — surfaces what activates. No interpretation. |
| `ask(question)` | Predicate-aware targeted retrieval. Returns one fact + confidence, or "I don't know." |
| `tell(text)` | Natural-language ingestion. Optional LLM enrichment at build time. |
| `tell_raw(facts)` | Programmatic ingestion. Zero LLM in the path. |

`domains/mcp/server.py`.

## Run it

### Quick: the interactive REPL

```bash
pip install numpy pyyaml pydantic pydantic-settings transformers torch anthropic
export ANTHROPIC_API_KEY=sk-ant-...   # optional; substrate works without
PYTHONPATH=. python scripts/ada_repl.py
```

Inside:
```
  ada> aurora                        # load 77-fact Project Aurora corpus
  ada> ask what hull does aurora use?
  → trimaran                         (grounded in: "The Aurora boat uses a trimaran hull design.")
  ada> history aurora.hull
  v1: single hull design  →  v2: catamaran  →  v3: trimaran
  ada> drift aurora.hull
  consecutive sims: [+0.812, +0.829]  drift rate: +0.306
  ada> tell key=chris.location Chris just moved to Brooklyn.
  absorbed v2 ✓
```

### The headline benchmark

```bash
# Three-way scoreboard (Ada / LLM-only / RAG) at 3.6M tokens
ANTHROPIC_API_KEY=... PYTHONPATH=. python curriculum/at-scale/bench_rag.py --n 5000

# Substrate-only timing (no API needed)
PYTHONPATH=. python curriculum/at-scale/bench.py --n 1000
```

### The curriculum

```bash
# Load 63 NSM primes (layer-0)
PYTHONPATH=. python curriculum/agi/layer-0/teach.py

# Load layer-0 + layer-1 (~122 concepts + 120 composition facts)
PYTHONPATH=. python curriculum/agi/layer-1/teach.py

# Five demos of substrate-native operations no LLM does the same way
PYTHONPATH=. python curriculum/agi/layer-1/explore.py
```

## When Ada is and isn't the right tool

| Use case | Right tool |
|---|---|
| Cross-session memory beyond context window | **Ada** |
| Aggregation / distribution / multi-role queries on >1M-token corpora | **Ada** |
| Versioned beliefs with drift trajectories (`history(key)`, `drift(key)`) | **Ada** |
| Audit trail of every learned schema element | **Ada** (markdown changelog) |
| Privacy-first deployment (no embedding API) | **Ada** (deterministic char n-grams) |
| Federated memory (merge across users) | **Ada** (HDC majority-bundle) |
| Compositional understanding by traversing primitives | **Ada** (curriculum + composition lattice) |
| Q&A over a small fixed corpus that fits in context | full-context LLM |
| Pure semantic similarity over documents | transformer embeddings + ANN |
| Strict typed-relationship reasoning (legal, compliance) | knowledge graph + Cypher/SPARQL |

The honest summary: **Ada is not a general-purpose retrieval winner**.
On a 77-fact corpus that fits in any LLM's window, a full-context LLM
beats Ada 96–72 on accuracy. Ada's contribution is the **specific
shape of queries the other architectures structurally cannot answer**.

## Architecture map

```
ada/
  core/                      HDC primitives (Vector, ops: bind, bundle, cosine)
  cognitive/
    universal.py             7-layer × 33-role universal schema
    surface.py               think / ask cognitive surfaces
    generate.py              LLM-as-renderer
    loop.py                  9-stage cognitive loop
  encoder/
    llm_enricher.py          LLM-at-build-time fact enrichment
    universal-schema adapter projects mapping → StructuredFact
  memory/
    atom.py                  2048-dim bipolar atoms
    binding.py               role-filler frames
    thought_space.py         tell_raw / absorb / recall + versioning
    glyph_dream.py           REM + slow-wave dream loops
    thought_glyph.py         5-layer hierarchical encoder
  temporal/
    delta.py                 compute / apply version-delta vectors
    predictor.py             drift statistics, beam search
  skill/
    schema_evolution.py      SchemaTracker (markdown + meta-glyphs)
  tui/                       CLI / REPL
  llm/                       local LLM backends (MLX, llama-cpp, API)

domains/
  brain/                     central think() pipeline
  mcp/server.py              MCP server: think / ask / tell / tell_raw
  query/, models/, nl_query/, listeners/, capabilities/

api/                         FastAPI HTTP routes
infrastructure/              config + db
shared/                      auth, exceptions, utilities

benchmark/
  paper.md                   the writeup — at-scale empirical claim
  aurora_versioned.py        77-fact / 25-query benchmark (= hard_test)
  llm_only_baseline.py       LLM with full corpus in context (small-scale only)

curriculum/
  README.md                  what curricula are
  agi/
    layer-0/                 63 NSM semantic primes
    layer-1/                 59 concepts + 120 composition facts
                             (colors, numbers, spatial, logical, temporal)
  at-scale/
    generate.py              deterministic synthetic corpus
    bench.py                 Ada-only timing
    bench_rag.py             three-way: Ada / LLM-only / RAG

scripts/
  ada_repl.py                interactive REPL
  hard_test.py               77-fact Aurora benchmark
  heldout_test.py            generalization test
  versioning_demo.py         absorb(key=…) → history → drift
  schema_demo.py             SchemaTracker over Alice corpus
  think_vs_ask.py            side-by-side cognitive ops
  multihop_demo.py           "what are Chris's children's names?"

docs/
  schema_history.md          auto-generated schema changelog (per-run)
```

## HTTP / MCP endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness probe |
| `POST /{org_id}/{model_id}/mcp` | MCP tool calls: `think` / `ask` / `tell` / `tell_raw` |
| `POST /{org_id}/{model_id}/listener` | async data ingestion |
| `POST /{org_id}/tokens` | API token management |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Enables Haiku-backed enricher + Claude as renderer in MCP `ask` |
| `ADA_MODEL` | `claude-haiku-4-5-20251001` | LLM model for enrichment / rendering |
| `DATABASE_URL` | SQLite | PostgreSQL + pgvector for production |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8002` | Server port |
| `JWT_SECRET_KEY` | — | JWT validation secret |
| `ENABLE_DOCS` | `false` | Enable /docs and /redoc |

## The paper

Read [`benchmark/paper.md`](benchmark/paper.md) for the full
architectural contribution and benchmark results. The honest claim:
**at corpus sizes that exceed an LLM context window — which is the
regime any production memory system enters within months of use — Ada
answers compositional queries that the LLM cannot see and that naive
RAG, by construction, cannot aggregate.**

## License

MIT
