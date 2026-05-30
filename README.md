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

## Competitive landscape — honest comparisons

The result table above used a *naive* RAG baseline (top-K cortex
cosine on the substrate's own vectors, no separate embedding model).
Several real-world competitors weren't directly tested. Where they
sit, honestly:

### vs. Claude / ChatGPT Memory (sliding-context injection)

Both Claude and ChatGPT have "memory" features that detect noteworthy
facts during conversation and inject them into future system prompts.
This **works fine for ~30–100 facts per user** — the sliding-context
budget. The honest comparison:

- **At ≤100 facts per user**: their memory wins on simplicity. Ada is
  overkill.
- **At >1,000 facts per user**: their memory truncates or summarises,
  losing precision. Ada's substrate stays addressable.
- **At >10,000 facts per user** (the eventual fate of any deployed
  assistant): the LLM Memory feature is the same structural bind as
  full-context — facts evicted from the budget can't be queried. Ada
  doesn't have an eviction budget.
- **Versioning**: they don't expose drift trajectory queryable by key.
  Ada does.
- **Audit**: they don't expose which facts contributed to which answer.
  Ada does.
- **Cross-session continuity across model swaps**: their memory is
  bound to their account/model. Ada is bipolar HDC on your disk.

### vs. Neo4j / knowledge graphs (Cypher / SPARQL)

A knowledge graph with a hand-designed schema would likely **handle
queries A–E well** — Cypher is purpose-built for `MATCH` +
aggregation + multi-hop traversal. We have not yet run a head-to-head
benchmark against Neo4j; **this is the most defensible comparison we
should add next**. Honest expectations:

- **Queries A, B (counts)**: Neo4j likely wins on latency
  (`MATCH (p:Person)-[:WORKS_AS]->(j:Job {name: 'engineer'}) RETURN count(*)`).
  Both correct.
- **Query C, D (distribution)**: Neo4j wins again — `GROUP BY` is
  native. Both correct.
- **Query E (multi-role intersection)**: Neo4j wins with native
  `MATCH` + multiple `WHERE`. Both correct.

Where Neo4j *loses* against Ada (so the comparison still favours Ada
on architectural grounds):

- **Schema design cost**: Neo4j requires hand-designing node and edge
  types per domain. Ada's universal schema is fixed once, reused
  forever, and the LLM (Haiku at build time) does the mapping.
- **Versioning**: Neo4j doesn't natively version concepts with delta
  vectors. Adding a temporal layer to a Cypher schema is real work.
- **Cross-model substrate portability**: A Neo4j database is bound to
  its schema; rewriting it under a new ontology is a migration. Ada's
  2048-dim bipolar substrate doesn't care which LLM consumes it.
- **Compositional algebra**: HDC's bind/bundle/unbind operations let
  Ada *bind* `(perceptual.color, blue)` and search role-level
  similarity natively. Neo4j has no comparable primitive.

**Bottom line: Neo4j wins on aggregation latency; Ada wins on
not-needing-hand-schema and on versioning + audit + compositional
primitives. They're complementary, not redundant.** A future paper
revision should include a Neo4j baseline.

### vs. pgvector / Chroma / Pinecone (production vector DBs)

The RAG baseline we ran used Ada's own cortex cosine for top-K
retrieval — *not* a separate transformer-embedding vector DB. We
**have not yet directly benchmarked against pgvector + sentence-
transformers**. Honest expectation:

A real vector DB with a real embedding model would likely **beat our
naive RAG baseline on retrieval quality for queries A–D** (more
semantically-tuned embeddings would surface more relevant facts in the
top-K). But it would **still score 0/5 on aggregation and
distribution queries** because:

- Top-K returns at most K facts.
- The count answer ("71 engineers exist") is **not a fact in any
  retrievable form** — it's a *property of the whole corpus*.
- Distribution ("top 5 cities") **requires the full distribution**, not
  a sample. Top-K returns a biased sample.

This is the structural argument repeating: RAG is the right tool for
single-entity retrieval; *no implementation of RAG fixes aggregation*
because aggregation isn't top-K. Adding pgvector to the bench would
strengthen the report but not change the conclusion.

**Bottom line: pgvector vs Ada is the same shape as our naive RAG vs
Ada — pgvector improves retrieval *quality*, but the failure category
on aggregation is implementation-independent.** Adding it is a
defensiveness move, not a research move.

### Summary table

| System | Wins at | Loses at |
|---|---|---|
| **Ada** | aggregation, distribution, multi-role intersection at scale; versioning; audit; on-device | small-corpus retrieval accuracy, semantic synonymy |
| **Full-context LLM** | small-corpus Q&A, synthesis | corpus > context window (structurally) |
| **Claude/ChatGPT Memory** | small per-user fact budget (~30-100), zero-config | beyond eviction budget, no aggregation, no drift |
| **Neo4j / KG** | aggregation latency, multi-hop traversal, typed relations | requires hand schema, no native versioning, no compositional algebra |
| **pgvector / RAG** | semantic single-entity retrieval at scale | aggregation, distribution (top-K can't contain a count) |

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
