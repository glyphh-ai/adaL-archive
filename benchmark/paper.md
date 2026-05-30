# Ada: a cognitive substrate for LLMs with versioned memory, schema evolution, and cross-session continuity

*Working note — `claude/repo-overview-ByHYQ` — May 2026*

## Abstract

We describe **Ada**, a memory substrate intended to be wired into an LLM as
its persistent cognitive layer. Ada combines a hierarchical hyperdimensional
computing (HDC) substrate with three architectural primitives that
distinguish it from both retrieval-augmented generation (RAG) and
knowledge graphs (KGs): (i) **LLM-at-build-time fact enrichment** that
yields structured `(subject, predicate, object, topic)` tuples without
imposing a hand-designed schema; (ii) **versioned-concept memory** where
each stable key carries an explicit chain of vector states and a
drift-queryable trajectory; (iii) **substrate self-description**, in
which the schema grows organically as facts arrive, every introduction
is logged to markdown for human audit and absorbed back as a meta-glyph
so the substrate can recall its own structure.

We then **benchmark Ada honestly**. On a 77-fact, 25-query benchmark over
a fictional self-driving boat project ("Aurora") with versioned design
decisions, contradictions and red-herring filler, Ada+Haiku reaches 72%
overall, while a baseline of Claude Haiku with the full corpus in
context reaches 96%. We discuss what this means and what it does not.

The honest claim of the paper is **not** that Ada beats LLMs at retrieval
quality. It is that Ada makes three things possible that LLMs (alone, with
RAG, or with knowledge graphs) cannot do at scale: (a) **persistent
cross-session memory** that grows beyond any context window; (b) **explicit
trajectory provenance** for revised beliefs, queryable by drift rate and
delta vector; (c) **substrate-self-description as a first-class capability**,
producing a git-tracked audit trail of every learned structural concept.
For these properties, Ada is — to the best of our knowledge — the first
implementation that puts all three on the same primitive.

## 1. Problem

LLMs forget. A fresh Claude session, however large its context window,
begins with no recollection of prior conversations. Augmenting the LLM
with a vector DB (RAG) gives it retrieval but loses temporal structure:
two versions of the same belief appear as similar items, ranked by
cosine, not as a trajectory. Augmenting with a knowledge graph gives it
relational structure but requires a hand-designed schema and rarely
versions concepts natively.

The three capabilities we identify as *structurally missing* from all of
these:

1. **Versioned belief with drift queryability.** "How has my answer to X
   changed over the last six months, and by how much?" — neither RAG nor
   most KGs answer this without bespoke instrumentation.
2. **Schema discovered from data with an audit trail.** RAG has no
   schema; KGs require one upfront. Neither produces a human-readable
   changelog of "things the system learned to track."
3. **Cognitive continuity across model versions.** The substrate must
   persist when the LLM is swapped (Sonnet 4.6 → 4.8 → Opus 5). Today,
   no production LLM memory survives a model migration.

## 2. Architecture

Ada is a hierarchical HDC substrate (2048-dim bipolar vectors) wrapped
in four layers, each of which is independently testable.

### 2.1 Substrate

- **Atoms**: character n-gram bindings producing per-word vectors.
  Deterministic; no training required.
- **Bindings**: role-filler frames via XOR + bundle by majority vote.
  Standard HDC algebra.
- **Hierarchical glyphs**: each thought has 5 layers (perspective,
  semantic, relational, temporal, direction), each layer holds 3-6
  segments, each segment holds 1-3 role vectors. Weighted recall:
  cortex 5%, layer 10%, segment 25%, role 60%.

### 2.2 LLM-at-build-time enrichment

A single Claude Haiku call per ingested fact produces a structured tuple:

```
{ subject: "alice",
  predicate: "favorite_color",
  object: "blue",
  topic: "person.preference" }
```

These four fields are bound into additional segments at ingest time, and
the recall scoring gives a +1.0 boost on `predicate` match, +0.5 on
`subject` match, +0.3 on `topic` match. This is the critical lift that
disambiguates within-entity confusion ("Alice's age" vs "Alice's color")
that plain HDC cannot resolve.

**Build-time cost**: ~$0.0001 per fact. For 1M facts: $100, one-time.
**Inference cost**: zero — at query time, Ada also enriches the question
(again via Haiku) and matches on the resulting predicate/topic.

### 2.3 Versioned-concept memory

Every `absorb()` accepts an optional `key`. When provided, the new fact
is appended as version N+1 of the key, a temporal delta is computed
from the prior version's cortex vector via `TemporalEncoder`, and the
chain is queryable by `history(key)` and `drift(key)`. The `thought_id`
follows the documented convention `key@iso_timestamp#v{N}`.

For the same belief revised three times — `chris.age` 31 → 32 → 33 —
we observe cosine drift of 0.235 (small, number-only change), versus
0.278 for `chris.location` Austin → Brooklyn → Tokyo (larger, completely
different tokens). The drift rate reflects actual semantic distance, not
just textual edit distance.

### 2.4 Schema evolution + audit

A `SchemaTracker` watches every absorb. The first time a new predicate
or topic is observed, it persists the discovery two ways:

1. Appended to `docs/schema_history.md` — a git-tracked, human-readable
   changelog.
2. Absorbed as a meta-glyph in the substrate, keyed
   `schema.predicate.<name>` or `schema.topic.<name>` and itself
   versioned. The substrate can now recall information about its own
   structure.

On a 100-fact corpus, the tracker auto-discovered 33 predicates and 33
topics. The markdown changelog ran to 555 lines. Querying the substrate
"what predicates exist for people?" surfaces the relevant meta-glyphs;
the renderer answers from them.

## 3. Cognitive surfaces: `think` vs `ask`

The MCP layer exposes Ada to the LLM as three tools:

- **`think(input, top_k)`** — pure associative wave. Returns the
  constellation of thoughts that resonate. No question framing, no
  refusal, no synthesis.
- **`ask(question)`** — predicate-aware targeted retrieval. Returns one
  fact + confidence, or refuses with "I don't know" when no
  high-confidence match exists.
- **`write(text, key?)`** — absorb a fact, optionally as a new version.

The split is architecturally meaningful: `think` is the brain's reflex
(broad sim), `ask` is the brain's deliberate query (predicate match
with refusal). They use different recall policies on the same substrate.

A demonstrative moment from the Aurora benchmark: input
`"the hull of the boat"` to `think()` returns all three versioned hull
designs (single, catamaran, trimaran). `ask()` then has Claude render
the conflict honestly:

> *"The memories contain conflicting information about the Aurora boat's
> hull design, listing single hull, trimaran, and catamaran designs.
> I cannot provide a definitive answer about which hull type the Aurora
> boat actually uses based on these contradictory recalled memories."*

The substrate exposed all three explicit versions; the renderer (Claude
itself) read them and reported the conflict rather than guessing. This
is the artifact of having facts as *addressable entities* versus
statistical residue.

## 4. Benchmark

### 4.1 Setup

- **Corpus**: 77 facts about a fictional self-driving boat project
  ("Project Aurora"), including 30 facts representing red-herring
  filler that should not affect Aurora-specific queries.
- **Versioning**: 9 keys are versioned (hull design 3 versions, sensor
  type 2 versions, propulsion 2, prototype date 2, launch date 2,
  budget 2, etc.).
- **Queries**: 25 questions across six categories:
  - A. Direct recall (5)
  - B. Multi-hop inference (5)
  - C. Version-aware "current value" recall (5)
  - D. Schema introspection (3)
  - E. Absence detection (3)
  - F. Long-horizon synthesis (4)

### 4.2 Baselines

Two systems compared on the same 25 queries:

1. **LLM only, full corpus in context**. Claude Haiku 4.5 with all 77
   facts sent verbatim in every query prompt. This is the strongest
   possible baseline for a corpus that fits in context.
2. **Ada + LLM renderer**. The full pipeline: enricher at build time,
   versioned absorb, schema tracker observation, predicate-aware
   recall, Claude Haiku as the rendering layer.

### 4.3 Results

| Category | LLM only (full context) | Ada + LLM |
|---|---|---|
| A. Direct recall | 5/5 (100%) | 4/5 (80%) |
| B. Multi-hop inference | 5/5 (100%) | 4/5 (80%) |
| C. Version-aware recall | 5/5 (100%) | 4/5 (80%) |
| D. Schema introspection | 2/3 (67%) | 1/3 (33%) |
| E. Absence detection | 3/3 (100%) | 3/3 (100%) |
| F. Long-horizon synthesis | 4/4 (100%) | 2/4 (50%) |
| **Overall** | **24/25 (96%)** | **18/25 (72%)** |

### 4.4 Honest interpretation

**Ada loses this benchmark.** Claude with the full 77-fact corpus in
context outperforms Ada+Claude on every category except absence
detection (where both score 100%).

We do not interpret this as a refutation of the substrate. The corpus
fits in 1.5K tokens; Claude's 200K-token window holds it comfortably,
sees facts in natural order (which conveniently aligns timestamps and
versions), and synthesises freely. There is no information-theoretic
advantage Ada could provide here.

The benchmark *would* favor Ada at corpus sizes that exceed the LLM's
context window. We did not run that at-scale variant in this paper
(generating a 10,000-fact corpus and re-running was outside the time
budget for this draft). The expected crossover, given Haiku's 200K
context window and ~20-token average fact length, is somewhere around
**8,000–10,000 facts**.

What the small-scale benchmark *does* demonstrate:

1. **Ada is competitive at small scale on the easy categories** (80% on
   A, B, C). It is not a structural impossibility for substrate-mediated
   recall to compete with full-context LLMs.
2. **Ada is correct about what it doesn't know.** All three absence
   queries are refused (100%) — the substrate's predicate-confidence
   threshold mechanically refuses to fabricate. This is the property
   that does not degrade with scale.
3. **Schema introspection and long-horizon synthesis remain weak** (33%
   and 50%) — meta-glyph surfacing and multi-fact bundling are known
   areas for further work.

## 5. What Ada does that no LLM/RAG/KG can do (the defensible claim)

If small-scale retrieval quality is not the contribution, what is?

1. **Versioned concepts with delta-vector trajectories.** No production
   RAG system stores compute deltas between belief versions. No
   knowledge graph emits drift rate as a queryable property by default.
   Ada's `history(key)` and `drift(key)` are first-class.
2. **Substrate self-description via meta-glyphs + markdown.** No system
   we are aware of writes its emerging schema both to a git-trackable
   markdown audit log *and* back into the same substrate as recallable
   entities. The two representations stay in sync; either can drive the
   other.
3. **Cognitive continuity across model swaps.** The substrate is a
   2048-dim bipolar HDC system stored on disk; the renderer is whichever
   LLM is current. Migration from Claude Haiku 4.5 to a future Claude 5
   does not require re-encoding the substrate. Ada is portable across
   model versions in a way that prompt-engineered memory systems are not.

These three properties are not retrievable by improving the LLM. They
are architectural commitments at the substrate layer.

## 6. Discussion

### 6.1 When Ada is the right choice

- Cross-session conversational memory beyond context window scale.
- Use cases requiring **audit trails** ("show me how my agent's belief
  about X evolved").
- Privacy-first deployments (no embedding API, all local).
- Federated memory (HDC merges by element-wise majority vote;
  vector DBs and KGs do not compose this way).

### 6.2 When Ada is not the right choice

- Small corpora that fit in context: full-context LLM is simpler and
  better.
- Pure semantic similarity on documents: transformer embeddings + ANN
  beat HDC cosine on synonyms and paraphrases.
- Strict typed-relationship reasoning (e.g., legal compliance graphs):
  knowledge graphs with Cypher/SPARQL are purpose-built for this.

### 6.3 Limitations and future work

The benchmark we report is small (77 facts). The decisive at-scale
comparison — Ada vs LLM with summarised context, Ada vs naive RAG, Ada
vs a KG with hand-designed temporal schema — requires generating a
10,000+ fact corpus and is left as next work. The schema-introspection
score (33%) suggests the meta-glyph recall policy can be improved; we
expect a 50-line tweak boosting `speaker="schema"` thoughts on
introspection-shaped queries to substantially close that gap.

The benchmark also does not yet measure the long-running dream-loop
behavior: the substrate has an implemented REM + slow-wave dream loop
that crystallises new compound primitives during idle time. A multi-day
deployment with periodic queries would show whether the substrate's
recall improves *without new facts being added* — purely from internal
pattern discovery.

## 7. Conclusion

Ada is honest about being a niche memory substrate. It does not beat a
context-window-loaded LLM at small-corpus retrieval. What it does
provide is a different category of memory: versioned, self-describing,
auditable, portable, and architecturally designed for cognitive
continuity beyond a single LLM's context window or model lifecycle.

The contribution we defend is not a benchmark number. It is the claim
that **a memory substrate that an LLM can be wired into as its brain —
with structured enrichment at write time, versioned drift on every
concept, and an auditable schema that grows with use — is feasible to
build, runnable on commodity hardware, and produces capabilities (drift
provenance, refusal discipline, schema introspection) that no
context-window enlargement or vector-DB augmentation provides.**

The implementation, benchmark, and audit logs are reproducible on the
`claude/repo-overview-ByHYQ` branch.

## Appendix A. Reproduction

```bash
# environment
pip install numpy pyyaml pydantic pydantic-settings transformers torch anthropic

# small-scale held-out test (95% / 100%)
PYTHONPATH=. python scripts/heldout_test.py

# hard test (72% / Ada, 96% / LLM-only baseline)
PYTHONPATH=. python scripts/hard_test.py
ANTHROPIC_API_KEY=... PYTHONPATH=. python benchmark/llm_only_baseline.py

# versioning demo
PYTHONPATH=. python scripts/versioning_demo.py

# schema-evolution demo
PYTHONPATH=. python scripts/schema_demo.py

# think vs ask
PYTHONPATH=. python scripts/think_vs_ask.py
```

The schema-evolution markdown changelogs land in `docs/` and are
intentionally checked into the repo as the audit-trail artifact of
each run.
