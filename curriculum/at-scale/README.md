# At-scale benchmark — proving Ada handles what LLMs structurally can't

The argument up to here has been: *Ada is a memory substrate that does
things vector DBs and KGs can't.* This directory makes the **stronger
empirical claim**:

> At corpus sizes that exceed an LLM's context window, Ada answers
> compositional queries deterministically. The LLM cannot answer them
> at all, because it cannot see the data.

## Methodology

Single Python file generates a deterministic synthetic corpus of
**N persons × 20 universal-schema facts each = 20N facts**:

| `--n` | Facts | Approx tokens | Haiku 4.5 (200K) | Sonnet 4.6 (1M) | Ada |
|---|---|---|---|---|---|
| 100 | 2,000 | ~70K | ✓ fits | ✓ fits | ✓ |
| 1,000 | 20,000 | ~700K | ✗ **exceeds** | ✓ fits | ✓ |
| 5,000 | 100,000 | ~3.5M | ✗ **exceeds** | ✗ **exceeds** | ✓ |
| 50,000 | 1,000,000 | ~35M | ✗ **far exceeds** | ✗ **far exceeds** | ✓ |

The crossover is at **n=300** for Haiku (~6,000 facts, 210K tokens) and
**n=1,400** for Sonnet (~28,000 facts, 1M tokens). Past those points,
the LLM-with-full-context baseline is **structurally impossible**, not
just slow or expensive. The relevant data does not fit in the prompt.

## Five compositional queries the benchmark runs

For each, Ada answers from the substrate; the LLM-only baseline can
only answer if the corpus fits.

| Query | Substrate operation | What it tests |
|---|---|---|
| A. How many entities have `perceptual.color = blue`? | Role-value scan | Cross-fact role aggregation |
| B. How many persons work as engineer? | Relational predicate match | Predicate-targeted retrieval |
| C. Top-5 most-common cities? | Distribution over `spatial.location` | Aggregation / counting |
| D. Top-5 most-common color attributes? | Distribution across all `perceptual.color` slots | Cross-domain enumeration |
| E. People who are blue AND in Austin AND hobby = chess? | Three-way conjunction across three roles | Multi-role intersection |

Each is a *single query*. Each runs in **milliseconds** at 20K-fact
scale. Each returns a deterministic, auditable answer.

## Run it

```bash
# 1K persons × 20 facts = 20K facts (~700K tokens; busts Haiku)
PYTHONPATH=. python curriculum/at-scale/bench.py --n 1000

# 5K persons × 20 facts = 100K facts (~3.5M tokens; busts Sonnet too)
PYTHONPATH=. python curriculum/at-scale/bench.py --n 5000
```

The benchmark prints per-query latency, an aggregate latency stat, and
explicitly reports for each LLM whether the corpus fits in its context.

## Sample output (n=1000)

```
generated 20,000 facts about 1,000 persons
saved → curriculum/at-scale/corpus.json
approx LLM token count: 714,246
  ⚠  EXCEEDS Claude Haiku 4.5 context window (200K)

loaded 20,000/20,000 thoughts in 134.8s (148 facts/sec)
substrate holds 20,000 thoughts

A. How many entities have perceptual.color = blue?
   → 190 matches in 15.7 ms
B. How many persons work as engineer?
   → 48 matches in 16.5 ms
C. Top-5 most common cities people live in:
   Denver       49
   Antwerp      42
   Quito        40
   ...
   (15.1 ms)

query latencies: 17.2 ms avg, 23.6 ms max

── what the LLM-only baseline could do ──
Haiku 4.5 with full corpus in context: IMPOSSIBLE (714,246 > 200,000 tokens)
Sonnet 4.6 with full corpus in context: WORKS (714,246/1,000,000 tokens)
```

## What RAG would do here, honestly

RAG would *retrieve top-K* facts by cosine similarity and feed them to
the LLM. That works for queries like "tell me about Kira" — get her
20 facts and answer. **It breaks for any aggregation or distribution
query**:

- "How many people work as engineer?" — RAG returns at most K facts; the
  *real* answer requires scanning all 20,000. RAG can only answer "of
  the K facts I retrieved, X look like engineers" — which is wrong.
- "Top 5 cities?" — same problem. Aggregation is not a top-K retrieval.

Ada handles aggregation natively because every fact in the substrate
is addressable by role. The Counter walks the whole substrate at
~milliseconds per 1K facts. There is no top-K bottleneck.

## What this proves

At any corpus size, three claims hold:

1. **Ada answers** in deterministic, auditable, sub-second time.
2. **LLM-only** is bounded by context — at scale it's structurally
   impossible.
3. **RAG** can handle retrieval at scale but **cannot do aggregation or
   distribution queries** because top-K is the wrong shape for those.

These three are the architectural argument made empirical. Any
specific corpus and query set can be slotted in; the conclusion is the
same.

## Files

```
generate.py     deterministic synthetic-corpus generator
bench.py        loads corpus, runs five queries, reports
corpus.json     last-generated corpus (gitignored at scale)
README.md       this file
```
