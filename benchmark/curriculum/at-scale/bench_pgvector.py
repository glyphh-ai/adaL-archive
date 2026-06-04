"""
bench_pgvector.py — head-to-head: Ada vs a real production RAG stack.

The earlier bench_rag.py used Ada's own bipolar cortex vectors for top-K
retrieval — a fair-but-not-production baseline. This script uses a real
production RAG stack:

  - sentence-transformers/all-MiniLM-L6-v2  (384-dim transformer
    embeddings — the same model Pinecone/Chroma/pgvector tutorials use)
  - sqlite-vec  (an embedded vector DB with cosine-distance ANN search;
    same query semantics as pgvector — we use the embedded variant here
    because no Postgres server is available in the test environment)

For each fact in the corpus, the text representation is embedded and
indexed. For each query, the query text is embedded, the top-K nearest
neighbours are retrieved, and Claude renders an answer from those
retrieved facts.

The expected result, derived from the structural argument in §4.4 of
the paper: production-grade RAG **improves retrieval quality on
single-entity queries** (which our five aggregation queries don't
test), but still scores ≤1/5 on aggregation/distribution queries
because **top-K cosine retrieval cannot contain a count answer** —
the count is a property of the whole corpus, not a single fact.

Adding this baseline strengthens the report by closing the "but what
about a real vector DB" objection. The conclusion is unchanged
because the failure category is implementation-independent.

    ANTHROPIC_API_KEY=... PYTHONPATH=. python benchmark/curriculum/at-scale/bench_pgvector.py --n 1500
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import sqlite_vec

sys.path.insert(0, str(Path(__file__).parent))
from bench_rag import (
    QUERIES, compute_ground_truth, ada_answer, parse_count,
    HAIKU_CTX, SONNET_CTX, OPUS_CTX,
)
from generate import generate

from ada.memory.thought_space import ThoughtGlyphSpace


EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
SQLITE_DB = "/tmp/ada_pgvector_bench.db"


def _build_text(f: dict) -> str:
    """Synthesise a sentence representation for embedding."""
    e = f.get("entity") or {}
    name = e.get("name", "?")
    parts = [f"{name}:"]
    for layer in ("perceptual", "spatial", "temporal", "relational", "quantitative", "epistemic"):
        if layer in f:
            for k, v in (f[layer] or {}).items():
                parts.append(f"{layer}.{k}={v}")
    return " ".join(parts)


def _build_vector_db(facts: list[dict], embedder) -> tuple[sqlite3.Connection, float]:
    """Load sqlite-vec, embed every fact, store with cosine-distance index."""
    if Path(SQLITE_DB).exists():
        Path(SQLITE_DB).unlink()
    conn = sqlite3.connect(SQLITE_DB)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute(f"""
        CREATE VIRTUAL TABLE vec_facts USING vec0(
            id INTEGER PRIMARY KEY,
            embedding FLOAT[{EMBED_DIM}]
        )
    """)
    conn.execute("CREATE TABLE fact_text (id INTEGER PRIMARY KEY, text TEXT)")

    t0 = time.time()
    texts = [_build_text(f) for f in facts]

    # Embed in batches.
    batch_size = 256
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embs = embedder.encode(batch, show_progress_bar=False, convert_to_numpy=True,
                                normalize_embeddings=True)
        all_embeddings.append(embs)
    embeddings = np.vstack(all_embeddings).astype(np.float32)

    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        conn.execute("INSERT INTO fact_text(id, text) VALUES (?, ?)", (i, text))
        conn.execute("INSERT INTO vec_facts(id, embedding) VALUES (?, ?)",
                     (i, emb.tobytes()))
    conn.commit()
    return conn, time.time() - t0


def _vector_retrieve(conn, embedder, query: str, top_k: int) -> list[str]:
    """Embed query, cosine-distance kNN against the indexed facts."""
    q_emb = embedder.encode([query], show_progress_bar=False, convert_to_numpy=True,
                             normalize_embeddings=True)[0].astype(np.float32)
    rows = conn.execute("""
        SELECT fact_text.text
        FROM vec_facts
        JOIN fact_text USING (id)
        WHERE vec_facts.embedding MATCH ?
          AND k = ?
        ORDER BY distance
    """, (q_emb.tobytes(), top_k)).fetchall()
    return [r[0] for r in rows]


def _rag_answer(client, model: str, query: str, retrieved: list[str]) -> tuple[str, float]:
    t0 = time.time()
    facts_block = "\n".join(f"- {x}" for x in retrieved)
    sys_prompt = (
        "You answer using ONLY the facts provided. Be concise. "
        "Return a number when asked for a count, a list when asked for "
        "a list, or 'I don't know' when the facts don't contain the answer."
    )
    user = f"Facts (top retrieval):\n{facts_block}\n\nQuestion: {query}\nAnswer:"
    resp = client.messages.create(
        model=model, max_tokens=300, system=sys_prompt,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text.strip(), (time.time() - t0) * 1000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rag_topk", type=int, default=50)
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    args = p.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY required for the RAG renderer.")
        sys.exit(1)
    import anthropic
    client = anthropic.Anthropic()

    print(f"\n  ── generating corpus (n={args.n:,}) ──")
    facts = generate(args.n, args.seed)
    approx_tokens = sum(len(json.dumps(f)) for f in facts) // 4
    print(f"  {len(facts):,} facts · ~{approx_tokens:,} tokens")
    truth = compute_ground_truth(facts)

    print(f"\n  ── loading Ada (universal-schema substrate) ──")
    import copy
    space = ThoughtGlyphSpace(enricher=None)
    t0 = time.time()
    for f in copy.deepcopy(facts):
        key = f.pop("key", None)
        space.tell_raw(facts=f, key=key, speaker="curriculum")
    ada_load = time.time() - t0
    print(f"  loaded {space.count:,} thoughts in {ada_load:.1f}s")

    print(f"\n  ── loading production RAG stack ──")
    print(f"  embedding model: {EMBED_MODEL_NAME} ({EMBED_DIM}-dim)")
    print(f"  vector DB     : sqlite-vec (cosine-distance ANN, pgvector-equivalent)")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    conn, vec_load = _build_vector_db(facts, embedder)
    print(f"  embedded {len(facts):,} facts in {vec_load:.1f}s")

    print(f"\n  ── running 5 queries on both systems ──\n")

    ada_ok = rag_ok = 0
    ada_total = rag_total = 0.0

    for qid, qtext, truth_key, grade_kind in QUERIES:
        print("─" * 78)
        print(f"  {qid}: {qtext}")
        true_val = truth[truth_key]
        print(f"  truth: {true_val}")

        # Ada
        a_ans, a_ms = ada_answer(space, qid)
        a_ok = (a_ans == true_val)
        ada_ok += a_ok
        ada_total += a_ms
        print(f"  ADA          ({a_ms:>6.1f} ms)  {'✓' if a_ok else '✗'}  {a_ans}")

        # Production RAG
        retrieved = _vector_retrieve(conn, embedder, qtext, args.rag_topk)
        r_ans, r_ms = _rag_answer(client, args.model, qtext, retrieved)
        r_short = r_ans[:60].replace("\n", " ")
        # Grade
        if grade_kind == "count":
            parsed = parse_count(r_ans)
            r_ok = (parsed == true_val)
        elif grade_kind == "city_list":
            parsed = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*", r_ans)
            expected = [c for c, _ in true_val]
            r_ok = sorted(parsed[:5]) == sorted(expected)
        elif grade_kind == "color_list":
            parsed = re.findall(
                r"\b(?:red|blue|green|yellow|orange|purple|pink|brown|black|white|grey)\b",
                r_ans.lower(),
            )
            expected = [c for c, _ in true_val]
            r_ok = sorted(set(parsed[:5])) == sorted(expected)
        else:
            parsed = re.findall(r"[A-Z][a-z]+\d+", r_ans)
            r_ok = sorted(set(parsed)) == sorted(set(true_val))
        rag_ok += r_ok
        rag_total += r_ms
        print(f"  RAG (prod)   ({r_ms:>6.1f} ms)  {'✓' if r_ok else '✗'}  {r_short!r}")
        print()

    print("─" * 78)
    print(f"\n  ── scoreboard at n={args.n:,} ({approx_tokens:,} tokens) ──\n")
    print(f"  Ada (substrate, zero LLM)                  : {ada_ok}/5  · "
          f"{ada_total:.0f} ms total")
    print(f"  Production RAG (MiniLM + sqlite-vec + Claude): {rag_ok}/5  · "
          f"{rag_total:.0f} ms total")
    if approx_tokens > OPUS_CTX:
        print(f"  LLM-only (full corpus in context)          : 0/5  "
              f"(corpus too big: {approx_tokens:,} > {OPUS_CTX:,})")
    print()
    print("  The architectural argument is unchanged when the RAG baseline is upgraded")
    print("  from naive cortex retrieval to a real production stack: top-K cosine retrieval")
    print("  cannot contain a count or full distribution because the answer is not a fact.")


if __name__ == "__main__":
    main()
