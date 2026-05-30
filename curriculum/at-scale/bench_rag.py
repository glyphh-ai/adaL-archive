"""
bench_rag.py — three-way comparison at Opus-busting scale.

For a single generated corpus, run the same 5 compositional queries
through three systems and compare:

  1. ADA          — structural recall via universal-schema role match.
                    Walks the substrate; deterministic; auditable.

  2. LLM-ONLY     — Claude with the ENTIRE corpus in context.
                    At n >> 1000 the corpus exceeds every public LLM
                    context window — the system literally cannot see
                    the data. We report 'EXCEEDS' rather than make a
                    failing call.

  3. RAG          — top-K cortex-cosine retrieval from the same substrate
                    Ada uses, then Claude as renderer. This is what
                    Pinecone+Claude / Chroma+Claude looks like in
                    practice: cheap retrieval, LLM does the synthesis.
                    Works for entity queries; structurally fails for
                    aggregation / distribution queries because top-K is
                    the wrong shape for those.

Each query is graded against ground truth computed from the corpus
generator. We score who's RIGHT, not who's first.

    ANTHROPIC_API_KEY=... PYTHONPATH=. python curriculum/at-scale/bench_rag.py --n 2500
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from generate import COLORS, CITIES, JOBS, HOBBIES, generate

from ada.core.ops import cosine_similarity
from ada.memory.thought_space import ThoughtGlyphSpace


# Public LLM context windows (2025 figures).
HAIKU_CTX = 200_000
SONNET_CTX = 1_000_000
OPUS_CTX = 200_000        # Claude Opus 4.x baseline. Some extended modes
                          # go higher; the 200K number is the default
                          # for the public model variants.


# ── ground truth from the generator ──────────────────────────────────

def compute_ground_truth(facts: list[dict]) -> dict:
    """The actual answers — from the deterministic generator."""
    color_count: Counter = Counter()
    job_count: Counter = Counter()
    city_count: Counter = Counter()
    blue_in_austin_chess: list[str] = []

    by_person_color: dict[str, str] = {}
    by_person_city: dict[str, str] = {}
    by_person_hobby: dict[str, str] = {}

    for f in facts:
        key = f.get("key", "")
        e = f.get("entity") or {}
        name = e.get("name", "")

        col = (f.get("perceptual") or {}).get("color")
        if col:
            color_count[col] += 1
            # Person's primary color (skip car_color)
            if "car_color" not in key and "color" in key and name and e.get("kind") == "person":
                by_person_color[name] = col

        loc = (f.get("spatial") or {}).get("location")
        if loc:
            city_count[loc] += 1
            if "city" in key and name and e.get("kind") == "person":
                by_person_city[name] = loc

        rel = f.get("relational") or {}
        if rel.get("predicate") == "works_as":
            job_count[rel.get("object", "")] += 1
        if rel.get("predicate") == "hobby":
            by_person_hobby[name] = rel.get("object", "")

    for name, col in by_person_color.items():
        if (col == "blue"
                and by_person_city.get(name) == "Austin"
                and by_person_hobby.get(name) == "chess"):
            blue_in_austin_chess.append(name)

    return {
        "blue_count": color_count.get("blue", 0),
        "engineer_count": job_count.get("engineer", 0),
        "top5_cities": city_count.most_common(5),
        "top5_colors": color_count.most_common(5),
        "blue_austin_chess": sorted(blue_in_austin_chess),
    }


# ── Ada structural answers ───────────────────────────────────────────

def ada_answer(space: ThoughtGlyphSpace, query_id: str) -> tuple[object, float]:
    t0 = time.time()
    if query_id == "A_blue_count":
        n = sum(1 for t in space._thoughts.values()
                if (t.glyph.metadata.get("_universal") or {})
                .get("perceptual", {}).get("color") == "blue")
        return n, (time.time() - t0) * 1000

    if query_id == "B_engineer_count":
        n = 0
        for t in space._thoughts.values():
            rel = (t.glyph.metadata.get("_universal") or {}).get("relational") or {}
            if rel.get("predicate") == "works_as" and rel.get("object") == "engineer":
                n += 1
        return n, (time.time() - t0) * 1000

    if query_id == "C_top5_cities":
        c: Counter = Counter()
        for t in space._thoughts.values():
            v = (t.glyph.metadata.get("_universal") or {}).get("spatial", {}).get("location")
            if v:
                c[v] += 1
        return c.most_common(5), (time.time() - t0) * 1000

    if query_id == "D_top5_colors":
        c = Counter()
        for t in space._thoughts.values():
            v = (t.glyph.metadata.get("_universal") or {}).get("perceptual", {}).get("color")
            if v:
                c[v] += 1
        return c.most_common(5), (time.time() - t0) * 1000

    if query_id == "E_blue_austin_chess":
        by_p_color: dict[str, str] = {}
        by_p_city: dict[str, str] = {}
        by_p_hobby: dict[str, str] = {}
        for t in space._thoughts.values():
            u = t.glyph.metadata.get("_universal") or {}
            e = u.get("entity") or {}
            name = e.get("name", "")
            if not name or e.get("kind") != "person":
                continue
            key = t.metadata.get("_key") or ""
            col = (u.get("perceptual") or {}).get("color")
            if col and "car_color" not in key and "color" in key:
                by_p_color[name] = col
            loc = (u.get("spatial") or {}).get("location")
            if loc:
                by_p_city[name] = loc
            rel = u.get("relational") or {}
            if rel.get("predicate") == "hobby":
                by_p_hobby[name] = rel.get("object", "")
        matches = sorted([
            n for n in by_p_color
            if by_p_color[n] == "blue"
            and by_p_city.get(n) == "Austin"
            and by_p_hobby.get(n) == "chess"
        ])
        return matches, (time.time() - t0) * 1000

    raise ValueError(f"unknown query: {query_id}")


# ── RAG: top-K cortex retrieval + Claude as renderer ─────────────────

def rag_retrieve(space: ThoughtGlyphSpace, query: str, top_k: int = 20) -> list[str]:
    """Brute-force cortex cosine across the substrate, return top-K text."""
    q_glyph = space._encoder.encode_thought(query, speaker="incoming")
    q_vec = q_glyph.global_cortex.data
    sims: list[tuple[float, str]] = []
    for t in space._thoughts.values():
        sim = float(cosine_similarity(q_vec, t.glyph.global_cortex.data))
        sims.append((sim, t.content))
    sims.sort(reverse=True)
    return [text for _sim, text in sims[:top_k]]


def rag_answer(client, model: str, query: str, retrieved: list[str]) -> tuple[str, float]:
    """Ask Claude with retrieved facts in context."""
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


def parse_count(s: str) -> int | None:
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else None


def parse_list(s: str) -> list[str]:
    """Crudely extract list-like names from a string."""
    # Pull capitalized words or quoted strings.
    items = re.findall(r"\b[A-Z][a-z]+\d*", s) + re.findall(r"['\"]([^'\"]+)['\"]", s)
    return list(dict.fromkeys(items))   # dedup, preserve order


# ── Bench runner ─────────────────────────────────────────────────────

QUERIES = [
    ("A_blue_count", "How many people have blue as their color?",
     "blue_count", "count"),
    ("B_engineer_count", "How many people work as engineers?",
     "engineer_count", "count"),
    ("C_top5_cities", "What are the top 5 most common cities people live in?",
     "top5_cities", "city_list"),
    ("D_top5_colors", "What are the top 5 most common color attributes across all people?",
     "top5_colors", "color_list"),
    ("E_blue_austin_chess",
     "Which people are blue, live in Austin, and have chess as their hobby?",
     "blue_austin_chess", "name_list"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2500,
                   help="number of synthetic persons (×20 = facts)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rag_topk", type=int, default=40)
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    args = p.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY required for the RAG/LLM baselines.")
        sys.exit(1)
    import anthropic
    client = anthropic.Anthropic()

    print(f"\n  ── generating corpus (n={args.n:,}) ──")
    facts = generate(args.n, args.seed)
    approx_tokens = sum(len(json.dumps(f)) for f in facts) // 4
    print(f"  {len(facts):,} facts · ~{approx_tokens:,} tokens")
    print(f"  Haiku 4.5  (200K):    "
          f"{'FITS' if approx_tokens <= HAIKU_CTX else 'EXCEEDS by ' + f'{approx_tokens/HAIKU_CTX:.1f}×'}")
    print(f"  Sonnet 4.6 (1M):      "
          f"{'FITS' if approx_tokens <= SONNET_CTX else 'EXCEEDS by ' + f'{approx_tokens/SONNET_CTX:.1f}×'}")
    print(f"  Opus 4.x   (200K):    "
          f"{'FITS' if approx_tokens <= OPUS_CTX else 'EXCEEDS by ' + f'{approx_tokens/OPUS_CTX:.1f}×'}")

    print(f"\n  ── ground truth (from generator) ──")
    truth = compute_ground_truth(facts)
    print(f"  blue_count      : {truth['blue_count']:,}")
    print(f"  engineer_count  : {truth['engineer_count']:,}")
    print(f"  top5_cities     : {truth['top5_cities']}")
    print(f"  top5_colors     : {truth['top5_colors']}")
    print(f"  blue_austin_chess: {truth['blue_austin_chess']}")

    print(f"\n  ── loading into Ada (zero LLM calls) ──")
    space = ThoughtGlyphSpace(enricher=None)
    # Deep-copy facts so generate's mutation doesn't bite the truth above.
    import copy
    t0 = time.time()
    loaded = 0
    for f in copy.deepcopy(facts):
        key = f.pop("key", None)
        if space.tell_raw(facts=f, key=key, speaker="curriculum") is not None:
            loaded += 1
    print(f"  loaded {loaded:,} thoughts in {time.time()-t0:.1f}s "
          f"({loaded/(time.time()-t0):.0f} facts/sec)")

    print(f"\n  ── running 5 queries through all three systems ──\n")

    ada_correct = rag_correct = llm_correct = 0

    for qid, query_text, truth_key, grade_kind in QUERIES:
        print("─" * 76)
        print(f"  QUERY {qid}: {query_text}")
        truth_val = truth[truth_key]
        print(f"  truth: {truth_val}")
        print()

        # ── Ada
        ada_ans, ada_ms = ada_answer(space, qid)
        ada_ok = (ada_ans == truth_val)
        ada_correct += ada_ok
        print(f"  ADA      ({ada_ms:>6.1f} ms)  {'✓' if ada_ok else '✗'}  {ada_ans}")

        # ── LLM-only
        if approx_tokens > OPUS_CTX:
            print(f"  LLM-ONLY (   N/A   )  ✗  IMPOSSIBLE — corpus exceeds Opus context")
        else:
            print(f"  LLM-ONLY (   N/A   )  ?  (skipped — would fit but cost prohibitive)")

        # ── RAG
        retrieved = rag_retrieve(space, query_text, top_k=args.rag_topk)
        rag_ans, rag_ms = rag_answer(client, args.model, query_text, retrieved)
        rag_short = rag_ans[:60].replace("\n", " ")
        if grade_kind == "count":
            parsed = parse_count(rag_ans)
            rag_ok = (parsed == truth_val)
        elif grade_kind == "city_list":
            parsed = parse_list(rag_ans)
            expected = [c for c, _ in truth_val]
            rag_ok = sorted(parsed[:5]) == sorted(expected)
        elif grade_kind == "color_list":
            parsed = re.findall(r"\b(?:red|blue|green|yellow|orange|purple|pink|brown|black|white|grey)\b", rag_ans.lower())
            expected = [c for c, _ in truth_val]
            rag_ok = sorted(set(parsed[:5])) == sorted(expected)
        else:  # name_list
            parsed = re.findall(r"[A-Z][a-z]+\d+", rag_ans)
            rag_ok = sorted(set(parsed)) == sorted(set(truth_val))
        rag_correct += rag_ok
        print(f"  RAG      ({rag_ms:>6.1f} ms)  {'✓' if rag_ok else '✗'}  {rag_short!r}")
        print()

    print("─" * 76)
    print(f"\n  ── final scoreboard at n={args.n:,} ({approx_tokens:,} tokens) ──\n")
    print(f"  Ada (substrate, zero LLM)    : {ada_correct}/5")
    print(f"  LLM-only (full corpus)        : {llm_correct}/5  "
          f"({'corpus too big' if approx_tokens > OPUS_CTX else 'skipped'})")
    print(f"  LLM + naive RAG (top-{args.rag_topk}) : {rag_correct}/5")
    print()


if __name__ == "__main__":
    main()
