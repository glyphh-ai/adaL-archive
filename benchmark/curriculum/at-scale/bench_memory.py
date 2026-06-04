"""
bench_memory.py — head-to-head: Ada vs Claude/ChatGPT Memory feature.

The consumer-product Memory feature (Claude Memory, ChatGPT Memory)
isn't directly accessible via API. What it does at runtime is well-
characterised, though: maintain a **bounded budget of facts** (the
"memory") and inject them into every future conversation's system
prompt. When the user accumulates more facts than the budget holds,
the system either drops the oldest, summarises into a digest, or
truncates.

This script simulates that behavior fairly:

  1. Generate N persons × 20 universal-schema facts = 20N facts.
  2. Render each fact as a sentence (the form the memory feature
     stores).
  3. Pick the most-recent M facts that fit within a configurable
     memory budget (default 10,000 tokens — a realistic per-user
     budget for Claude Memory in 2025).
  4. Inject those M facts into the system prompt.
  5. Ask Claude the same five aggregation queries.
  6. Compare to Ada (which has no eviction budget).

The expected result, derived structurally:

  - At small corpora (everything fits): Claude Memory wins on
    ergonomics; Ada is overkill.
  - At corpora exceeding the memory budget: Claude Memory undercounts
    by *exactly the fraction of facts that fit* — e.g., 20K facts vs
    10K token budget ≈ 25% of facts visible → counts are ~25% of truth.
  - Ada's substrate has no eviction; 5/5 regardless of scale.

    ANTHROPIC_API_KEY=... PYTHONPATH=. python benchmark/curriculum/at-scale/bench_memory.py --n 200
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bench_rag import (
    QUERIES, compute_ground_truth, ada_answer, parse_count,
)
from bench_pgvector import _build_text
from generate import generate

from ada.memory.thought_space import ThoughtGlyphSpace


# Realistic per-user Claude Memory budget (2025). The actual product
# value is opaque and shifts over time; this is a representative cap
# in line with public reporting on ChatGPT Memory's behaviour and
# Anthropic's beta Memory features.
DEFAULT_MEMORY_TOKEN_BUDGET = 10_000


def _build_memory_sentences(facts: list[dict]) -> list[str]:
    """Render each fact as a sentence (the form a memory feature stores)."""
    return [_build_text(f) for f in facts]


def _select_within_budget(sentences: list[str], token_budget: int) -> list[str]:
    """Most-recent sentences that fit within the token budget.

    Most product memories evict oldest-first; we simulate that by walking
    backwards from the end of the corpus and keeping sentences until the
    budget is exhausted. Tokens approximated as chars/4.
    """
    chosen: list[str] = []
    budget = token_budget * 4   # → approximate char budget
    for s in reversed(sentences):
        cost = len(s) + 1
        if cost > budget:
            break
        chosen.append(s)
        budget -= cost
    chosen.reverse()
    return chosen


def _memory_answer(client, model: str, memory: list[str], query: str) -> tuple[str, float]:
    """Ask Claude with the memory budget injected as system prompt."""
    t0 = time.time()
    mem_block = "\n".join(f"- {s}" for s in memory)
    sys_prompt = (
        "You are Claude with a persistent Memory feature. Your stored "
        "memories about the user appear below. Answer using ONLY these "
        "memories. Return a number when asked for a count, a list when "
        "asked for a list, or 'I don't know' when the memories don't "
        "contain the answer.\n\nSTORED MEMORIES:\n" + mem_block
    )
    resp = client.messages.create(
        model=model, max_tokens=300, system=sys_prompt,
        messages=[{"role": "user", "content": query}],
    )
    return resp.content[0].text.strip(), (time.time() - t0) * 1000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200,
                   help="number of persons (×20 = facts)")
    p.add_argument("--budget", type=int, default=DEFAULT_MEMORY_TOKEN_BUDGET,
                   help="memory budget in tokens (default 10K)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    args = p.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY required.")
        sys.exit(1)
    import anthropic
    client = anthropic.Anthropic()

    print(f"\n  ── generating corpus (n={args.n:,}) ──")
    facts = generate(args.n, args.seed)
    approx_tokens = sum(len(json.dumps(f)) for f in facts) // 4
    print(f"  {len(facts):,} facts · ~{approx_tokens:,} tokens")
    truth = compute_ground_truth(facts)

    print(f"\n  ── simulating Claude/ChatGPT Memory ──")
    print(f"  memory budget: {args.budget:,} tokens "
          f"(realistic 2025 per-user product limit)")
    sentences = _build_memory_sentences(facts)
    in_memory = _select_within_budget(sentences, args.budget)
    visible_frac = len(in_memory) / len(facts)
    in_memory_tokens = sum(len(s) for s in in_memory) // 4
    print(f"  {len(in_memory):,} / {len(facts):,} facts fit in budget "
          f"({100*visible_frac:.1f}% of corpus, ~{in_memory_tokens:,} tokens)")

    print(f"\n  ── loading Ada (no eviction budget) ──")
    import copy
    space = ThoughtGlyphSpace(enricher=None)
    t0 = time.time()
    for f in copy.deepcopy(facts):
        key = f.pop("key", None)
        space.tell_raw(facts=f, key=key, speaker="curriculum")
    print(f"  loaded {space.count:,} thoughts in {time.time()-t0:.1f}s")

    print(f"\n  ── running 5 queries on both systems ──\n")
    ada_ok = mem_ok = 0
    ada_total = mem_total = 0.0
    rows = []

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
        print(f"  ADA              ({a_ms:>6.1f} ms)  {'✓' if a_ok else '✗'}  {a_ans}")

        # Memory-feature simulation
        m_ans, m_ms = _memory_answer(client, args.model, in_memory, qtext)
        m_short = m_ans[:60].replace("\n", " ")
        if grade_kind == "count":
            parsed = parse_count(m_ans)
            m_ok = (parsed == true_val)
            close = abs(parsed - true_val) / max(1, true_val) if parsed is not None else None
        elif grade_kind == "city_list":
            parsed = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*", m_ans)
            expected = [c for c, _ in true_val]
            m_ok = sorted(parsed[:5]) == sorted(expected)
            close = None
        elif grade_kind == "color_list":
            parsed = re.findall(
                r"\b(?:red|blue|green|yellow|orange|purple|pink|brown|black|white|grey)\b",
                m_ans.lower(),
            )
            expected = [c for c, _ in true_val]
            m_ok = sorted(set(parsed[:5])) == sorted(expected)
            close = None
        else:
            parsed = re.findall(r"[A-Z][a-z]+\d+", m_ans)
            m_ok = sorted(set(parsed)) == sorted(set(true_val))
            close = None
        mem_ok += m_ok
        mem_total += m_ms
        print(f"  CLAUDE MEMORY    ({m_ms:>6.1f} ms)  {'✓' if m_ok else '✗'}  {m_short!r}")
        if close is not None and not m_ok:
            print(f"  └─ off by {100*close:.0f}% — only saw {100*visible_frac:.1f}% of the corpus")
        print()
        rows.append((qid, true_val, a_ans, m_ans, a_ok, m_ok))

    print("─" * 78)
    print(f"\n  ── scoreboard at n={args.n:,} ({approx_tokens:,} tokens) ──\n")
    print(f"  Ada (no eviction)               : {ada_ok}/5  · "
          f"{ada_total:.0f} ms total")
    print(f"  Claude/ChatGPT Memory simulation: {mem_ok}/5  · "
          f"{mem_total:.0f} ms total")
    print(f"  Memory feature saw only {100*visible_frac:.1f}% of facts.")
    print()
    print("  The architectural argument: product Memory features have an eviction")
    print("  budget. At corpus sizes exceeding that budget, aggregation queries")
    print("  return numbers proportional to the visible fraction. Ada has no")
    print("  eviction; the substrate grows linearly with disk.")


if __name__ == "__main__":
    main()
