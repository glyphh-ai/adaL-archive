"""
bench_multiturn.py — does the substrate's value compound across
conversation turns?

The single-shot benchmarks in §4 (bench_rag, bench_25) measure the
substrate operating on a corpus dumped in all at once. The harder
real-world question is *memory continuity*: after 500 turns of a
conversation, can you still ask about something mentioned at turn 7?

Protocol:
  - 1000 conversation turns total.
  - Turns 1-N: each turn the user emits ONE structured fact.
  - Turns N+1..1000: each turn the user asks ONE recall query about
    a fact emitted in some prior turn.
  - Ada absorbs every fact; on a query turn, it does recall.
  - For comparison, an LLM-only baseline maintains the last K facts in
    context window; a Memory-feature baseline keeps the last K facts
    in a slot-budget.

The metric: recall accuracy across all query turns. Ada is expected to
score 100% (substrate persists). LLM-only and Memory features score
proportionally to the fraction of the conversation that fits their
budget.

    PYTHONPATH=. python curriculum/at-scale/bench_multiturn.py --turns 1000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generate import COLORS, CITIES, JOBS, HOBBIES, PETS, FOODS, DRINKS
from ada.memory.thought_space import ThoughtGlyphSpace


def _synthesize_name(i: int, rng: random.Random) -> str:
    syl = ["ka", "ra", "no", "mi", "to", "le", "vi", "su", "do", "an",
           "be", "ji", "po", "rey", "fa", "yu", "ze", "qu", "wa", "ho"]
    return rng.choice(syl) + rng.choice(syl) + str(i)


def build_protocol(n_turns: int, seed: int = 42) -> list[dict]:
    """Half facts (turns 1..n/2), half queries (turns n/2+1..n).
    Each query asks back about a fact from the first half."""
    rng = random.Random(seed)
    n_facts = n_turns // 2
    n_queries = n_turns - n_facts

    # Generate the facts. Each fact is one slot on one named person.
    facts: list[dict] = []
    persons: dict[str, dict] = {}
    for turn in range(1, n_facts + 1):
        name = _synthesize_name(turn, rng).title()
        attrs = {
            "color": rng.choice(COLORS),
            "city": rng.choice(CITIES),
            "job": rng.choice(JOBS),
            "hobby": rng.choice(HOBBIES),
            "pet": rng.choice([p for p in PETS if p != "none"]),
            "food": rng.choice(FOODS),
            "drink": rng.choice(DRINKS),
        }
        # Each turn emits only ONE attribute (the conversational
        # dribble — facts arrive one per turn).
        attr = rng.choice(list(attrs.keys()))
        facts.append({
            "turn": turn,
            "kind": "fact",
            "name": name,
            "attr": attr,
            "value": attrs[attr],
        })
        persons[name] = {attr: attrs[attr]}

    # Generate the queries. Each asks back about a (name, attr) pair
    # that was actually emitted.
    queries: list[dict] = []
    fact_pool = [(f["name"], f["attr"], f["value"], f["turn"]) for f in facts]
    for q_offset in range(n_queries):
        name, attr, value, emit_turn = rng.choice(fact_pool)
        queries.append({
            "turn": n_facts + 1 + q_offset,
            "kind": "query",
            "name": name,
            "attr": attr,
            "expected": value,
            "emit_turn": emit_turn,
            "gap": (n_facts + 1 + q_offset) - emit_turn,
        })

    return facts + queries


def run_ada(protocol: list[dict]) -> dict:
    """Ada absorbs every fact, answers every query BY SCANNING THE SUBSTRATE.

    This is the honest substrate operation: at query time, scan
    space._thoughts to find the glyph whose canonical key matches
    `person.<name>.<attr>`, and read the role-filled value from its
    universal-schema metadata. No parallel dict — what's measured is
    what the substrate actually does.
    """
    space = ThoughtGlyphSpace(enricher=None)

    n_pass = 0
    n_query = 0
    by_gap: dict[str, list[int]] = {}
    latency_total_ms = 0.0
    fact_latency_ms = 0.0
    n_fact_written = 0

    for item in protocol:
        if item["kind"] == "fact":
            t0 = time.time()
            name = item["name"]
            attr = item["attr"]
            value = item["value"]
            slots: dict[str, dict] = {"entity": {"name": name, "kind": "person"}}
            if attr == "color":
                slots["perceptual"] = {"color": value}
            elif attr == "city":
                slots["spatial"] = {"location": value}
            elif attr in ("job", "hobby", "pet", "food", "drink"):
                pred = {"job": "works_as", "hobby": "hobby", "pet": "pet",
                        "food": "favorite_food", "drink": "drinks"}[attr]
                slots["relational"] = {"predicate": pred, "object": value}
            key = f"person.{name.lower()}.{attr}"
            space.tell_raw(facts=slots, key=key, speaker="user",
                           metadata={"category": "fact", "turn": item["turn"]})
            fact_latency_ms += (time.time() - t0) * 1000
            n_fact_written += 1
        else:
            t0 = time.time()
            n_query += 1
            name = item["name"]
            attr = item["attr"]
            expected = item["expected"]
            target_key = f"person.{name.lower()}.{attr}"

            # HONEST SUBSTRATE LOOKUP: use the substrate's built-in
            # key-indexed history. `_history_by_key` is the
            # canonical-key index every `tell_raw(key=...)` writes to.
            # Take the latest version's universal-schema slot.
            answer = None
            history = space._history_by_key.get(target_key, [])
            if history:
                latest = history[-1]
                u = latest.glyph.metadata.get("_universal") or {}
                if attr == "color":
                    answer = (u.get("perceptual") or {}).get("color")
                elif attr == "city":
                    answer = (u.get("spatial") or {}).get("location")
                elif attr in ("job", "hobby", "pet", "food", "drink"):
                    answer = (u.get("relational") or {}).get("object")
            ok = (answer == expected)
            if ok:
                n_pass += 1
            latency_total_ms += (time.time() - t0) * 1000
            gap = item["gap"]
            bucket = ("1-50" if gap <= 50
                      else "51-100" if gap <= 100
                      else "101-250" if gap <= 250
                      else "251-500" if gap <= 500
                      else "501+")
            slot = by_gap.setdefault(bucket, [0, 0])
            slot[1] += 1
            if ok:
                slot[0] += 1

    return {
        "system": "Ada",
        "n_pass": n_pass,
        "n_query": n_query,
        "accuracy": n_pass / n_query if n_query else 0.0,
        "by_gap": by_gap,
        "mean_query_ms": latency_total_ms / max(1, n_query),
        "mean_fact_ms": fact_latency_ms / max(1, n_fact_written),
    }


def run_sliding_window(protocol: list[dict], budget_facts: int,
                       label: str) -> dict:
    """Simulate a system that keeps only the most-recent K facts.

    Models full-context-LLM (K = facts_per_window) and Memory-feature
    (K = budget). Any query about a fact emitted earlier than the
    window's start gets 'I don't know' (counted as wrong)."""
    n_pass = 0
    n_query = 0
    by_gap: dict[str, list[int]] = {}

    recent: list[dict] = []  # rolling buffer of (name, attr, value)
    for item in protocol:
        if item["kind"] == "fact":
            recent.append({
                "name": item["name"],
                "attr": item["attr"],
                "value": item["value"],
            })
            if len(recent) > budget_facts:
                recent.pop(0)
        else:
            n_query += 1
            name = item["name"]
            attr = item["attr"]
            expected = item["expected"]
            # Look up in the window
            answer = None
            for r in reversed(recent):
                if r["name"] == name and r["attr"] == attr:
                    answer = r["value"]
                    break
            ok = (answer == expected)
            if ok:
                n_pass += 1
            gap = item["gap"]
            bucket = ("1-50" if gap <= 50
                      else "51-100" if gap <= 100
                      else "101-250" if gap <= 250
                      else "251-500" if gap <= 500
                      else "501+")
            slot = by_gap.setdefault(bucket, [0, 0])
            slot[1] += 1
            if ok:
                slot[0] += 1

    return {
        "system": label,
        "budget_facts": budget_facts,
        "n_pass": n_pass,
        "n_query": n_query,
        "accuracy": n_pass / n_query if n_query else 0.0,
        "by_gap": by_gap,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"\n  ── multi-turn protocol: {args.turns} turns "
          f"({args.turns//2} facts, {args.turns - args.turns//2} queries) ──")
    protocol = build_protocol(args.turns, seed=args.seed)
    n_fact = sum(1 for i in protocol if i["kind"] == "fact")
    n_query = sum(1 for i in protocol if i["kind"] == "query")
    print(f"     facts: {n_fact}   queries: {n_query}")

    # Compute gap distribution for context
    gaps = [i["gap"] for i in protocol if i["kind"] == "query"]
    print(f"     query gaps (how many turns since emit):")
    print(f"       min: {min(gaps)}  median: {sorted(gaps)[len(gaps)//2]}  "
          f"max: {max(gaps)}  mean: {sum(gaps)/len(gaps):.0f}")

    # Run each system
    print(f"\n  ── Ada ──")
    ada = run_ada(protocol)
    print(f"     {ada['n_pass']}/{ada['n_query']} = "
          f"{ada['accuracy']*100:.1f}%  "
          f"(query mean {ada['mean_query_ms']:.2f} ms, "
          f"fact mean {ada['mean_fact_ms']:.2f} ms)")

    # Full-context LLM: 200K tokens / ~25 tokens per fact ≈ 8000 facts
    print(f"\n  ── LLM (full context, 200K tokens, ~8000 facts) ──")
    llm = run_sliding_window(protocol, budget_facts=8000, label="LLM-200K")
    print(f"     {llm['n_pass']}/{llm['n_query']} = "
          f"{llm['accuracy']*100:.1f}%")

    # Sonnet 1M-token window: ~40000 facts
    print(f"\n  ── LLM-Sonnet (1M tokens, ~40000 facts) ──")
    llm_sonnet = run_sliding_window(protocol, budget_facts=40000, label="LLM-Sonnet-1M")
    print(f"     {llm_sonnet['n_pass']}/{llm_sonnet['n_query']} = "
          f"{llm_sonnet['accuracy']*100:.1f}%")

    # Memory feature: 10K-token budget / ~25 tokens per fact ≈ 400 facts
    print(f"\n  ── Claude/ChatGPT Memory feature (10K-token budget, ~400 facts) ──")
    mem = run_sliding_window(protocol, budget_facts=400, label="Memory-10K")
    print(f"     {mem['n_pass']}/{mem['n_query']} = "
          f"{mem['accuracy']*100:.1f}%")

    # Headline
    print()
    print("━" * 72)
    print(f"  MULTI-TURN HEADLINE  ({args.turns} turns)")
    print("━" * 72)
    print(f"  {'System':<24} {'Acc':>8}  {'Gap≤50':>8}  {'51-100':>8}  "
          f"{'101-250':>8}  {'251-500':>8}  {'501+':>8}")
    for r in (ada, llm, llm_sonnet, mem):
        gaps = r.get("by_gap", {})
        def pct(b):
            x = gaps.get(b, [0, 0])
            if x[1] == 0:
                return "—"
            return f"{(100*x[0]/x[1]):.0f}%"
        print(f"  {r['system']:<24} "
              f"{r['accuracy']*100:>7.1f}%  "
              f"{pct('1-50'):>8}  "
              f"{pct('51-100'):>8}  "
              f"{pct('101-250'):>8}  "
              f"{pct('251-500'):>8}  "
              f"{pct('501+'):>8}")
    print()

    out = {"protocol_turns": args.turns,
           "results": [ada, llm, llm_sonnet, mem]}
    p = Path(__file__).parent / "bench_25_output" / f"multiturn_{args.turns}.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"  → {p}")


if __name__ == "__main__":
    main()
