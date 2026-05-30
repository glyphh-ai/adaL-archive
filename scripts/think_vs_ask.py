"""
think vs ask — the two cognitive surfaces, side by side.

Same substrate (the Aurora hard test corpus). Same input. Different
cognitive operation:

  think(input)   — the brain's reflex. Surfaces the constellation of
                   thoughts that resonate. No question, no answer, no
                   refusal. Just: what activates?

  ask(question)  — the brain's deliberate query. Returns one fact and
                   a confidence, or refuses with "I don't know."

For the MCP surface, these become two tools Claude can call:
  - think when it wants context to bring into a response
  - ask when it needs a specific answer

This demo shows the SAME input passed to both, so the difference is
obvious.

    ANTHROPIC_API_KEY=... PYTHONPATH=. python scripts/think_vs_ask.py
"""

from __future__ import annotations

import sys
sys.path.insert(0, "scripts")

from hard_test import SESSION_1

from ada.cognitive.generate import build_llm_renderer, build_qwen_renderer
from ada.cognitive.surface import CognitiveSurface
from ada.encoder.llm_enricher import auto_enricher
from ada.memory.glyph_cognitive import GlyphCognitiveLoop
from ada.memory.thought_space import ThoughtGlyphSpace
from ada.skill import SchemaTracker


def main() -> None:
    enricher = auto_enricher()
    space = ThoughtGlyphSpace(enricher=enricher)
    tracker = SchemaTracker(space, history_path="docs/think_ask_schema.md")
    cognitive = GlyphCognitiveLoop(space)
    renderer = build_llm_renderer() or build_qwen_renderer()
    surface = CognitiveSurface(space, cognitive, renderer=renderer)

    print(f"  loading the Aurora corpus ({len(SESSION_1)} facts)…\n")
    for text, key in SESSION_1:
        stored = space.absorb(text, key=key)
        if stored:
            tracker.observe(stored)
    print(f"  substrate holds {space.count} thoughts.\n")

    prompts = [
        "project aurora",
        "the hull of the boat",
        "investors",
        "what does the lead engineer do",
        "tell me about the budget",
        "are there any patents?",
    ]

    for p in prompts:
        print("═" * 90)
        print(f"INPUT:  {p!r}\n")

        # ── think — pure associative ──
        act = surface.think(p, top_k=6)
        print(f"think()  → {len(act)} thoughts activated"
              f"{f', pathway={act.pathway}' if act.pathway else ''}")
        for t, s in list(act)[:6]:
            print(f"    [{s:+.2f}]  {t.content[:75]}")

        # ── ask — targeted retrieval ──
        a = surface.ask(p)
        print()
        if a.refused:
            print(f"ask()    → REFUSED (conf={a.confidence:.2f})  → {a.rendered}")
        else:
            print(f"ask()    → ANSWERED (conf={a.confidence:.2f})")
            print(f"    fact: {a.fact.content if a.fact else '—'}")
            print(f"    rendered: {a.rendered}")
        print()


if __name__ == "__main__":
    main()
