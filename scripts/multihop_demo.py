"""
Multi-hop join test — does the substrate + renderer infer relations
it was never told directly?

Facts: I am Chris. Chris is married to Brandi. Brandi has two children:
Jim and Bob.

Query: What are Chris's children's names?

There is no fact "Chris has children jim and bob" in the corpus. The
system has to chain married→shared-children to answer.
"""

from __future__ import annotations

from ada.cognitive.generate import (
    GlyphGenerator,
    build_llm_renderer,
    build_qwen_renderer,
)
from ada.memory.glyph_cognitive import GlyphCognitiveLoop
from ada.memory.thought_space import ThoughtGlyphSpace


FACTS = [
    "I am Chris.",
    "Chris is married to Brandi.",
    "Brandi has two children: Jim and Bob.",
]

QUERIES = [
    "What are Chris's children's names?",
    "Who is Brandi married to?",
    "Who are Jim's parents?",
    "Does Chris have children?",
]


def main() -> None:
    space = ThoughtGlyphSpace()
    cog = GlyphCognitiveLoop(space)
    renderer = build_llm_renderer() or build_qwen_renderer()
    gen = GlyphGenerator(space, cog, renderer=renderer)

    print("  ingesting:")
    for f in FACTS:
        space.absorb(f)
        print(f"    + {f}")
    print()

    for q in QUERIES:
        print("─" * 80)
        result = gen.generate(q, top_chains=5, absorb_response=False)
        print(f"Q: {q}")
        print("\n  substrate top-3 recall:")
        for i, chain in enumerate(result.chains[:3], 1):
            for hop in chain.hops:
                print(f"    [{i}] conf={chain.confidence:.2f}  {hop.result.thought.content!r}")
        print(f"\n  facts given to renderer:")
        for line in result.render_facts.split("\n"):
            print(f"    {line}")
        if result.response:
            print(f"\n  RENDERER: {result.response}")
    print("─" * 80)


if __name__ == "__main__":
    main()
