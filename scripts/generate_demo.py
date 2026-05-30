"""
GENERATE-stage demo — see what adding the missing piece buys.

Absorbs a small corpus of facts into Ada's HDC substrate, then runs the
new GlyphGenerator on a few queries. With ANTHROPIC_API_KEY set you also
see the LLM-rendered response. Without it you see the substrate's raw
"this is what I think is relevant" output — which IS the new piece.

Run:
    python scripts/generate_demo.py
"""

from __future__ import annotations

from ada.cognitive.generate import GlyphGenerator, build_llm_renderer, build_qwen_renderer
from ada.memory.glyph_cognitive import GlyphCognitiveLoop
from ada.memory.thought_space import ThoughtGlyphSpace


FACTS = [
    "My name is Chris.",
    "I work on the Ada project.",
    "Ada is a self-evolving cognitive brain.",
    "Ada uses HDC primitives.",
    "HDC stands for hyperdimensional computing.",
    "My favorite color is blue.",
    "The Ada server runs on port 8002.",
    "Claude is an AI assistant made by Anthropic.",
    "I live in Austin.",
    "Ada's memory is stored as 2048-dimensional bipolar vectors.",
    "Each thought has five hierarchical layers.",
    "Glyph similarity uses weighted cortex, layer, segment, and role.",
]

QUERIES = [
    "What is my name?",
    "Where do I live?",
    "What does Ada use?",
    "What is HDC?",
    "What color do I like?",
    "Who built Claude?",
    "What port does Ada run on?",
]


def main() -> None:
    space = ThoughtGlyphSpace()
    cognitive = GlyphCognitiveLoop(space)
    # Prefer Anthropic if a key is set, else local Qwen3.
    renderer = build_llm_renderer() or build_qwen_renderer()
    gen = GlyphGenerator(space, cognitive, renderer=renderer)

    print("──────── ingesting facts ────────")
    for f in FACTS:
        stored = space.absorb(f)
        marker = "+" if stored else " "
        print(f"  {marker}  {f}")
    print(f"\n  substrate now holds {space.count} thoughts.\n")

    if renderer is None:
        print("  (no LLM renderer wired — showing substrate context only.)")
        print("  (set ANTHROPIC_API_KEY + `pip install anthropic` to enable rendering.)\n")

    for q in QUERIES:
        print("════════════════════════════════════════════════════════════════")
        result = gen.generate(q, top_chains=3, absorb_response=False)
        print(result.context)
        print(f"\n  facts to renderer:")
        for line in result.render_facts.split("\n"):
            print(f"    {line}")
        if result.response:
            print(f"\n  → renderer: {result.response}")
        else:
            top = result.top_recall
            if top:
                print(f"\n  → top-recall surrogate: {top!r}")
            else:
                print("\n  → (no recall)")
    print("════════════════════════════════════════════════════════════════")
    print(f"\nFinal substrate size: {space.count} thoughts "
          f"({len(FACTS)} input + {space.count - len(FACTS)} renderer outputs absorbed back).")


if __name__ == "__main__":
    main()
