"""
Schema-evolution demo — Ada watches her own substrate grow.

For every absorb, SchemaTracker checks the enricher's StructuredFact.
The first time a new predicate or topic shows up, it:

  1. Appends a markdown entry to docs/schema_history.md (audit trail)
  2. Absorbs a meta-glyph describing the new schema element back into
     the substrate, under a stable key like `schema.predicate.age`
     (so Ada can recall + version her own schema)

End of run we print:
  - the tally (how many predicates/topics learned)
  - the markdown changelog
  - a query asking the substrate ABOUT ITS OWN SCHEMA — the meta-glyphs
    surface like any other thought, because they ARE thoughts.
"""

from __future__ import annotations

import sys
sys.path.insert(0, "scripts")

from generate_demo_100 import FACTS

from ada.cognitive.generate import GlyphGenerator, build_llm_renderer, build_qwen_renderer
from ada.encoder.llm_enricher import auto_enricher
from ada.memory.glyph_cognitive import GlyphCognitiveLoop
from ada.memory.thought_space import ThoughtGlyphSpace
from ada.skill import SchemaTracker


def main() -> None:
    enricher = auto_enricher()
    space = ThoughtGlyphSpace(enricher=enricher)
    tracker = SchemaTracker(space, history_path="docs/schema_history.md")
    cognitive = GlyphCognitiveLoop(space)
    renderer = build_llm_renderer() or build_qwen_renderer()
    gen = GlyphGenerator(space, cognitive, renderer=renderer)

    print(f"  ingesting {len(FACTS)} facts; SchemaTracker watching...\n")
    for f in FACTS:
        stored = space.absorb(f)
        if stored:
            info = tracker.observe(stored)
            if info["new_predicate"] or info["new_topic"]:
                # Briefly note what the tracker just persisted.
                sf = stored.glyph.metadata.get("_structured")
                marks = []
                if info["new_predicate"]:
                    marks.append(f"+pred `{sf.predicate}`")
                if info["new_topic"]:
                    marks.append(f"+topic `{sf.topic}`")
                print(f"   {' '.join(marks):<60}  ← {f}")

    print()
    s = tracker.summary()
    print("─── schema state ───")
    print(f"   predicates known : {s['predicates_known']}")
    print(f"   topics known     : {s['topics_known']}")
    print(f"   introductions    : {s['introductions']}")
    print(f"   top predicates   : {s['top_predicates']}")
    print(f"   top topics       : {s['top_topics']}")
    print()

    print("─── docs/schema_history.md (head) ───")
    from pathlib import Path
    head = Path("docs/schema_history.md").read_text().splitlines()
    for line in head[:40]:
        print(f"   {line}")
    print(f"   ... ({len(head)} lines total)")
    print()

    # ── The clever part: ask the substrate about its OWN schema. ──
    # Meta-glyphs are stored exactly like fact glyphs, so they surface in
    # recall when queried about schema concepts.
    print("─── Ada querying her own schema ───\n")
    queries = [
        "What predicates does Ada know about?",
        "Are there any topics related to capitals?",
        "What predicates exist for people?",
    ]
    for q in queries:
        result = gen.generate(q, top_chains=3, absorb_response=False, reinforce=False)
        print(f"  Q: {q}")
        print(f"  facts shown to renderer:")
        for line in result.render_facts.split("\n"):
            print(f"    {line}")
        if result.response:
            print(f"  → {result.response}")
        print()


if __name__ == "__main__":
    main()
