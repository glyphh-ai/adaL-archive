"""
Held-out generalization test — does the enricher overfit?

Same architecture as generate_demo_100.py, but with 50 brand-new facts
about brand-new entities. The Haiku enricher (or heuristic fallback) has
no parse history specific to these facts — it must apply the same general
patterns it would apply to any input. If the architecture generalizes,
substrate top-1 and renderer accuracy should land in the same range as
the 100-fact in-distribution test (90% / 90%). If it overfits to the
specific facts seen before, accuracy will collapse on these unseen ones.

Run:
    ANTHROPIC_API_KEY=... PYTHONPATH=. python scripts/heldout_test.py
"""

from __future__ import annotations

import time

from ada.cognitive.generate import (
    GlyphGenerator,
    build_llm_renderer,
    build_qwen_renderer,
)
from ada.encoder.llm_enricher import auto_enricher
from ada.memory.glyph_cognitive import GlyphCognitiveLoop
from ada.memory.thought_space import ThoughtGlyphSpace


# ── 50-fact held-out corpus — all new entities ────────────────────────

# 10 new people × 2 facts = 20 person facts. Cities chosen to NOT collide
# with the 30 country capitals below.
PEOPLE = [
    ("Kira",  28, "Phoenix"),
    ("Lex",   35, "Calgary"),
    ("Mira",  41, "Munich"),
    ("Noah",  22, "Marseille"),
    ("Owen",  50, "Belfast"),
    ("Pria",  30, "Naples"),
    ("Quinn", 45, "Bordeaux"),
    ("Reza",  38, "Antwerp"),
    ("Sam",   27, "Bristol"),
    ("Tara",  33, "Lyon"),
]
PERSON_FACTS = []
for name, age, city in PEOPLE:
    PERSON_FACTS.append(f"{name} is {age} years old.")
    PERSON_FACTS.append(f"{name} lives in {city}.")

# 15 new country → capital pairs, none overlapping with the original 30.
CAPITALS = [
    ("South Korea", "Seoul"),
    ("Philippines", "Manila"),
    ("Malaysia",    "Kuala Lumpur"),
    ("Pakistan",    "Islamabad"),
    ("Nigeria",     "Abuja"),
    ("Saudi Arabia","Riyadh"),
    ("Qatar",       "Doha"),
    ("Switzerland", "Bern"),
    ("Netherlands", "Amsterdam"),
    ("Belgium",     "Brussels"),
    ("Denmark",     "Copenhagen"),
    ("Hungary",     "Budapest"),
    ("Austria",     "Vienna"),
    ("Romania",     "Bucharest"),
    ("Czechia",     "Prague"),
]
CAPITAL_FACTS = [f"The capital of {c} is {cap}." for c, cap in CAPITALS]

# 10 new science facts.
SCIENCE_FACTS = [
    "The Earth has one moon.",
    "Mercury is the closest planet to the Sun.",
    "Jupiter is the largest planet in the solar system.",
    "Neptune is the eighth planet from the Sun.",
    "A leap year has 366 days.",
    "The tongue has 8,000 taste buds.",
    "Humans have 23 pairs of chromosomes.",
    "The human brain weighs about 1.4 kilograms.",
    "Silver has atomic number 47.",
    "Mars has two moons.",
]

# 5 new Ada / HDC facts.
ADA_FACTS = [
    "Ada caches encodings by content hash.",
    "Ada's similarity calculator uses cosine distance.",
    "Ada's REM dream loop runs every 3 seconds.",
    "Ada's pathway library stores activation patterns by hash.",
    "Ada's encoder normalizes vectors to bipolar.",
]

FACTS = PERSON_FACTS + CAPITAL_FACTS + SCIENCE_FACTS + ADA_FACTS
assert len(FACTS) == 50, f"expected 50, got {len(FACTS)}"


# ── 20-query held-out test set ──────────────────────────────────────

QUERIES: list[tuple[str, str, str]] = [
    # Person attribute lookups (5)
    ("What is Kira's age?",                "Kira is 28",       "28"),
    ("Where does Mira live?",              "Mira lives",       "Munich"),
    ("How old is Reza?",                   "Reza is 38",       "38"),
    ("Where does Noah live?",              "Noah lives",       "Marseille"),
    ("What is Sam's age?",                 "Sam is 27",        "27"),

    # Capitals (5)
    ("What is the capital of South Korea?","Korea",            "Seoul"),
    ("What is the capital of Nigeria?",    "Nigeria",          "Abuja"),
    ("Capital of Denmark?",                "Denmark",          "Copenhagen"),
    ("What is Switzerland's capital?",     "Switzerland",      "Bern"),
    ("Capital city of Belgium?",           "Belgium",          "Brussels"),

    # Science (5)
    ("How many moons does Earth have?",    "Earth has one",    "one"),
    ("Which planet is closest to the Sun?","Mercury",          "Mercury"),
    ("Which planet is the largest?",       "Jupiter",          "Jupiter"),
    ("How many pairs of chromosomes do humans have?",
                                           "23 pairs",         "23"),
    ("What is the atomic number of silver?","atomic number 47","47"),

    # Ada / HDC (5)
    ("What does Ada cache by content hash?",  "caches encodings", "encod"),
    ("What metric does Ada's similarity use?","cosine",          "cosine"),
    ("How often does Ada's REM dream loop run?", "3 seconds",     "3 seconds"),
    ("Where does Ada store activation patterns?","pathway library","pathway"),
    ("What does Ada's encoder normalize vectors to?","bipolar",  "bipolar"),
]


def main() -> None:
    enricher = auto_enricher()
    space = ThoughtGlyphSpace(enricher=enricher)
    cognitive = GlyphCognitiveLoop(space)
    renderer = build_llm_renderer() or build_qwen_renderer()
    gen = GlyphGenerator(space, cognitive, renderer=renderer)

    print(f"  ingesting {len(FACTS)} held-out facts "
          f"(20 person / 15 capitals / 10 science / 5 ada)...")
    t0 = time.time()
    for f in FACTS:
        space.absorb(f)
    print(f"  ingested in {time.time() - t0:.1f}s. "
          f"Substrate holds {space.count} thoughts.\n")

    rows = []
    t0 = time.time()
    for i, (q, want_recall, want_answer) in enumerate(QUERIES, 1):
        result = gen.generate(q, top_chains=3, absorb_response=False, reinforce=False)
        top = result.top_recall or ""
        ok_recall = want_recall.lower() in top.lower()
        ans = (result.response or "").strip()
        ok_answer = want_answer.lower() in ans.lower()
        rows.append((q, top, ans, ok_recall, ok_answer))
        print(f"  [{i:>2}/20] {'✓' if ok_recall else '✗'}r "
              f"{'✓' if ok_answer else '✗'}a  {q}")
    elapsed = time.time() - t0
    print(f"\n  ran 20 queries in {elapsed:.1f}s "
          f"({elapsed/20:.1f}s avg)\n")

    print("─" * 100)
    print(f"{'QUERY':<40} {'TOP RECALL':<35} {'RENDERER':<20}")
    print("─" * 100)
    for q, top, ans, ok_r, ok_a in rows:
        rt = "✓" if ok_r else "✗"
        at = "✓" if ok_a else "✗"
        print(f"{q[:38]:<40} {rt} {top[:32]:<33} {at} {ans[:18]:<18}")
    print("─" * 100)

    r_correct = sum(1 for *_, ok_r, _ in rows if ok_r)
    a_correct = sum(1 for *_, _, ok_a in rows if ok_a)
    print(f"\n  HELD-OUT substrate top-1 recall : {r_correct}/20  "
          f"({100*r_correct/20:.0f}%)")
    print(f"  HELD-OUT renderer correct answer: {a_correct}/20  "
          f"({100*a_correct/20:.0f}%)")


if __name__ == "__main__":
    main()
