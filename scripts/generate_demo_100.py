"""
Scaled demo — 100 facts, 20 queries, substrate + Qwen3-0.6B.

Measures both substrate top-1 recall accuracy AND renderer answer
correctness. Prints a table summarising what worked.

Run:
    PYTHONPATH=. python scripts/generate_demo_100.py
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


# ── 100-fact corpus ──────────────────────────────────────────────────

# 10 people × 4 attributes = 40 facts.
PEOPLE = [
    ("Chris",  31, "Austin",    "engineer",     "blue"),
    ("Alice",  27, "Boston",    "doctor",       "green"),
    ("Bob",    45, "Chicago",   "teacher",      "red"),
    ("Diana",  33, "Denver",    "designer",     "purple"),
    ("Eli",    52, "Eugene",    "scientist",    "orange"),
    ("Fran",   29, "Fresno",    "lawyer",       "yellow"),
    ("Greta",  41, "Glasgow",   "pilot",        "white"),
    ("Hugo",   38, "Helsinki",  "chef",         "black"),
    ("Iris",   24, "Istanbul",  "writer",       "pink"),
    ("Jonah",  60, "Jakarta",   "musician",     "brown"),
]
PERSON_FACTS: list[str] = []
for name, age, city, job, color in PEOPLE:
    PERSON_FACTS.append(f"{name} is {age} years old.")
    PERSON_FACTS.append(f"{name} lives in {city}.")
    PERSON_FACTS.append(f"{name} works as a {job}.")
    PERSON_FACTS.append(f"{name}'s favorite color is {color}.")

# 30 country → capital pairs.
CAPITALS = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"),
    ("Germany", "Berlin"), ("Spain", "Madrid"), ("Portugal", "Lisbon"),
    ("Greece", "Athens"), ("Egypt", "Cairo"), ("Kenya", "Nairobi"),
    ("Brazil", "Brasilia"), ("Argentina", "Buenos Aires"), ("Chile", "Santiago"),
    ("Canada", "Ottawa"), ("Mexico", "Mexico City"), ("Cuba", "Havana"),
    ("India", "New Delhi"), ("China", "Beijing"), ("Thailand", "Bangkok"),
    ("Vietnam", "Hanoi"), ("Indonesia", "Jakarta"), ("Australia", "Canberra"),
    ("Russia", "Moscow"), ("Ukraine", "Kyiv"), ("Poland", "Warsaw"),
    ("Sweden", "Stockholm"), ("Norway", "Oslo"), ("Finland", "Helsinki"),
    ("Iceland", "Reykjavik"), ("Ireland", "Dublin"), ("Iran", "Tehran"),
]
CAPITAL_FACTS = [f"The capital of {c} is {cap}." for c, cap in CAPITALS]

# 20 science / math constants.
SCIENCE_FACTS = [
    "The speed of light is 299,792,458 meters per second.",
    "Water boils at 100 degrees Celsius.",
    "Water freezes at 0 degrees Celsius.",
    "Pi is approximately 3.14159.",
    "Earth's gravity is 9.81 meters per second squared.",
    "The Earth orbits the Sun in 365 days.",
    "The Moon orbits the Earth in 27 days.",
    "Light travels from the Sun to Earth in 8 minutes.",
    "A water molecule has two hydrogen atoms and one oxygen atom.",
    "Gold has atomic number 79.",
    "Hydrogen is the lightest element.",
    "Diamond is the hardest natural mineral.",
    "Sound travels at 343 meters per second in air.",
    "Absolute zero is -273.15 degrees Celsius.",
    "Avogadro's number is 6.022 times 10 to the 23.",
    "The human body has 206 bones.",
    "DNA stands for deoxyribonucleic acid.",
    "The mitochondria is the powerhouse of the cell.",
    "There are 118 elements in the periodic table.",
    "Photosynthesis converts sunlight into chemical energy.",
]

# 10 Ada / HDC technical facts.
ADA_FACTS = [
    "Ada uses 2048-dimensional bipolar vectors.",
    "HDC stands for hyperdimensional computing.",
    "Ada's atoms are encoded with character n-grams.",
    "Bindings combine atoms through XOR.",
    "Bundling aggregates vectors by majority vote.",
    "The Ada server runs on port 8002.",
    "Each thought has five hierarchical layers.",
    "Glyph similarity uses weighted cortex, layer, segment, and role.",
    "Ada has a REM and slow-wave dream loop for pattern discovery.",
    "Hebbian reinforcement strengthens recalled memories.",
]

FACTS: list[str] = PERSON_FACTS + CAPITAL_FACTS + SCIENCE_FACTS + ADA_FACTS
assert len(FACTS) == 100, f"expected 100 facts, got {len(FACTS)}"


# ── 20-query test set with expected substance ───────────────────────

# (query, expected_keyword_in_top_recall, expected_keyword_in_renderer_answer)
QUERIES: list[tuple[str, str, str]] = [
    # Person attribute lookups
    ("What is Alice's age?",                 "Alice is 27",       "27"),
    ("Where does Bob live?",                 "Bob lives in",      "Chicago"),
    ("What job does Diana have?",            "Diana works",       "designer"),
    ("What is Hugo's favorite color?",       "Hugo's favorite",   "black"),
    ("How old is Jonah?",                    "Jonah is 60",       "60"),

    # Capitals
    ("What is the capital of Japan?",        "Japan is Tokyo",    "Tokyo"),
    ("Where is the capital of Egypt?",       "Egypt is Cairo",    "Cairo"),
    ("Capital of Iceland?",                  "Iceland",           "Reykjavik"),
    ("What is Brazil's capital?",            "Brazil",            "Brasilia"),
    ("Capital city of China?",               "China is Beijing",  "Beijing"),

    # Science
    ("What is the speed of light?",          "speed of light",    "299"),
    ("How many bones in the human body?",    "206 bones",         "206"),
    ("What does DNA stand for?",             "deoxyribonucleic",  "deoxyribonucleic"),
    ("At what temperature does water boil?", "Water boils",       "100"),
    ("How long is a year on Earth?",         "Earth orbits",      "365"),

    # Ada / HDC
    ("What dimension are Ada's vectors?",    "2048",              "2048"),
    ("How does Ada bind atoms?",             "XOR",               "XOR"),
    ("What port does Ada run on?",           "port 8002",         "8002"),
    ("How many layers per thought?",         "five hierarchical", "five"),
    ("What is hyperdimensional computing?",  "HDC stands",        "hyperdimensional"),
]


# ── runner ──────────────────────────────────────────────────────────

def main() -> None:
    enricher = auto_enricher()
    space = ThoughtGlyphSpace(enricher=enricher)
    cognitive = GlyphCognitiveLoop(space)
    renderer = build_llm_renderer() or build_qwen_renderer()
    gen = GlyphGenerator(space, cognitive, renderer=renderer)

    print(f"  ingesting {len(FACTS)} facts (4 domains: 40 people / 30 capitals / 20 science / 10 ada)...")
    t0 = time.time()
    for f in FACTS:
        space.absorb(f)
    print(f"  ingested in {time.time() - t0:.1f}s. Substrate holds {space.count} thoughts.\n")

    if renderer is None:
        print("  (no renderer wired — set ANTHROPIC_API_KEY or run after `python -c ...` to download Qwen3)\n")

    rows: list[tuple[str, str, str, bool, bool]] = []
    t0 = time.time()
    for i, (q, want_recall, want_answer) in enumerate(QUERIES, 1):
        result = gen.generate(q, top_chains=3, absorb_response=False, reinforce=False)
        top = result.top_recall or ""
        ok_recall = want_recall.lower() in top.lower()
        ans = (result.response or "").strip()
        ok_answer = want_answer.lower() in ans.lower()
        rows.append((q, top, ans, ok_recall, ok_answer))
        print(f"  [{i:>2}/20] {'✓' if ok_recall else '✗'}r {'✓' if ok_answer else '✗'}a  {q}")
    elapsed = time.time() - t0
    print(f"\n  ran 20 queries in {elapsed:.1f}s ({elapsed/20:.1f}s avg per query)\n")

    # ── summary table ──
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
    print(f"\n  substrate top-1 recall : {r_correct}/20  ({100*r_correct/20:.0f}%)")
    print(f"  renderer correct answer: {a_correct}/20  ({100*a_correct/20:.0f}%)")


if __name__ == "__main__":
    main()
