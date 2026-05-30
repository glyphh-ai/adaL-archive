"""
generate.py — programmatically generate a large structured corpus.

Defaults: 1000 named persons, each with 20 universal-schema facts about
them = 20,000 facts. At ~25 tokens per fact (the way they'd appear in an
LLM context), that's ~500K tokens — more than 2× Claude Haiku 4.5's
200K-token window, and within the 1M for the larger models. Push --n to
100,000 entities to put it firmly out of reach of every public LLM.

The point: the corpus that Ada handles natively cannot fit in any single
LLM's context. We can then run identical queries on Ada vs LLM-only and
show that the LLM doesn't get to see the answer because the answer
isn't in its prompt.

Generation is deterministic given a seed so the same corpus can be
regenerated for the LLM baseline.

    PYTHONPATH=. python curriculum/at-scale/generate.py --n 1000 --out corpus.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# 11 colors, 30 cities, 20 jobs, 11 hobbies — gives us a known ground-truth
# distribution we can query against. Names are synthetic but distinct.

COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "pink",
          "brown", "black", "white", "grey"]

CITIES = ["Austin", "Boston", "Chicago", "Denver", "Eugene", "Fresno",
          "Glasgow", "Helsinki", "Istanbul", "Jakarta", "Kyoto", "Lagos",
          "Madrid", "Naples", "Oslo", "Paris", "Quito", "Riga",
          "Stockholm", "Tokyo", "Uppsala", "Vienna", "Warsaw", "Xi'an",
          "York", "Zurich", "Phoenix", "Calgary", "Lyon", "Antwerp"]

JOBS = ["engineer", "doctor", "teacher", "designer", "scientist",
        "lawyer", "pilot", "chef", "writer", "musician",
        "architect", "nurse", "carpenter", "plumber", "electrician",
        "accountant", "farmer", "actor", "athlete", "researcher"]

HOBBIES = ["chess", "painting", "hiking", "gardening", "cooking",
           "running", "swimming", "reading", "music", "photography",
           "knitting"]

PETS = ["dog", "cat", "rabbit", "hamster", "bird", "fish", "none"]

FOODS = ["pizza", "pasta", "sushi", "tacos", "curry", "salad",
         "soup", "burger", "rice", "noodles"]

DRINKS = ["coffee", "tea", "water", "juice", "wine", "beer"]


def _name_for(i: int, rng: random.Random) -> str:
    """Deterministic-ish synthetic name from a tiny phoneme set + index."""
    syllables = ["ka", "ra", "no", "mi", "to", "le", "vi", "su", "do", "an",
                 "be", "ji", "po", "rey", "fa", "yu", "ze", "qu", "wa", "ho"]
    parts = rng.choice(syllables) + rng.choice(syllables)
    return parts.title() + str(i)


def generate(n: int = 1000, seed: int = 0) -> list[dict]:
    """Generate n people × 20 structured facts each = 20n facts."""
    rng = random.Random(seed)
    out: list[dict] = []
    for i in range(n):
        name = _name_for(i, rng)
        age = rng.randint(18, 80)
        color = rng.choice(COLORS)
        city = rng.choice(CITIES)
        job = rng.choice(JOBS)
        hobby = rng.choice(HOBBIES)
        pet = rng.choice(PETS)
        food = rng.choice(FOODS)
        drink = rng.choice(DRINKS)
        height_cm = rng.randint(150, 200)
        siblings = rng.randint(0, 5)
        years_in_city = rng.randint(0, 30)
        pet_years = rng.randint(0, 15) if pet != "none" else 0

        # 20 universal-schema facts per person.
        nlow = name.lower()
        out.append({"key": f"person.{nlow}.age",
                    "entity": {"name": name, "kind": "person"},
                    "temporal": {"age": str(age)}})
        out.append({"key": f"person.{nlow}.color",
                    "entity": {"name": name, "kind": "person"},
                    "perceptual": {"color": color}})
        out.append({"key": f"person.{nlow}.city",
                    "entity": {"name": name, "kind": "person"},
                    "spatial": {"location": city}})
        out.append({"key": f"person.{nlow}.job",
                    "entity": {"name": name, "kind": "person"},
                    "relational": {"predicate": "works_as", "object": job}})
        out.append({"key": f"person.{nlow}.hobby",
                    "entity": {"name": name, "kind": "person"},
                    "relational": {"predicate": "hobby", "object": hobby}})
        out.append({"key": f"person.{nlow}.pet",
                    "entity": {"name": name, "kind": "person"},
                    "relational": {"predicate": "pet", "object": pet}})
        out.append({"key": f"person.{nlow}.food",
                    "entity": {"name": name, "kind": "person"},
                    "relational": {"predicate": "favorite_food", "object": food}})
        out.append({"key": f"person.{nlow}.drink",
                    "entity": {"name": name, "kind": "person"},
                    "relational": {"predicate": "drinks", "object": drink}})
        out.append({"key": f"person.{nlow}.height",
                    "entity": {"name": name, "kind": "person"},
                    "quantitative": {"magnitude": str(height_cm),
                                     "unit": "centimeters"}})
        out.append({"key": f"person.{nlow}.siblings",
                    "entity": {"name": name, "kind": "person"},
                    "quantitative": {"count": str(siblings),
                                     "unit": "siblings"}})
        out.append({"key": f"person.{nlow}.tenure",
                    "entity": {"name": name, "kind": "person"},
                    "temporal": {"duration": str(years_in_city),
                                 "frequency": "years_in_city"}})
        out.append({"key": f"person.{nlow}.pet_years",
                    "entity": {"name": name, "kind": "person"},
                    "temporal": {"duration": str(pet_years),
                                 "frequency": "years_with_pet"}})
        # And 8 more compositional facts about their preferences
        out.append({"key": f"person.{nlow}.travel",
                    "entity": {"name": name, "kind": "person"},
                    "relational": {"predicate": "wants_to_visit",
                                   "object": rng.choice(CITIES)}})
        out.append({"key": f"person.{nlow}.dislikes",
                    "entity": {"name": name, "kind": "person"},
                    "relational": {"predicate": "dislikes",
                                   "object": rng.choice(FOODS)}})
        out.append({"key": f"person.{nlow}.morning",
                    "entity": {"name": name, "kind": "person"},
                    "temporal": {"time": "morning"},
                    "relational": {"predicate": "wakes_at",
                                   "object": f"{rng.randint(5,9)}am"}})
        out.append({"key": f"person.{nlow}.bed",
                    "entity": {"name": name, "kind": "person"},
                    "temporal": {"time": "evening"},
                    "relational": {"predicate": "sleeps_at",
                                   "object": f"{rng.randint(9,12)}pm"}})
        out.append({"key": f"person.{nlow}.exercise",
                    "entity": {"name": name, "kind": "person"},
                    "relational": {"predicate": "exercises",
                                   "object": rng.choice(["daily", "weekly", "rarely", "never"])}})
        out.append({"key": f"person.{nlow}.tv",
                    "entity": {"name": name, "kind": "person"},
                    "relational": {"predicate": "watches",
                                   "object": rng.choice(["news", "sports", "movies", "shows", "none"])}})
        out.append({"key": f"person.{nlow}.car_color",
                    "entity": {"name": name, "kind": "person"},
                    "perceptual": {"color": rng.choice(COLORS)},
                    "relational": {"predicate": "owns",
                                   "object": "car"}})
        out.append({"key": f"person.{nlow}.house_size",
                    "entity": {"name": name, "kind": "person"},
                    "perceptual": {"size": rng.choice(["small", "medium", "large"])},
                    "relational": {"predicate": "lives_in",
                                   "object": "house"}})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000,
                   help="number of synthetic persons (×20 = facts)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="curriculum/at-scale/corpus.json")
    args = p.parse_args()

    facts = generate(args.n, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(facts, fh, ensure_ascii=False)

    # Approximate token count if this were sent to an LLM in context.
    approx_chars = sum(len(json.dumps(f)) for f in facts)
    approx_tokens = approx_chars // 4

    print(f"  generated {len(facts):,} facts about {args.n:,} persons")
    print(f"  saved → {args.out}")
    print(f"  approx LLM token count: {approx_tokens:,}")
    if approx_tokens > 200_000:
        print(f"  ⚠  EXCEEDS Claude Haiku 4.5 context window (200K)")
    if approx_tokens > 1_000_000:
        print(f"  ⚠  EXCEEDS Claude Sonnet 4.6 context window (1M)")


if __name__ == "__main__":
    main()
