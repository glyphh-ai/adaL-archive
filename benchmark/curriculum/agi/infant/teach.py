"""
Load the infant developmental curriculum into a substrate already
preloaded with layer-0 (NSM primes) and layer-1 (cognitive primitives).

The chain:
   layer-0 primes
     → stage-0 binary distinctions
       → stage-1 sensations
         → stage-2 perception
           → stage-3 categories (people, objects, body parts)
             → stage-4 causality
               → stage-5 object permanence + spatial
                 → stage-6 intent + proto-language
                   → stage-7 intentional actions
                     → stage-8 first words
                       → stage-9 possession + self
                         → stage-10 differentiated emotions
                           → stage-11 social cognition
                             → stage-12 combinatorial cognition

NO LLM at ingest. Every fact arrives via tell_raw.

    PYTHONPATH=. python benchmark/curriculum/agi/infant/teach.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from ada.memory.thought_space import ThoughtGlyphSpace

from ada.memory.foundation import FOUNDATION_DIR

ROOT = Path(__file__).parent.parent
LAYER_0 = FOUNDATION_DIR / "primes.yaml"
LAYER_1 = [
    FOUNDATION_DIR / f
    for f in ("colors.yaml", "numbers.yaml", "spatial.yaml",
              "logical.yaml", "temporal.yaml")
]
INFANT_FILES = sorted((Path(__file__).parent).glob("stage-*.yaml"),
                      key=lambda p: int(p.stem.split("_")[0].split("-")[1]))


def _load_yaml(path: Path) -> list[dict]:
    if not path.exists():
        print(f"Missing {path}", file=sys.stderr)
        return []
    with path.open() as fh:
        return yaml.safe_load(fh) or []


def _ingest_entries(space: ThoughtGlyphSpace, entries: list[dict],
                    layer_label: str) -> tuple[int, int]:
    n_entries = 0
    n_comp = 0
    for e in entries:
        key = e.get("key")
        if not key:
            continue
        category = e.get("category", "uncategorized")
        composed_of = e.get("composed_of") or []
        facts = {k: v for k, v in e.items()
                 if k not in ("key", "category", "source", "composed_of")
                 and isinstance(v, dict)}
        meta = {"category": category, "source": e.get("source", ""),
                "layer": layer_label}
        stored = space.tell_raw(facts=facts, key=key,
                                speaker="curriculum", metadata=meta)
        if stored is None:
            continue
        n_entries += 1

        for parent in composed_of:
            comp_facts = {
                "entity": {"name": f"{key}.composes.{parent}",
                           "kind": "composition"},
                "relational": {
                    "subject": key,
                    "predicate": "composed_of",
                    "object": parent,
                },
            }
            comp_key = f"composition.{key}.{parent}"
            stored = space.tell_raw(
                facts=comp_facts, key=comp_key, speaker="curriculum",
                metadata={"category": "composition", "layer": layer_label,
                          "from": key, "to": parent},
            )
            if stored is not None:
                n_comp += 1
    return n_entries, n_comp


def main() -> ThoughtGlyphSpace:
    space = ThoughtGlyphSpace(enricher=None)

    print("  ── layer-0: NSM primes ──")
    n0, c0 = _ingest_entries(space, _load_yaml(LAYER_0), "layer-0")
    print(f"    {n0} primes ingested.\n")

    print("  ── layer-1: cognitive primitives ──")
    n1 = c1 = 0
    for path in LAYER_1:
        n, c = _ingest_entries(space, _load_yaml(path), "layer-1")
        n1 += n
        c1 += c
        print(f"    {path.name:<16} {n:>3} entries, {c:>3} comp facts")
    print(f"    layer-1 totals: {n1} entries, {c1} composition facts.\n")

    print("  ── infant developmental curriculum ──")
    ni = ci = 0
    for path in INFANT_FILES:
        entries = _load_yaml(path)
        n, c = _ingest_entries(space, entries, "infant")
        ni += n
        ci += c
        print(f"    {path.name:<28} {n:>3} entries, {c:>3} comp facts")
    print(f"    infant totals: {ni} entries, {ci} composition facts.\n")

    print(f"  substrate now holds {space.count} thoughts "
          f"({n0 + n1 + ni} concepts + {c0 + c1 + ci} composition facts).\n")

    return space


if __name__ == "__main__":
    main()
