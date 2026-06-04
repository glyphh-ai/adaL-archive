"""
Load layer-1 (and layer-0 first) into a fresh substrate.

For each entry, two kinds of facts go in:
  1. The entry itself as a universal-schema glyph (tell_raw).
  2. One composition fact per item in `composed_of`, expressed as a
     relational binding (subject = entry key, predicate = composed_of,
     object = the referenced key). This makes composition queryable
     via standard recall.

NO LLM calls in either path.

    PYTHONPATH=. python benchmark/curriculum/agi/layer-1/teach.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from ada.memory.thought_space import ThoughtGlyphSpace
# Foundation primitives now live in the runtime core (ada/memory/foundation/).
from ada.memory.foundation import FOUNDATION_DIR

LAYER_0 = FOUNDATION_DIR / "primes.yaml"
LAYER_1_FILES = [
    FOUNDATION_DIR / f
    for f in ("colors.yaml", "numbers.yaml", "spatial.yaml",
              "logical.yaml", "temporal.yaml")
]


def _load_yaml(path: Path) -> list[dict]:
    if not path.exists():
        print(f"Missing {path}", file=sys.stderr)
        return []
    with path.open() as fh:
        return yaml.safe_load(fh) or []


def _ingest_entries(space: ThoughtGlyphSpace, entries: list[dict],
                    layer_label: str) -> tuple[int, int]:
    """Ingest a list of curriculum entries plus their composition facts.
    Returns (entries_loaded, composition_facts_loaded).
    """
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

        # Composition facts: one per parent prime/key.
        for parent in composed_of:
            comp_facts = {
                "entity": {"name": f"{key}.composes.{parent}", "kind": "composition"},
                "relational": {
                    "subject": key,
                    "predicate": "composed_of",
                    "object": parent,
                },
            }
            comp_key = f"composition.{key}.{parent}"
            comp_stored = space.tell_raw(
                facts=comp_facts, key=comp_key, speaker="curriculum",
                metadata={"category": "composition", "layer": layer_label,
                          "from": key, "to": parent},
            )
            if comp_stored is not None:
                n_comp += 1
    return n_entries, n_comp


def main() -> ThoughtGlyphSpace:
    space = ThoughtGlyphSpace(enricher=None)

    # Layer-0 first (so composition references resolve into existing keys).
    print("  ── layer-0: NSM primes ──")
    primes = _load_yaml(LAYER_0)
    n0, c0 = _ingest_entries(space, primes, "layer-0")
    print(f"    {n0} primes ingested, {c0} composition facts (primes are atomic).\n")

    # Layer-1.
    print("  ── layer-1 ──")
    grand_total = 0
    grand_comp = 0
    for path in LAYER_1_FILES:
        entries = _load_yaml(path)
        n, c = _ingest_entries(space, entries, "layer-1")
        grand_total += n
        grand_comp += c
        print(f"    {path.name:<18} {n:>3} entries, {c:>3} composition facts")
    print(f"\n  layer-1 totals: {grand_total} entries, {grand_comp} composition facts.")
    print(f"  substrate now holds {space.count} thoughts "
          f"({n0 + grand_total} concepts + {c0 + grand_comp} composition facts).\n")

    return space


if __name__ == "__main__":
    main()
