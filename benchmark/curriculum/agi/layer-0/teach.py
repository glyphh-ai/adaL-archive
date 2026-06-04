"""
Load the NSM primes (layer-0) into a fresh substrate. NO LLM calls.

Each entry in primes.yaml becomes one tell_raw() call. The substrate
ends with ~65 versioned primes, keyed `prime.<name>`, ready to be
referenced by any higher-layer curriculum.

    PYTHONPATH=. python benchmark/curriculum/agi/layer-0/teach.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from ada.memory.thought_space import ThoughtGlyphSpace
from ada.memory.foundation import FOUNDATION_DIR

PRIMES_FILE = FOUNDATION_DIR / "primes.yaml"


def main() -> None:
    if not PRIMES_FILE.exists():
        print(f"Missing {PRIMES_FILE}", file=sys.stderr)
        sys.exit(1)

    with PRIMES_FILE.open() as fh:
        entries = yaml.safe_load(fh) or []

    print(f"  loading {len(entries)} NSM primes into a fresh substrate (no LLM)...\n")

    # Construct a substrate without an enricher — we don't need one and
    # don't want one for raw curriculum ingestion.
    space = ThoughtGlyphSpace(enricher=None)

    loaded = 0
    skipped = 0
    by_category: dict[str, int] = {}
    for e in entries:
        key = e.get("key")
        if not key:
            skipped += 1
            continue
        category = e.get("category", "uncategorized")
        # Strip non-schema metadata before handing to tell_raw.
        meta_only = {"category": category, "source": e.get("source", "")}
        facts = {k: v for k, v in e.items()
                 if k not in ("key", "category", "source") and isinstance(v, dict)}
        stored = space.tell_raw(facts=facts, key=key, speaker="curriculum",
                                metadata=meta_only)
        if stored is None:
            skipped += 1
            print(f"    ✗ {key} (duplicate or empty)")
            continue
        loaded += 1
        by_category[category] = by_category.get(category, 0) + 1
        print(f"    ✓ {stored.thought_id}  ← {key}")

    print(f"\n  loaded {loaded}/{len(entries)} primes, skipped {skipped}.")
    print(f"  substrate now holds {space.count} thoughts.\n")
    print("  by category:")
    for cat in sorted(by_category):
        print(f"    {cat:<25} {by_category[cat]}")


if __name__ == "__main__":
    main()
