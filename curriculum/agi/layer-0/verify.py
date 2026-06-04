"""
Verify layer-0 — does the substrate actually contain the NSM primes?

Loads the curriculum, then asks Ada to enumerate primes by category via
role-weighted recall on the `entity.subkind` slot. No LLM in the path.

    PYTHONPATH=. python curriculum/agi/layer-0/verify.py
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ada.cognitive.universal import UniversalSpace
from ada.memory.thought_space import ThoughtGlyphSpace
from ada.memory.foundation import FOUNDATION_DIR

PRIMES_FILE = FOUNDATION_DIR / "primes.yaml"


def main() -> None:
    # Load the curriculum into a fresh substrate first.
    with PRIMES_FILE.open() as fh:
        entries = yaml.safe_load(fh) or []

    space = ThoughtGlyphSpace(enricher=None)
    expected_by_category: dict[str, list[str]] = {}
    for e in entries:
        cat = e.get("category", "uncategorized")
        expected_by_category.setdefault(cat, []).append(e.get("key", ""))
        facts = {k: v for k, v in e.items()
                 if k not in ("key", "category", "source") and isinstance(v, dict)}
        space.tell_raw(facts=facts, key=e.get("key"), speaker="curriculum")

    print(f"  loaded {space.count} primes into substrate.\n")
    print(f"  expected categories ({len(expected_by_category)}):")
    for cat, keys in sorted(expected_by_category.items()):
        print(f"    {cat:<25} {len(keys)}")
    print()

    # ── Pure-substrate verification — enumerate by entity.subkind ──
    # We walk the stored thoughts and group by their _universal slot
    # fill. No LLM, no enricher, no rendering. This is what 'ask Ada
    # about her own primes' looks like as a pure HDC operation.
    found_by_subkind: dict[str, list[str]] = {}
    for thought in space._thoughts.values():
        universal = thought.glyph.metadata.get("_universal") or {}
        subkind = (universal.get("entity") or {}).get("subkind", "unknown")
        name = (universal.get("entity") or {}).get("name", thought.thought_id)
        found_by_subkind.setdefault(subkind, []).append(name)

    print(f"  found_by_subkind (substrate-native enumeration):")
    for subkind, names in sorted(found_by_subkind.items()):
        print(f"    {subkind:<25} {len(names):>3}  e.g. {', '.join(names[:4])}")
    print()

    # Coverage check.
    missing: list[str] = []
    expected_keys = {e.get("key") for e in entries}
    for k in expected_keys:
        if not k:
            continue
        if k not in space._history_by_key:
            missing.append(k)
    if missing:
        print(f"  ✗ MISSING {len(missing)} expected primes: {missing[:10]}")
    else:
        print(f"  ✓ all {len(expected_keys)} expected primes present.")


if __name__ == "__main__":
    main()
