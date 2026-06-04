"""
Verify chemistry: every entry resolves, every composition target exists.
Pure substrate inspection — no LLM.

    PYTHONPATH=. python benchmark/curriculum/agi/chemistry/verify.py
"""

from __future__ import annotations

import sys
sys.path.insert(0, str(__file__.rsplit("/", 1)[0]))

from teach import (main as teach_main, CHEM_FILES, LAYER_0, LAYER_1,
                   _load_yaml)


def main() -> None:
    space = teach_main()

    all_yaml_files = [LAYER_0, *LAYER_1, *CHEM_FILES]
    all_entries: list[dict] = []
    for p in all_yaml_files:
        all_entries.extend(_load_yaml(p))
    all_keys = {e.get("key") for e in all_entries if e.get("key")}

    chem_entries: list[dict] = []
    for p in CHEM_FILES:
        chem_entries.extend(_load_yaml(p))
    chem_keys = {e.get("key") for e in chem_entries if e.get("key")}

    # Coverage.
    missing = [k for k in chem_keys if k not in space._history_by_key]
    print(f"  ── coverage ──")
    print(f"    chemistry: {len(chem_keys) - len(missing)}/{len(chem_keys)} present"
          f"{' ✓' if not missing else f'  ✗ missing: {missing[:5]}'}")
    print()

    # Composition integrity.
    dangling: list[tuple[str, str]] = []
    n_compositions = 0
    by_key: dict[str, list[str]] = {}
    for e in chem_entries:
        key = e.get("key")
        for parent in e.get("composed_of") or []:
            n_compositions += 1
            by_key.setdefault(key, []).append(parent)
            if parent not in all_keys:
                dangling.append((key, parent))
    print(f"  ── composition integrity ──")
    print(f"    {n_compositions} chemistry composition links declared.")
    if dangling:
        print(f"    ✗ {len(dangling)} dangling references:")
        for from_k, to_k in dangling[:10]:
            print(f"        {from_k}  → {to_k}  (no such key)")
    else:
        print(f"    ✓ every chemistry composition target exists.")
    print()

    # Sample compositions.
    print(f"  ── sample composition chains ──")
    for k in ["element.carbon", "bond.hydrogen", "group.carboxyl",
              "compound.water", "compound.glucose", "compound.atp",
              "property.water.bp", "reaction.photosynthesis"]:
        parents = by_key.get(k, [])
        print(f"    {k:<28} ← {parents}")


if __name__ == "__main__":
    main()
