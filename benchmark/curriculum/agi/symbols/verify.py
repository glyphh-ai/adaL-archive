"""
Verify the symbol curriculum: every composed_of target must resolve.

    PYTHONPATH=. python benchmark/curriculum/agi/symbols/verify.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from teach import (INFANT_FILES, LAYER_0, LAYER_1, SYMBOL_FILES,
                   _load_yaml, main as teach_main)


def main() -> None:
    space = teach_main()

    all_yaml_files = [LAYER_0, *LAYER_1, *INFANT_FILES, *SYMBOL_FILES]
    all_entries: list[dict] = []
    for p in all_yaml_files:
        all_entries.extend(_load_yaml(p))
    all_keys = {e.get("key") for e in all_entries if e.get("key")}

    symbol_entries: list[dict] = []
    for p in SYMBOL_FILES:
        symbol_entries.extend(_load_yaml(p))
    symbol_keys = {e.get("key") for e in symbol_entries if e.get("key")}

    missing = [k for k in symbol_keys if k not in space._history_by_key]
    print(f"  ── coverage ──")
    print(f"    symbols: {len(symbol_keys) - len(missing)}/{len(symbol_keys)} present"
          f"{' ✓' if not missing else f'  ✗ missing: {missing[:5]}'}")
    print()

    dangling = []
    n_compositions = 0
    for e in symbol_entries:
        key = e.get("key")
        for parent in e.get("composed_of") or []:
            n_compositions += 1
            if parent not in all_keys:
                dangling.append((key, parent))
    print(f"  ── composition integrity ──")
    print(f"    {n_compositions} symbol composition links declared.")
    if dangling:
        print(f"    ✗ {len(dangling)} dangling references:")
        for from_k, to_k in dangling[:20]:
            print(f"        {from_k}  → {to_k}")
    else:
        print(f"    ✓ every symbol composition target exists.")
    print()

    print(f"  ── sample compositions ──")
    by_sym = {e["key"]: e.get("composed_of") or [] for e in symbol_entries}
    for k in ["digit.0", "digit.5", "letter.a", "letter.z",
              "phoneme.ae", "phoneme.k", "phoneme.ch", "phoneme.ng"]:
        if k in by_sym:
            print(f"    {k:<22} ← {by_sym[k]}")


if __name__ == "__main__":
    main()
