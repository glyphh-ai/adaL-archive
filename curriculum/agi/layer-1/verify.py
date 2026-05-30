"""
Verify layer-1: every layer-1 entry resolves, and every composition
points at a key that actually exists in the substrate.

No LLM. Pure substrate inspection.

    PYTHONPATH=. python curriculum/agi/layer-1/verify.py
"""

from __future__ import annotations

import sys
sys.path.insert(0, str(__file__.rsplit("/", 1)[0]))

from teach import main as teach_main, LAYER_1_FILES, LAYER_0, _load_yaml


def main() -> None:
    space = teach_main()

    # Pull all expected keys from yaml.
    layer_0_keys = {e.get("key") for e in _load_yaml(LAYER_0)}
    layer_1_entries = []
    for path in LAYER_1_FILES:
        layer_1_entries.extend(_load_yaml(path))
    layer_1_keys = {e.get("key") for e in layer_1_entries}

    # Coverage check.
    missing_0 = [k for k in layer_0_keys if k and k not in space._history_by_key]
    missing_1 = [k for k in layer_1_keys if k and k not in space._history_by_key]
    print(f"  ── coverage ──")
    print(f"    layer-0: {len(layer_0_keys) - len(missing_0)}/{len(layer_0_keys)} present"
          f"{' ✓' if not missing_0 else f'  ✗ missing: {missing_0[:5]}'}")
    print(f"    layer-1: {len(layer_1_keys) - len(missing_1)}/{len(layer_1_keys)} present"
          f"{' ✓' if not missing_1 else f'  ✗ missing: {missing_1[:5]}'}")
    print()

    # Composition integrity: every `composed_of` reference points at an
    # actual key.
    all_keys = layer_0_keys | layer_1_keys
    dangling: list[tuple[str, str]] = []
    n_compositions = 0
    by_layer1_key: dict[str, list[str]] = {}
    for e in layer_1_entries:
        key = e.get("key")
        for parent in e.get("composed_of") or []:
            n_compositions += 1
            by_layer1_key.setdefault(key, []).append(parent)
            if parent not in all_keys:
                dangling.append((key, parent))
    print(f"  ── composition integrity ──")
    print(f"    {n_compositions} composition links declared.")
    if dangling:
        print(f"    ✗ {len(dangling)} dangling references:")
        for from_k, to_k in dangling[:8]:
            print(f"        {from_k}  → {to_k}  (no such key)")
    else:
        print(f"    ✓ every composition target exists.")
    print()

    # Sample some composition chains.
    print(f"  ── sample composition chains ──")
    for sample_key in ["color.blue", "color.purple", "number.5",
                       "spatial.east", "logic.xor", "time.year"]:
        parents = by_layer1_key.get(sample_key, [])
        print(f"    {sample_key:<16} composed_of: {parents}")


if __name__ == "__main__":
    main()
