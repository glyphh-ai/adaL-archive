"""
Verify the infant curriculum: every composed_of target must resolve.

    PYTHONPATH=. python benchmark/curriculum/agi/infant/verify.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from teach import (INFANT_FILES, LAYER_0, LAYER_1, _load_yaml,
                   main as teach_main)


def main() -> None:
    space = teach_main()

    all_yaml_files = [LAYER_0, *LAYER_1, *INFANT_FILES]
    all_entries: list[dict] = []
    for p in all_yaml_files:
        all_entries.extend(_load_yaml(p))
    all_keys = {e.get("key") for e in all_entries if e.get("key")}

    infant_entries: list[dict] = []
    for p in INFANT_FILES:
        infant_entries.extend(_load_yaml(p))
    infant_keys = {e.get("key") for e in infant_entries if e.get("key")}

    # Coverage.
    missing = [k for k in infant_keys if k not in space._history_by_key]
    print(f"  ── coverage ──")
    print(f"    infant: {len(infant_keys) - len(missing)}/{len(infant_keys)} present"
          f"{' ✓' if not missing else f'  ✗ missing: {missing[:5]}'}")
    print()

    # Composition integrity.
    dangling: list[tuple[str, str]] = []
    n_compositions = 0
    depth_of: dict[str, int] = {}
    for e in infant_entries:
        key = e.get("key")
        for parent in e.get("composed_of") or []:
            n_compositions += 1
            if parent not in all_keys:
                dangling.append((key, parent))
    print(f"  ── composition integrity ──")
    print(f"    {n_compositions} infant composition links declared.")
    if dangling:
        print(f"    ✗ {len(dangling)} dangling references:")
        for from_k, to_k in dangling[:20]:
            print(f"        {from_k}  → {to_k}  (no such key)")
    else:
        print(f"    ✓ every infant composition target exists.")
    print()

    # Depth-to-primes analysis.
    print(f"  ── depth to NSM primes ──")
    # BFS each infant entry down to primes.
    by_key = {e["key"]: e for e in all_entries if e.get("key")}
    from collections import deque
    max_seen = 0
    depths = []
    for e in infant_entries:
        start = e["key"]
        # BFS until everything is prime.*
        visited = set()
        q = deque([(start, 0)])
        depth = 0
        ok = True
        while q:
            cur, d = q.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            depth = max(depth, d)
            if d > 12:
                ok = False
                break
            node = by_key.get(cur)
            if not node:
                continue
            if cur.startswith("prime."):
                continue
            for p in node.get("composed_of") or []:
                q.append((p, d + 1))
        if ok:
            depths.append(depth)
            max_seen = max(max_seen, depth)
    if depths:
        print(f"    mean depth from infant.* to NSM primes: {sum(depths)/len(depths):.1f}")
        print(f"    max depth: {max_seen}")
    print()

    # Sample composition chains.
    print(f"  ── sample compositions ──")
    samples = ["binary.absence", "infant.hunger", "infant.mother",
               "infant.object-permanence", "infant.empathy",
               "infant.question-why"]
    by_infant = {e["key"]: e.get("composed_of") or []
                 for e in infant_entries}
    for k in samples:
        if k in by_infant:
            print(f"    {k:<32} ← {by_infant[k]}")


if __name__ == "__main__":
    main()
