"""
explore.py — what can Ada do with layer-0 + layer-1 loaded?

Five demonstrations of substrate-native operations that no LLM can do
the same way (LLMs would have to text-reason about the same content;
Ada does these as structural retrievals).

NO LLM. Pure substrate inspection + role-weighted recall.

    PYTHONPATH=. python benchmark/curriculum/agi/layer-1/explore.py
"""

from __future__ import annotations

import sys
sys.path.insert(0, str(__file__.rsplit("/", 1)[0]))

from teach import main as teach_main


def main() -> None:
    space = teach_main()
    print()

    # ── 1. Composition traversal: "what is blue composed of?"
    print("─" * 70)
    print("  1. What primes is the concept 'color.blue' composed of?")
    print("─" * 70)
    composed = _composed_of(space, "color.blue")
    for parent in composed:
        print(f"     ← {parent}")
    print()

    # ── 2. Inverse: "what composites use prime.see?"
    print("─" * 70)
    print("  2. What higher-layer concepts compose from 'prime.see'?")
    print("─" * 70)
    children = _composers_of(space, "prime.see")
    for child in children:
        print(f"     → {child}")
    print()

    # ── 3. Enumerate a category from the substrate.
    print("─" * 70)
    print("  3. Enumerate everything Ada knows that is a 'color'.")
    print("─" * 70)
    colors = _by_entity_kind(space, "color")
    for c in colors:
        print(f"     • {c}")
    print()

    # ── 4. Role-level cross-fact similarity.
    print("─" * 70)
    print("  4. Find everything Ada knows where perceptual.temperature = warm.")
    print("─" * 70)
    warm = _by_role_value(space, "perceptual", "temperature", "warm")
    for w in warm:
        print(f"     • {w}")
    print()

    # ── 5. Composite chain: "trace orange back to its primes."
    print("─" * 70)
    print("  5. Trace 'color.orange' back to all the primes it depends on.")
    print("─" * 70)
    chain = _trace_to_primes(space, "color.orange")
    for level, items in enumerate(chain):
        print(f"     level {level}: {items}")
    print()

    print("─" * 70)
    print("  None of these required an LLM call. Each is a structural")
    print("  retrieval the substrate supports natively because every fact")
    print("  occupies the universal-schema lattice. An LLM with the same")
    print("  text in context could approximate them by reasoning — but")
    print("  not as cheap, not as auditable, not deterministic.")
    print("─" * 70)


# ── helpers ────────────────────────────────────────────────────────────

def _composed_of(space, key: str) -> list[str]:
    """Composition facts where this key is the subject."""
    result = []
    for t in space._thoughts.values():
        u = t.glyph.metadata.get("_universal") or {}
        rel = u.get("relational") or {}
        if rel.get("predicate") == "composed_of" and rel.get("subject") == key:
            result.append(rel.get("object", ""))
    return sorted(set(result))


def _composers_of(space, key: str) -> list[str]:
    """Composition facts where this key is the object — who composes from it?"""
    result = []
    for t in space._thoughts.values():
        u = t.glyph.metadata.get("_universal") or {}
        rel = u.get("relational") or {}
        if rel.get("predicate") == "composed_of" and rel.get("object") == key:
            result.append(rel.get("subject", ""))
    return sorted(set(result))


def _by_entity_kind(space, kind: str) -> list[str]:
    """All facts where entity.kind = X."""
    result = []
    for t in space._thoughts.values():
        u = t.glyph.metadata.get("_universal") or {}
        e = u.get("entity") or {}
        if e.get("kind") == kind:
            result.append(e.get("name", t.thought_id))
    return sorted(set(result))


def _by_role_value(space, layer: str, role: str, value: str) -> list[str]:
    """All facts where layer.role = value (exact match)."""
    result = []
    for t in space._thoughts.values():
        u = t.glyph.metadata.get("_universal") or {}
        if (u.get(layer) or {}).get(role) == value:
            e = u.get("entity") or {}
            result.append(e.get("name", t.thought_id))
    return sorted(set(result))


def _trace_to_primes(space, key: str, max_depth: int = 5) -> list[list[str]]:
    """BFS along composed_of edges until everything is a prime.* or we
    exceed max_depth. Returns one list per level."""
    levels: list[list[str]] = [[key]]
    current = {key}
    for _ in range(max_depth):
        next_level: set[str] = set()
        for k in current:
            if k.startswith("prime."):
                continue
            for parent in _composed_of(space, k):
                next_level.add(parent)
        if not next_level:
            break
        levels.append(sorted(next_level))
        current = next_level
        if all(k.startswith("prime.") for k in current):
            levels.append(sorted(current))   # final level (all primes)
            break
    return levels


if __name__ == "__main__":
    main()
