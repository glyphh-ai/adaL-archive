"""
Versioning demo — same key, multiple versions, observed drift.

Wires the existing ada/temporal/delta.py math into the absorb path so a
single stable key (e.g. "chris.age") can carry a version chain with
computed deltas between consecutive vectors.

    PYTHONPATH=. python scripts/versioning_demo.py
"""

from __future__ import annotations

from ada.encoder.llm_enricher import auto_enricher
from ada.memory.thought_space import ThoughtGlyphSpace


def main() -> None:
    enricher = auto_enricher()
    space = ThoughtGlyphSpace(enricher=enricher)

    # Same concept, evolving over time.
    timeline = [
        ("chris.age",       "Chris is 31 years old."),
        ("chris.age",       "Chris is 32 years old."),
        ("chris.age",       "Chris is 33 years old."),
        ("chris.location",  "Chris lives in Austin."),
        ("chris.location",  "Chris lives in Brooklyn."),
        ("chris.location",  "Chris lives in Tokyo."),
        ("ada.architecture","Ada uses 2048-dimensional bipolar vectors."),
        ("ada.architecture","Ada uses 4096-dimensional bipolar vectors."),
    ]

    print("─── ingesting versioned facts ───\n")
    for key, text in timeline:
        stored = space.absorb(text, key=key)
        if stored is None:
            print(f"   (dup) {text}")
            continue
        v = stored.metadata.get("_version")
        parent = stored.metadata.get("_parent_id") or "—"
        delta = stored.metadata.get("_delta")
        delta_tag = "[delta]" if delta is not None else "[v1]"
        print(f"   {stored.thought_id}  {delta_tag}  parent={parent}")
        print(f"      ↳ {text}")
    print()

    print("─── histories ───\n")
    for key in {k for k, _ in timeline}:
        hist = space.history(key)
        print(f"   {key!r}  ({len(hist)} versions)")
        for st in hist:
            print(f"      v{st.metadata['_version']}: {st.content}")
    print()

    print("─── drift ───\n")
    for key in {k for k, _ in timeline}:
        d = space.drift(key)
        if d.get("drift") is None and d.get("drift_rate") is None:
            print(f"   {key!r}: only one version")
            continue
        cons = ", ".join(f"{s:+.3f}" for s in d["consecutive_similarities"])
        print(f"   {key!r}")
        print(f"      consecutive sims: [{cons}]")
        print(f"      first→last sim:   {d['first_to_last_similarity']:+.3f}")
        print(f"      drift_rate:       {d['drift_rate']:+.3f}")
    print()


if __name__ == "__main__":
    main()
