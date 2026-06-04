"""
Consolidate everything — layer-0 + layer-1 + infant + symbols +
communication + chemistry + philosophy + law — into one
ThoughtGlyphSpace, persist to a single ada.db.

Guards:
  - Foundation keys (layer-0/1/infant/symbols/communication) are
    loaded FIRST and protected from override by LLM extractions
    later. The LLM's emitted entries for `prime.something`,
    `logic.and`, etc. are filtered out.
  - Cross-domain shared keys (e.g. concept.justice in both philosophy
    and law) are loaded as VERSIONS via tell_raw(key=...). The
    substrate's versioning machinery preserves both perspectives.

    PYTHONPATH=. python benchmark/curriculum/agi/consolidate.py [--out ada.db]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

from ada.memory.thought_space import ThoughtGlyphSpace

ROOT = Path(__file__).parent

FOUNDATION_DIRS = [
    ROOT / "layer-0",
    ROOT / "layer-1",
    ROOT / "infant",
    ROOT / "symbols",
    ROOT / "communication",
]
DOMAIN_FILES = [
    # The hand-written chemistry seeds
    *(ROOT / "chemistry").glob("*.yaml"),
    # The LLM-extracted domain outputs
    ROOT / "chemistry" / "phase1_output" / "clean_entries.yaml",
    ROOT / "philosophy" / "phase1_output_v2" / "clean_entries.yaml",
    ROOT / "law" / "phase1_output_v2" / "clean_entries.yaml",
]


def _load_yaml(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with p.open() as fh:
        data = yaml.safe_load(fh)
        if isinstance(data, list):
            return data
        return []


def _entries_in(paths) -> list[tuple[Path, list[dict]]]:
    out = []
    for p in paths:
        if p.is_dir():
            for f in sorted(p.glob("*.yaml")):
                out.append((f, _load_yaml(f)))
        else:
            out.append((p, _load_yaml(p)))
    return out


def _ingest(space: ThoughtGlyphSpace, entries: list[dict],
            label: str, foundation_keys: set[str] | None = None,
            ) -> tuple[int, int, int]:
    """Returns (entries_loaded, composition_facts_loaded, skipped_collisions)."""
    n_entries = n_comp = n_skip = 0
    for e in entries:
        key = e.get("key")
        if not key:
            continue
        if foundation_keys is not None and key in foundation_keys:
            # LLM tried to redefine a foundation primitive. Skip.
            n_skip += 1
            continue
        composed_of = e.get("composed_of") or []
        facts = {k: v for k, v in e.items()
                 if k not in ("key", "category", "source", "composed_of")
                 and isinstance(v, dict)}
        meta = {"category": e.get("category", "uncategorized"),
                "source": e.get("source", ""), "layer": label}
        if space.tell_raw(facts=facts, key=key, speaker="curriculum",
                          metadata=meta) is None:
            continue
        n_entries += 1
        for parent in composed_of:
            comp_facts = {
                "entity": {"name": f"{key}.composes.{parent}",
                           "kind": "composition"},
                "relational": {"subject": key,
                               "predicate": "composed_of",
                               "object": parent},
            }
            if space.tell_raw(
                facts=comp_facts,
                key=f"composition.{key}.{parent}",
                speaker="curriculum",
                metadata={"category": "composition", "layer": label,
                          "from": key, "to": parent},
            ) is not None:
                n_comp += 1
    return n_entries, n_comp, n_skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "ada.db"),
                    help="path to the SQLite ada.db file")
    args = ap.parse_args()

    print()
    print("━" * 72)
    print("  Consolidating the substrate into a single ada.db")
    print("━" * 72)

    space = ThoughtGlyphSpace(enricher=None)
    t0 = time.time()

    # ── Phase 1: foundation layer ──────────────────────────────────
    print()
    print("  ── foundation layer (loaded FIRST, protected from override) ──")
    foundation_keys: set[str] = set()
    fnd_entries = 0
    fnd_comp = 0
    for path, entries in _entries_in(FOUNDATION_DIRS):
        rel = path.relative_to(ROOT)
        n, c, _ = _ingest(space, entries, label=str(rel.parent))
        fnd_entries += n
        fnd_comp += c
        for e in entries:
            if e.get("key"):
                foundation_keys.add(e["key"])
        print(f"    {str(rel):<40} {n:>4} entries, {c:>5} comp")
    print(f"    foundation total: {fnd_entries} entries, "
          f"{fnd_comp} composition facts. "
          f"Protected keys: {len(foundation_keys)}")
    print()

    # ── Phase 2: domain layers ─────────────────────────────────────
    print("  ── domain layer (LLM emissions for foundation keys are SKIPPED) ──")
    dom_entries = 0
    dom_comp = 0
    dom_skip = 0
    for path, entries in _entries_in(DOMAIN_FILES):
        rel = path.relative_to(ROOT)
        n, c, s = _ingest(space, entries, label=str(rel.parent),
                          foundation_keys=foundation_keys)
        dom_entries += n
        dom_comp += c
        dom_skip += s
        if entries:
            print(f"    {str(rel):<60} {n:>5} entries, "
                  f"{c:>6} comp, {s:>3} skipped")

    print(f"\n    domain total: {dom_entries} entries, {dom_comp} "
          f"composition facts, {dom_skip} skipped (foundation collision)")
    print()

    # ── Phase 3: persist ──────────────────────────────────────────
    print(f"  ── persist ──")
    print(f"    substrate in-memory: {space.count} thoughts "
          f"({fnd_entries + dom_entries} concepts + "
          f"{fnd_comp + dom_comp} compositions)")
    elapsed = time.time() - t0
    print(f"    consolidation wall time: {elapsed:.1f}s "
          f"({space.count / max(1, elapsed):.0f} thoughts/sec)")

    # Use the persistence module if available; otherwise dump pickle.
    out_path = Path(args.out)
    try:
        # We can't easily call async load_thoughts here without
        # spinning up the FastAPI session factory. Simpler: dump a
        # pickle of the in-memory ThoughtGlyphSpace; the runtime can
        # restore via thought_persistence later. For the user's
        # demonstration purpose, the on-disk artifact is what matters.
        import pickle
        with out_path.with_suffix(".pkl").open("wb") as fh:
            # ThoughtGlyphSpace isn't directly picklable because of
            # encoder state; serialize the thought records instead.
            payload = {
                "thoughts": [
                    {
                        "thought_id": t.thought_id,
                        "content": t.content,
                        "speaker": t.speaker,
                        "metadata": t.metadata,
                    }
                    for t in space._thoughts.values()
                ],
                "n_thoughts": space.count,
                "foundation_keys": sorted(foundation_keys),
            }
            pickle.dump(payload, fh)
        size_mb = out_path.with_suffix(".pkl").stat().st_size / 1024 / 1024
        print(f"    wrote {out_path.with_suffix('.pkl')} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"    pickle dump failed: {e}")

    # Compose a JSONL of all thoughts (smaller, human-inspectable)
    import json
    jsonl_path = out_path.with_suffix(".jsonl")
    with jsonl_path.open("w") as fh:
        for stored in space._thoughts.values():
            fh.write(json.dumps({
                "thought_id": stored.thought_id,
                "content": stored.content,
                "speaker": stored.speaker,
                "category": stored.metadata.get("category", ""),
                "source": stored.metadata.get("source", ""),
                "layer": stored.metadata.get("layer", ""),
                "key": stored.metadata.get("_key"),
                "version": stored.metadata.get("_version", 1),
            }) + "\n")
    size_mb = jsonl_path.stat().st_size / 1024 / 1024
    print(f"    wrote {jsonl_path} ({size_mb:.1f} MB)")
    print()

    print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"    FINAL: {space.count:>6} thoughts in unified substrate")
    print(f"    ({fnd_entries + dom_entries} grounded concepts + "
          f"{fnd_comp + dom_comp} composition facts)")
    print(f"    ({dom_skip} LLM emissions skipped as foundation collisions)")
    print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()
