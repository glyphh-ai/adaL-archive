"""
Load the communication curriculum into a substrate already preloaded
with layer-0, layer-1, infant, and symbols.

Stack:
   prime.* + binary.* + infant.* + layer-1 + symbols.* (digits,
   letters, phonemes)
     → phonotactics.* (phoneme-sequence rules)
       → syllables.* (syllable templates + stress)
         → lexicon.* (first words: sound + meaning anchored)
           → grammar.* + act.* + dialogue.* (rules of communication)

NO LLM at ingest.

    PYTHONPATH=. python benchmark/curriculum/agi/communication/teach.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from ada.memory.thought_space import ThoughtGlyphSpace

ROOT = Path(__file__).parent.parent
LAYER_0 = ROOT / "layer-0" / "primes.yaml"
LAYER_1 = [ROOT / "layer-1" / f for f in
           ("colors.yaml", "numbers.yaml", "spatial.yaml",
            "logical.yaml", "temporal.yaml")]
INFANT_FILES = sorted((ROOT / "infant").glob("stage-*.yaml"),
                      key=lambda p: int(p.stem.split("_")[0].split("-")[1]))
SYMBOL_FILES = [ROOT / "symbols" / f for f in
                ("digits.yaml", "letters.yaml", "phonemes.yaml")]
COMM_FILES = [Path(__file__).parent / f for f in
              ("phonotactics.yaml", "syllables.yaml", "lexicon.yaml",
               "grammar.yaml")]


def _load_yaml(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return yaml.safe_load(fh) or []


def _ingest(space, entries, layer_label):
    n_entries = n_comp = 0
    for e in entries:
        key = e.get("key")
        if not key:
            continue
        composed_of = e.get("composed_of") or []
        facts = {k: v for k, v in e.items()
                 if k not in ("key", "category", "source", "composed_of")
                 and isinstance(v, dict)}
        meta = {"category": e.get("category", "uncategorized"),
                "source": e.get("source", ""), "layer": layer_label}
        if space.tell_raw(facts=facts, key=key, speaker="curriculum",
                          metadata=meta) is None:
            continue
        n_entries += 1
        for parent in composed_of:
            comp_facts = {
                "entity": {"name": f"{key}.composes.{parent}",
                           "kind": "composition"},
                "relational": {"subject": key, "predicate": "composed_of",
                               "object": parent},
            }
            if space.tell_raw(
                facts=comp_facts, key=f"composition.{key}.{parent}",
                speaker="curriculum",
                metadata={"category": "composition", "layer": layer_label,
                          "from": key, "to": parent},
            ) is not None:
                n_comp += 1
    return n_entries, n_comp


def main():
    space = ThoughtGlyphSpace(enricher=None)

    print("  ── layer-0 NSM primes ──")
    n0, _ = _ingest(space, _load_yaml(LAYER_0), "layer-0")
    print(f"    {n0} primes\n")

    print("  ── layer-1 cognitive primitives ──")
    n1 = c1 = 0
    for p in LAYER_1:
        n, c = _ingest(space, _load_yaml(p), "layer-1")
        n1 += n; c1 += c
    print(f"    {n1} entries, {c1} composition facts\n")

    print("  ── infant developmental ──")
    ni = ci = 0
    for p in INFANT_FILES:
        n, c = _ingest(space, _load_yaml(p), "infant")
        ni += n; ci += c
    print(f"    {ni} entries, {ci} composition facts\n")

    print("  ── symbols (digits, letters, phonemes) ──")
    ns = cs = 0
    for p in SYMBOL_FILES:
        n, c = _ingest(space, _load_yaml(p), "symbols")
        ns += n; cs += c
    print(f"    {ns} entries, {cs} composition facts\n")

    print("  ── communication ──")
    nc = cc = 0
    for p in COMM_FILES:
        entries = _load_yaml(p)
        n, c = _ingest(space, entries, "communication")
        nc += n; cc += c
        print(f"    {p.name:<20} {n:>3} entries, {c:>3} comp facts")
    print(f"    communication totals: {nc} entries, {cc} composition facts\n")

    print(f"  substrate now holds {space.count} thoughts "
          f"({n0 + n1 + ni + ns + nc} concepts + "
          f"{c1 + ci + cs + cc} composition facts).\n")
    return space


if __name__ == "__main__":
    main()
