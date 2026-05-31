"""
Verify the communication curriculum.

    PYTHONPATH=. python curriculum/agi/communication/verify.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from teach import (COMM_FILES, INFANT_FILES, LAYER_0, LAYER_1,
                   SYMBOL_FILES, _load_yaml, main as teach_main)


def main():
    space = teach_main()

    all_yaml_files = [LAYER_0, *LAYER_1, *INFANT_FILES, *SYMBOL_FILES,
                      *COMM_FILES]
    all_entries = []
    for p in all_yaml_files:
        all_entries.extend(_load_yaml(p))
    all_keys = {e.get("key") for e in all_entries if e.get("key")}

    comm_entries = []
    for p in COMM_FILES:
        comm_entries.extend(_load_yaml(p))
    comm_keys = {e.get("key") for e in comm_entries if e.get("key")}

    missing = [k for k in comm_keys if k not in space._history_by_key]
    print(f"  ── coverage ──")
    print(f"    communication: {len(comm_keys) - len(missing)}/"
          f"{len(comm_keys)} present"
          f"{' ✓' if not missing else f'  ✗ missing: {missing[:5]}'}")
    print()

    dangling = []
    n_compositions = 0
    for e in comm_entries:
        key = e.get("key")
        for parent in e.get("composed_of") or []:
            n_compositions += 1
            if parent not in all_keys:
                dangling.append((key, parent))
    print(f"  ── composition integrity ──")
    print(f"    {n_compositions} communication composition links declared.")
    if dangling:
        print(f"    ✗ {len(dangling)} dangling references:")
        for from_k, to_k in dangling[:20]:
            print(f"        {from_k}  → {to_k}")
    else:
        print(f"    ✓ every composition target exists.")
    print()

    # Sample compositions illustrating semantic clustering
    print(f"  ── sample lexicon compositions ──")
    by_comm = {e["key"]: e.get("composed_of") or [] for e in comm_entries}
    for k in ["lexicon.mama", "lexicon.milk", "lexicon.warm", "lexicon.hug",
              "lexicon.rock", "lexicon.gone", "lexicon.cold",
              "grammar.S-declarative", "act.greeting"]:
        if k in by_comm:
            print(f"    {k:<28} ← {by_comm[k][:6]}{'...' if len(by_comm[k])>6 else ''}")

    # Distance proof: count shared keys between word pairs
    print()
    print(f"  ── semantic distance proof ──")
    pairs = [
        ("lexicon.mama", "lexicon.milk"),
        ("lexicon.mama", "lexicon.warm"),
        ("lexicon.mama", "lexicon.hug"),
        ("lexicon.mama", "lexicon.rock"),
        ("lexicon.mama", "lexicon.gone"),
        ("lexicon.mama", "lexicon.cold"),
        ("lexicon.warm", "lexicon.cold"),
        ("lexicon.warm", "lexicon.hug"),
    ]
    for a, b in pairs:
        sa = set(by_comm.get(a, []))
        sb = set(by_comm.get(b, []))
        shared = sa & sb
        total = sa | sb
        # Phoneme keys are sound-overlap, not semantic — separate them
        phoneme_shared = {k for k in shared if k.startswith("phoneme.")}
        sem_shared = shared - phoneme_shared
        print(f"    {a:<16} ↔ {b:<16}  semantic-shared={len(sem_shared):>2}  "
              f"phoneme-shared={len(phoneme_shared)}")


if __name__ == "__main__":
    main()
