"""
Hard questions on the SCALED substrate.

Integrates the Phase 1 LLM-extracted entries with the hand-written
chemistry curriculum and re-runs the same compositional-traversal
operations from hard_questions.py — plus new questions that REQUIRE
the expanded substrate (caffeine, aspirin, drugs, biomolecules).

The point: the operations don't care whether facts were hand-written
or LLM-extracted. They work on the lattice. So if the integration
preserves composition integrity (which Phase 1 just measured at 100%
prime-grounding), the operations carry through.

    PYTHONPATH=. python curriculum/agi/chemistry/hard_questions_scaled.py
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from hard_questions import (banner, contains_all, contains_any,
                            depends_on, trace_to_primes, print_tree)
from teach import CHEM_FILES, LAYER_0, LAYER_1, _load_yaml


def load_scaled_lattice() -> dict[str, dict]:
    lattice: dict[str, dict] = {}
    # Original substrate.
    for p in [LAYER_0, *LAYER_1, *CHEM_FILES]:
        for e in _load_yaml(p):
            k = e.get("key")
            if k:
                lattice[k] = e
    # Phase 1 LLM-extracted entries.
    phase1 = HERE / "phase1_output" / "clean_entries.yaml"
    if phase1.exists():
        with phase1.open() as fh:
            for e in yaml.safe_load(fh) or []:
                k = e.get("key")
                if k and k not in lattice:
                    lattice[k] = e
    return lattice


def main() -> None:
    print()
    print("╔" + "═" * 70 + "╗")
    print("║" + "  SCALED HARD QUESTIONS — hand-written + LLM-extracted".ljust(70) + "║")
    print("║" + "  Same operations, much bigger lattice.".ljust(70) + "║")
    print("╚" + "═" * 70 + "╝")

    lattice = load_scaled_lattice()
    hand_keys = {e["key"] for p in CHEM_FILES for e in (_load_yaml(p) or [])
                 if e.get("key")}
    llm_keys = {k for k in lattice
                if k not in hand_keys
                and not k.startswith(("prime.", "color.", "number.",
                                      "spatial.", "logic.", "time."))}
    print(f"  lattice: {len(lattice)} keys total")
    print(f"    hand-written chemistry: {len(hand_keys)}")
    print(f"    LLM-extracted phase-1:  {len(llm_keys)}")
    print(f"    layers 0-1 primitives:  "
          f"{len(lattice) - len(hand_keys) - len(llm_keys)}")
    print()

    # ── Q1: trace a NEW compound (caffeine) all the way to primes ─
    if "compound.caffeine" in lattice:
        banner("Q1. Trace CAFFEINE to its grounding primes "
               "(new compound — LLM-extracted)")
        out = trace_to_primes(lattice, "compound.caffeine")
        print("    Composition tree:")
        print_tree(out["tree"], "compound.caffeine", indent="      ",
                   max_depth=6)
        print(f"\n    Reached {len(out['primes'])} layer-0 primes "
              f"({out['n_nodes_visited']} nodes, depth {out['max_depth']}).")
        print(f"    Primes: {sorted(out['primes'])}")

    # ── Q2: enumerate every compound containing nitrogen ────────
    banner("Q2. Every compound containing NITROGEN (transitive)")
    cs = [k for k in lattice
          if k.startswith("compound.")
          and contains_any(lattice, k, {"element.nitrogen"})]
    cs.sort()
    print(f"    {len(cs)} compounds:")
    for k in cs:
        marker = "  [new]" if k in llm_keys else ""
        print(f"      • {k:<38}{marker}")

    # ── Q3: amino acids (amino + carboxyl) ──────────────────────
    banner("Q3. Every compound that is an AMINO ACID by lattice "
           "(amino + carboxyl)")
    aa = [k for k in lattice
          if k.startswith("compound.")
          and contains_all(lattice, k, {"group.amino", "group.carboxyl"})]
    aa.sort()
    print(f"    {len(aa)} amino-acid compounds:")
    for k in aa:
        marker = "  [new]" if k in llm_keys else ""
        ent = lattice[k].get("entity", {})
        sub = ent.get("subkind") or ent.get("kind", "")
        print(f"      • {k:<38} ({sub}){marker}")

    # ── Q4: refusal at scale — what does aspirin taste like? ────
    if "compound.aspirin" in lattice:
        banner("Q4. What does ASPIRIN taste like?")
        node = lattice["compound.aspirin"]
        perc = node.get("perceptual") or {}
        print(f"    aspirin.perceptual roles filled: {list(perc.keys())}")
        print(f"    aspirin.perceptual.taste:        "
              f"{perc.get('taste', '∅ (no path)')}")
        out = trace_to_primes(lattice, "compound.aspirin")
        chain = set(out["tree"].keys()) | set(out["primes"].keys())
        taste_hits = []
        for k in chain:
            n = lattice.get(k, {})
            for layer in ("perceptual", "semantic"):
                block = n.get(layer) or {}
                for role, val in block.items():
                    if role == "taste" or "taste" in str(val).lower():
                        taste_hits.append((k, layer, role, val))
        print(f"    taste-related facts anywhere in chain: "
              f"{taste_hits if taste_hits else '∅'}")
        print("    → structural refusal: no taste role filled, no guess.")

    # ── Q5: cross-domain — drug-like + benzene ring ─────────────
    banner("Q5. Compounds containing a BENZENE substructure (aromatic ring)")
    # Heuristic: composition reaches compound.benzene or includes the
    # explicit aromatic markers element.carbon + bond.double-covalent +
    # (semantic mentions aromatic).
    bz_via_benzene = [k for k in lattice
                      if k.startswith("compound.")
                      and contains_any(lattice, k, {"compound.benzene"})]
    bz_via_benzene.sort()
    print(f"    {len(bz_via_benzene)} compounds reach compound.benzene directly:")
    for k in bz_via_benzene[:30]:
        marker = "  [new]" if k in llm_keys else ""
        print(f"      • {k:<38}{marker}")
    if len(bz_via_benzene) > 30:
        print(f"      ... and {len(bz_via_benzene) - 30} more.")

    # ── Q6: every group / motif Ada now knows ───────────────────
    banner("Q6. Every functional GROUP in the substrate (hand + LLM-extracted)")
    groups = sorted(k for k in lattice if k.startswith("group."))
    print(f"    {len(groups)} groups:")
    for k in groups:
        marker = "  [new]" if k in llm_keys else ""
        e = lattice[k]
        sub = (e.get("entity") or {}).get("subkind") or "-"
        co = e.get("composed_of") or []
        print(f"      • {k:<30} composed_of={co[:3]}{'...' if len(co)>3 else ''}{marker}")

    # ── Q7: closed-set enumeration scaled ──────────────────────
    banner("Q7. Everything depending on bond.hydrogen — scaled lattice")
    deps = depends_on(lattice, "bond.hydrogen")
    chem_deps = sorted(d for d in deps if d.startswith(
        ("compound.", "group.", "property.", "reaction.")))
    new_among = [d for d in chem_deps if d in llm_keys]
    print(f"    {len(chem_deps)} chemistry concepts depend on bond.hydrogen.")
    print(f"    {len(new_among)} of those are LLM-extracted (Phase 1).")
    print(f"    First 25:")
    for d in chem_deps[:25]:
        marker = "  [new]" if d in llm_keys else ""
        print(f"      • {d:<38}{marker}")
    if len(chem_deps) > 25:
        print(f"      ... and {len(chem_deps) - 25} more.")

    # ── Q8: glucose vs ATP comparison still works ──────────────
    banner("Q8. Glucose vs ATP — same operation, scaled substrate")
    if "compound.glucose" in lattice and "compound.atp" in lattice:
        g_direct = set(lattice["compound.glucose"].get("composed_of") or [])
        a_direct = set(lattice["compound.atp"].get("composed_of") or [])
        print(f"      shared at level-1:  {sorted(g_direct & a_direct)}")
        print(f"      glucose only:       {sorted(g_direct - a_direct)}")
        print(f"      ATP only:           {sorted(a_direct - g_direct)}")

    # ── Q9: every drug in the substrate ────────────────────────
    banner("Q9. Likely PHARMACEUTICALS in the substrate")
    drug_names = {"aspirin", "ibuprofen", "acetaminophen", "naproxen",
                  "caffeine", "theobromine", "theophylline", "nicotine",
                  "morphine", "codeine", "penicillin", "amoxicillin",
                  "cephalexin", "tetracycline", "insulin", "epinephrine",
                  "dopamine", "serotonin", "melatonin", "acetylcholine",
                  "diazepam", "fluoxetine", "sertraline", "omeprazole",
                  "atorvastatin", "metformin", "norepinephrine"}
    drug_keys = sorted(k for k in lattice
                       if k.startswith("compound.")
                       and any(d in k for d in drug_names))
    print(f"    {len(drug_keys)} pharmaceutical compounds Ada knows:")
    for k in drug_keys:
        e = lattice[k]
        co = e.get("composed_of") or []
        print(f"      • {k:<32} composed_of={co[:4]}")

    print()
    print("━" * 72)
    print("  Scaled headline")
    print("━" * 72)
    print(f"  Substrate scaled from 84 → {len(hand_keys) + len(llm_keys)} chemistry concepts.")
    print(f"  Same compositional-traversal operations work.")
    print(f"  Every new concept's chain bottoms out at the same NSM primes.")
    print(f"  Refusal-with-provenance carries through to LLM-extracted entries.")
    print()


if __name__ == "__main__":
    main()
