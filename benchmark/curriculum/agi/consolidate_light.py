"""
Light consolidation: produce a unified ada.jsonl + ada.lattice.json of
all entries + composition facts across the substrate, WITHOUT the heavy
HDC encoding step. The runtime can rehydrate vectors at boot.

This produces the on-disk artifact in seconds instead of hours.

    PYTHONPATH=. python benchmark/curriculum/agi/consolidate_light.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).parent

FOUNDATION_DIRS = [
    ROOT / "layer-0",
    ROOT / "layer-1",
    ROOT / "infant",
    ROOT / "symbols",
    ROOT / "communication",
]
DOMAIN_FILES = [
    *(ROOT / "chemistry").glob("*.yaml"),
    ROOT / "chemistry" / "phase1_output" / "clean_entries.yaml",
    ROOT / "philosophy" / "phase1_output_v2" / "clean_entries.yaml",
    ROOT / "law" / "phase1_output_v2" / "clean_entries.yaml",
]


def _load(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with p.open() as fh:
        d = yaml.safe_load(fh)
        return d if isinstance(d, list) else []


def _files(paths):
    out = []
    for p in paths:
        if p.is_dir():
            for f in sorted(p.glob("*.yaml")):
                out.append(f)
        else:
            out.append(p)
    return out


def main() -> None:
    print()
    print("━" * 72)
    print("  Light consolidation — lattice + composition graph only")
    print("━" * 72)
    t0 = time.time()

    # ── Foundation: load and protect these keys ────────────────────
    foundation_keys: set[str] = set()
    foundation_entries: list[tuple[str, dict]] = []
    fnd_files = _files(FOUNDATION_DIRS)
    print(f"\n  ── foundation ({len(fnd_files)} files) ──")
    for f in fnd_files:
        entries = _load(f)
        rel = f.relative_to(ROOT)
        layer = str(rel.parent)
        for e in entries:
            k = e.get("key")
            if not k:
                continue
            foundation_keys.add(k)
            foundation_entries.append((layer, e))
    print(f"    {len(foundation_keys):>6} foundation keys loaded "
          f"(protected from override)")

    # ── Domain: load and skip collisions ───────────────────────────
    dom_files = _files(DOMAIN_FILES)
    print(f"\n  ── domain ({len(dom_files)} files) ──")
    domain_entries: list[tuple[str, dict]] = []
    by_domain = defaultdict(int)
    skipped = 0
    for f in dom_files:
        entries = _load(f)
        rel = f.relative_to(ROOT)
        # tag with domain prefix (the parent of the parent for phase1)
        if "phase1_output" in str(rel):
            layer = str(rel.parents[1])  # chemistry / philosophy / law
        else:
            layer = str(rel.parent)
        for e in entries:
            k = e.get("key")
            if not k:
                continue
            if k in foundation_keys:
                skipped += 1
                continue
            domain_entries.append((layer, e))
            by_domain[layer] += 1
        print(f"    {str(rel):<60} {len(entries):>6} entries")

    print(f"\n    domain totals by source:")
    for layer, n in sorted(by_domain.items(), key=lambda x: -x[1]):
        print(f"      {layer:<40} {n:>6}")
    print(f"    skipped (foundation collisions): {skipped}")

    # ── Build the consolidated lattice ─────────────────────────────
    all_entries = foundation_entries + domain_entries
    lattice: dict[str, dict] = {}  # key -> latest entry data
    versions: dict[str, list[dict]] = defaultdict(list)  # key -> [entries]
    composition_graph: list[tuple[str, str]] = []  # (from, to)
    for layer, e in all_entries:
        k = e.get("key")
        if not k:
            continue
        # Strip non-serializable bits, keep the schema-relevant fields
        record = {
            "key": k,
            "layer": layer,
            "category": e.get("category", "uncategorized"),
            "source": e.get("source", ""),
            "entity": e.get("entity"),
            "semantic": e.get("semantic"),
            "perceptual": e.get("perceptual"),
            "quantitative": e.get("quantitative"),
            "relational": e.get("relational"),
            "temporal": e.get("temporal"),
            "epistemic": e.get("epistemic"),
            "composed_of": e.get("composed_of") or [],
        }
        # Drop None-valued fields for compactness
        record = {k2: v for k2, v in record.items() if v is not None}
        lattice[k] = record  # overwrite policy: last wins (domain after foundation)
        versions[k].append(record)
        for parent in e.get("composed_of") or []:
            composition_graph.append((k, parent))

    # ── Write artifacts ────────────────────────────────────────────
    print()
    print(f"  ── writing artifacts ──")

    # 1. JSONL — one entry per line, easy to grep / stream
    jsonl_path = ROOT / "ada.jsonl"
    with jsonl_path.open("w") as fh:
        for k in sorted(lattice):
            fh.write(json.dumps(lattice[k]) + "\n")
    sz = jsonl_path.stat().st_size / 1024 / 1024
    print(f"    {jsonl_path.name:<24} {sz:>7.1f} MB  "
          f"({len(lattice)} entries)")

    # 2. lattice.json — keyed dict for direct lookup
    lattice_path = ROOT / "ada.lattice.json"
    with lattice_path.open("w") as fh:
        json.dump(lattice, fh)
    sz = lattice_path.stat().st_size / 1024 / 1024
    print(f"    {lattice_path.name:<24} {sz:>7.1f} MB  "
          f"({len(lattice)} keys)")

    # 3. composition.jsonl — edge list (from, to)
    edges_path = ROOT / "ada.composition.jsonl"
    with edges_path.open("w") as fh:
        for src, dst in composition_graph:
            fh.write(json.dumps({"from": src, "to": dst}) + "\n")
    sz = edges_path.stat().st_size / 1024 / 1024
    print(f"    {edges_path.name:<24} {sz:>7.1f} MB  "
          f"({len(composition_graph)} edges)")

    # 4. versions.jsonl — multi-version keys (cross-domain shared concepts)
    multi_versioned = {k: v for k, v in versions.items() if len(v) > 1}
    versions_path = ROOT / "ada.versions.jsonl"
    with versions_path.open("w") as fh:
        for k, vs in sorted(multi_versioned.items()):
            fh.write(json.dumps({"key": k, "versions": vs}) + "\n")
    sz = versions_path.stat().st_size / 1024 / 1024
    print(f"    {versions_path.name:<24} {sz:>7.1f} MB  "
          f"({len(multi_versioned)} multi-version keys)")

    # 5. summary
    summary = {
        "n_keys": len(lattice),
        "n_foundation": len(foundation_keys),
        "n_domain": sum(by_domain.values()),
        "n_composition_edges": len(composition_graph),
        "n_multi_version_keys": len(multi_versioned),
        "skipped_foundation_collisions": skipped,
        "by_domain": dict(by_domain),
        "build_seconds": round(time.time() - t0, 1),
    }
    summary_path = ROOT / "ada.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"    {summary_path.name:<24} {summary_path.stat().st_size:>7} B")

    print()
    print("━" * 72)
    print(f"  consolidated {len(lattice)} unique keys + "
          f"{len(composition_graph)} composition edges in "
          f"{time.time() - t0:.1f}s")
    print("━" * 72)


if __name__ == "__main__":
    main()
