"""
Foundation primitives — Ada's innate conceptual bedrock, seeded at boot.

These are NOT curriculum. They are core: the layer-0 Natural Semantic
Metalanguage primes (Wierzbicka) and the layer-1 cognitive primitives
(colors, numbers, spatial, temporal, logical) that every higher concept is
composed from. They live in the runtime package (`ada/memory/foundation/`)
and are seeded into the substrate when the runtime cognitive core boots, so a
fresh process starts WARM — no curriculum run required before data flows in.

Each entry is a universal-schema glyph ingested via `tell_raw` (NO LLM calls).
`composed_of` references become relational composition facts so the
foundation lattice is queryable via standard recall. This is the same
ingestion the old `curriculum/agi/layer-1/teach.py` did — moved into core.
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from ada.memory.thought_space import ThoughtGlyphSpace

logger = logging.getLogger(__name__)

FOUNDATION_DIR = Path(__file__).parent / "foundation"
FOUNDATION_SNAPSHOT = FOUNDATION_DIR / "snapshot.pkl.gz"

# Bump when the encoder/glyph format changes in a way that invalidates a
# previously-built snapshot (the YAML hash alone won't catch code changes).
SNAPSHOT_VERSION = 1

# Load order matters: primes first, so layer-1 `composed_of` references
# resolve into keys that already exist in the substrate.
FOUNDATION_FILES: tuple[str, ...] = (
    "primes.yaml",     # layer-0: NSM primes (atomic)
    "colors.yaml",     # layer-1: cognitive primitives
    "numbers.yaml",
    "spatial.yaml",
    "logical.yaml",
    "temporal.yaml",
)


def load_foundation_entries(
    directory: str | Path | None = None,
) -> list[tuple[str, dict]]:
    """Read the foundation YAML files in load order.

    Returns a list of (layer_label, entry) tuples, primes ("layer-0") first.
    """
    base = Path(directory) if directory else FOUNDATION_DIR
    out: list[tuple[str, dict]] = []
    for name in FOUNDATION_FILES:
        path = base / name
        if not path.exists():
            logger.warning("Foundation file missing: %s", path)
            continue
        with path.open() as fh:
            entries = yaml.safe_load(fh) or []
        label = "layer-0" if name == "primes.yaml" else "layer-1"
        for entry in entries:
            out.append((label, entry))
    return out


def _source_hash(directory: str | Path | None = None) -> str:
    """Content hash of the foundation YAML + snapshot version.

    Invalidates a stale snapshot when the source primitives change.
    """
    h = hashlib.sha256()
    h.update(f"v{SNAPSHOT_VERSION}".encode())
    base = Path(directory) if directory else FOUNDATION_DIR
    for name in FOUNDATION_FILES:
        path = base / name
        if path.exists():
            h.update(path.read_bytes())
    return h.hexdigest()


def seed_foundation(
    space: "ThoughtGlyphSpace",
    directory: str | Path | None = None,
    use_snapshot: bool = True,
) -> dict[str, int]:
    """Seed the foundation primitives into `space`, warm and fast.

    Fast path: load a pre-encoded snapshot (~ms) shipped with the package.
    Fallback: encode live via `tell_raw` (~seconds) if the snapshot is
    missing or stale. Idempotent either way.
    """
    if use_snapshot and load_snapshot(space, directory):
        return {
            "concepts": sum(
                1 for t in space._thoughts.values()
                if t.metadata.get("foundation") and t.metadata.get("category") != "composition"
            ),
            "composition": sum(
                1 for t in space._thoughts.values()
                if t.metadata.get("category") == "composition"
            ),
        }
    return _seed_live(space, directory)


def _seed_live(
    space: "ThoughtGlyphSpace",
    directory: str | Path | None = None,
) -> dict[str, int]:
    """Encode the foundation primitives + composition facts via tell_raw.

    LLM-free. Idempotent: `tell_raw` dedups by key/text. Returns counts.
    """
    n_concepts = 0
    n_composition = 0

    for layer_label, entry in load_foundation_entries(directory):
        key = entry.get("key")
        if not key:
            continue

        # Universal-schema slots are the dict-valued fields (entity, semantic,
        # perceptual, ...). Bookkeeping fields are skipped.
        facts = {
            k: v
            for k, v in entry.items()
            if k not in ("key", "category", "source", "composed_of")
            and isinstance(v, dict)
        }
        meta: dict[str, Any] = {
            "category": entry.get("category", "uncategorized"),
            "source": entry.get("source", ""),
            "layer": layer_label,
            "foundation": True,
        }
        stored = space.tell_raw(
            facts=facts, key=key, speaker="foundation", metadata=meta
        )
        if stored is None:
            continue
        n_concepts += 1

        # Composition: one relational fact per referenced parent key.
        for parent in entry.get("composed_of") or []:
            comp_facts = {
                "entity": {"name": f"{key}.composes.{parent}", "kind": "composition"},
                "relational": {
                    "subject": key,
                    "predicate": "composed_of",
                    "object": parent,
                },
            }
            comp_stored = space.tell_raw(
                facts=comp_facts,
                key=f"composition.{key}.{parent}",
                speaker="foundation",
                metadata={
                    "category": "composition",
                    "layer": layer_label,
                    "foundation": True,
                    "from": key,
                    "to": parent,
                },
            )
            if comp_stored is not None:
                n_composition += 1

    logger.info(
        "Foundation seeded (live): %d primitives + %d composition facts",
        n_concepts,
        n_composition,
    )
    return {"concepts": n_concepts, "composition": n_composition}


# ── Snapshot (pre-encoded warm start) ────────────────────────────────────────

def load_snapshot(
    space: "ThoughtGlyphSpace",
    directory: str | Path | None = None,
    path: str | Path | None = None,
) -> bool:
    """Load a pre-encoded foundation snapshot into `space`.

    Returns True on a successful, hash-matched load; False if the snapshot is
    missing, stale, or unreadable (caller falls back to live encoding).
    """
    snap_path = Path(path) if path else FOUNDATION_SNAPSHOT
    if not snap_path.exists():
        return False
    try:
        with gzip.open(snap_path, "rb") as fh:
            blob = pickle.load(fh)
    except Exception as exc:  # pragma: no cover — corrupt/incompatible snapshot
        logger.warning("Foundation snapshot unreadable (%s); encoding live.", exc)
        return False

    if blob.get("hash") != _source_hash(directory):
        logger.info("Foundation snapshot stale; encoding live.")
        return False

    for stored in blob["thoughts"]:
        space._thoughts[stored.thought_id] = stored
        space._absorbed_texts.add(stored.content.strip().lower())
        key = stored.metadata.get("_key")
        if key is not None:
            space._history_by_key.setdefault(key, []).append(stored)

    logger.info("Foundation loaded from snapshot: %d thoughts.", len(blob["thoughts"]))
    return True


def build_snapshot(
    directory: str | Path | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Encode the foundation live and write a snapshot artifact.

    Build-time helper (run via `python -m ada.memory.foundation`). Commit the
    resulting snapshot so fresh, ephemeral runtimes boot warm in ~ms.
    """
    from ada.memory.thought_space import ThoughtGlyphSpace

    space = ThoughtGlyphSpace(enricher=None)  # with_foundation=False
    counts = _seed_live(space, directory)

    snap_path = Path(path) if path else FOUNDATION_SNAPSHOT
    blob = {
        "hash": _source_hash(directory),
        "version": SNAPSHOT_VERSION,
        "thoughts": list(space._thoughts.values()),
    }
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(snap_path, "wb", compresslevel=6) as fh:
        pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)

    size_kb = snap_path.stat().st_size / 1024
    logger.info("Foundation snapshot written: %s (%.0f KB)", snap_path, size_kb)
    return {**counts, "thoughts": len(blob["thoughts"]), "size_kb": round(size_kb)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = build_snapshot()
    print(
        f"Built foundation snapshot: {result['thoughts']} thoughts "
        f"({result['concepts']} primitives + {result['composition']} composition), "
        f"{result['size_kb']} KB"
    )
