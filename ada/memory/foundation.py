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

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from ada.memory.thought_space import ThoughtGlyphSpace

logger = logging.getLogger(__name__)

FOUNDATION_DIR = Path(__file__).parent / "foundation"

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


def seed_foundation(
    space: "ThoughtGlyphSpace",
    directory: str | Path | None = None,
) -> dict[str, int]:
    """Seed the foundation primitives + composition facts into `space`.

    LLM-free (uses `tell_raw`). Idempotent: `tell_raw` dedups by key/text, so
    seeding an already-warm space is a no-op. Returns counts for logging.
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
        "Foundation seeded: %d primitives + %d composition facts",
        n_concepts,
        n_composition,
    )
    return {"concepts": n_concepts, "composition": n_composition}
