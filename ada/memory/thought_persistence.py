"""
Thought persistence — saves and reconstructs Ada's memories from the DB.

ThoughtGlyphSpace stays in-memory for fast recall. This module syncs it to/from
the database (pgvector on Postgres, JSON text on SQLite):

  - Boot: load all non-archived thoughts and RECONSTRUCT their glyphs from
    stored vectors — no re-encoding. Encode once, reuse forever.
  - Absorb: write new thought (full hierarchy + global cortex) to the DB.
  - Reinforce/decay: update strength + last_accessed.
  - Archive: set archived=1, keep in DB for restoration.
  - Shutdown: flush pending strength updates.

The earlier version re-encoded each thought from its text on load, which both
discarded the stored vectors and was slow. This reconstructs the exact glyph.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: F401  (type hint reference)

from ada.core.types import Glyph, Layer, Segment, Vector
from ada.memory.thought_space import ThoughtGlyphSpace, StoredThought

logger = logging.getLogger(__name__)


# ── Vector <-> JSON helpers ──────────────────────────────────────────────────

def _vec_to_list(v: Any) -> list[int] | None:
    """Bipolar Vector / ndarray -> list[int]."""
    data = v.data if hasattr(v, "data") else v
    if isinstance(data, np.ndarray):
        return data.astype(int).tolist()
    if isinstance(data, list):
        return [int(x) for x in data]
    return None


def _list_to_vec(lst: list, space_id: str) -> Vector:
    arr = np.asarray(lst, dtype=np.int8)
    return Vector(data=arr, dimension=len(arr), space_id=space_id)


def _json_safe_meta(meta: dict) -> dict:
    """Drop / convert values that aren't JSON-serializable (Vectors, ndarrays)."""
    safe: dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if isinstance(v, Vector):
            continue  # e.g. _delta — recomputable, not worth persisting
        if isinstance(v, np.ndarray):
            safe[k] = v.astype(int).tolist()
        elif isinstance(v, (str, int, float, bool, type(None))):
            safe[k] = v
        elif isinstance(v, (list, dict)):
            try:
                json.dumps(v)
                safe[k] = v
            except (TypeError, ValueError):
                continue
    return safe


# ── Glyph (de)serialization — full hierarchy, no re-encode on load ───────────

def _serialize_glyph(glyph: Glyph) -> str:
    """Serialize a full Glyph (cortex + layers/segments/roles) to JSON."""
    layers: dict[str, Any] = {}
    for lname, layer in getattr(glyph, "layers", {}).items():
        segs: dict[str, Any] = {}
        for sname, seg in getattr(layer, "segments", {}).items():
            segs[sname] = {
                "cortex": _vec_to_list(seg.cortex),
                "roles": {r: _vec_to_list(rv) for r, rv in seg.roles.items()},
                "role_values": dict(getattr(seg, "role_values", {}) or {}),
                "weights": dict(getattr(seg, "weights", {}) or {}),
            }
        layers[lname] = {
            "cortex": _vec_to_list(layer.cortex),
            "segments": segs,
            "weights": dict(getattr(layer, "weights", {}) or {}),
        }
    data = {
        "identifier": glyph.identifier,
        "name": glyph.name,
        "space_id": glyph.space_id,
        "version": getattr(glyph, "version", "v1"),
        "global_cortex": _vec_to_list(glyph.global_cortex),
        "security_levels": dict(getattr(glyph, "security_levels", {}) or {}),
        "metadata": _json_safe_meta(getattr(glyph, "metadata", {}) or {}),
        "layers": layers,
    }
    return json.dumps(data)


def _deserialize_glyph(raw: str) -> Glyph:
    """Reconstruct a Glyph from JSON — no encoder, no re-encode."""
    data = json.loads(raw)
    space_id = data["space_id"]

    layers: dict[str, Layer] = {}
    for lname, ld in data.get("layers", {}).items():
        segments: dict[str, Segment] = {}
        for sname, sd in ld.get("segments", {}).items():
            segments[sname] = Segment(
                name=sname,
                cortex=_list_to_vec(sd["cortex"], space_id),
                roles={r: _list_to_vec(rv, space_id) for r, rv in sd.get("roles", {}).items()},
                role_values=sd.get("role_values", {}),
                weights=sd.get("weights", {}),
            )
        layers[lname] = Layer(
            name=lname,
            cortex=_list_to_vec(ld["cortex"], space_id),
            segments=segments,
            weights=ld.get("weights", {}),
        )

    meta = data.get("metadata", {}) or {}
    # Restore the content vector ndarray used by recall's content-cosine signal.
    if isinstance(meta.get("_content_vector"), list):
        meta["_content_vector"] = np.asarray(meta["_content_vector"], dtype=np.int8)

    identifier = data["identifier"]
    if "@" not in identifier or "#" not in identifier:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        identifier = f"{data['name']}@{stamp}#{data.get('version', 'v1')}"

    return Glyph(
        identifier=identifier,
        name=data["name"],
        space_id=space_id,
        global_cortex=_list_to_vec(data["global_cortex"], space_id),
        layers=layers,
        security_levels=data.get("security_levels", {}),
        metadata=meta,
        version=data.get("version", "v1"),
    )


# ── Persistence operations ───────────────────────────────────────────────────

async def count_thoughts(session_factory: Any) -> int:
    """Number of non-archived persisted thoughts (for store-once checks)."""
    from domains.models.db_models import AdaThought

    async with session_factory() as session:
        result = await session.execute(
            select(func.count()).select_from(AdaThought).where(AdaThought.archived == 0)
        )
        return int(result.scalar() or 0)


async def load_thoughts(session_factory: Any, space: ThoughtGlyphSpace) -> int:
    """Reconstruct all non-archived thoughts from the DB into `space`.

    Glyphs are rebuilt from their stored vectors — no re-encoding. Restores the
    in-memory store, the dedup set, and the versioned-concept history chains.
    Returns the count loaded.
    """
    from domains.models.db_models import AdaThought

    async with session_factory() as session:
        result = await session.execute(
            select(AdaThought).where(AdaThought.archived == 0)
        )
        rows = result.scalars().all()

    # Map existing in-memory seeds by content so a persisted thought replaces
    # its seed instead of duplicating it (the server loads after seeding).
    seed_by_content = {t.content.strip().lower(): tid for tid, t in space._thoughts.items()}

    count = 0
    by_key: dict[str, list[StoredThought]] = {}
    for row in rows:
        try:
            glyph = _deserialize_glyph(row.glyph_data)
        except Exception as exc:  # pragma: no cover — tolerate legacy/corrupt rows
            logger.warning("Skipping unreconstructable thought %s: %s", row.thought_id, exc)
            continue

        # Drop a matching seed so the persisted version wins (no duplicate).
        dup_id = seed_by_content.pop(row.content.strip().lower(), None)
        if dup_id is not None and dup_id != row.thought_id:
            space._thoughts.pop(dup_id, None)

        stored = StoredThought(
            thought_id=row.thought_id,
            content=row.content,
            speaker=row.speaker,
            glyph=glyph,
            strength=row.strength,
            created_at=row.created_at,
            last_accessed=row.last_accessed,
            access_count=row.access_count,
            metadata=row.extra_data or {},
        )
        space._thoughts[row.thought_id] = stored
        space._absorbed_texts.add(row.content.strip().lower())

        key = (row.extra_data or {}).get("_key")
        if key is not None:
            by_key.setdefault(key, []).append(stored)
        count += 1

    # Rebuild version chains in version order.
    for key, chain in by_key.items():
        chain.sort(key=lambda t: t.metadata.get("_version", 1))
        space._history_by_key[key] = chain

    if count:
        logger.info("Loaded (reconstructed) %d thoughts from database", count)
    return count


async def save_thought(session_factory: Any, thought: StoredThought) -> None:
    """Persist a thought (full hierarchy + global cortex vector)."""
    from domains.models.db_models import AdaThought

    cortex = _vec_to_list(thought.glyph.global_cortex)
    content_vec = thought.glyph.metadata.get("_content_vector")
    content_vec_json = (
        json.dumps(_vec_to_list(content_vec)) if content_vec is not None else None
    )

    async with session_factory() as session:
        row = AdaThought(
            thought_id=thought.thought_id,
            content=thought.content,
            speaker=thought.speaker,
            glyph_data=_serialize_glyph(thought.glyph),
            content_vector=content_vec_json,
            global_cortex=cortex,
            strength=thought.strength,
            access_count=thought.access_count,
            created_at=thought.created_at,
            last_accessed=thought.last_accessed,
            extra_data=_json_safe_meta(thought.metadata),
            archived=0,
        )
        await session.merge(row)  # upsert by primary key — idempotent re-saves
        await session.commit()


async def update_thought_strength(
    session_factory: Any,
    thought_id: str,
    strength: float,
    last_accessed: float,
    access_count: int,
) -> None:
    """Update strength/access for a thought."""
    from domains.models.db_models import AdaThought

    async with session_factory() as session:
        await session.execute(
            update(AdaThought)
            .where(AdaThought.thought_id == thought_id)
            .values(strength=strength, last_accessed=last_accessed, access_count=access_count)
        )
        await session.commit()


async def archive_thought(session_factory: Any, thought_id: str) -> None:
    """Mark a thought as archived."""
    from domains.models.db_models import AdaThought

    async with session_factory() as session:
        await session.execute(
            update(AdaThought).where(AdaThought.thought_id == thought_id).values(archived=1)
        )
        await session.commit()


async def flush_all_strengths(session_factory: Any, space: ThoughtGlyphSpace) -> int:
    """Batch-update all thought strengths. Called on shutdown."""
    from domains.models.db_models import AdaThought

    count = 0
    async with session_factory() as session:
        for thought in space._thoughts.values():
            await session.execute(
                update(AdaThought)
                .where(AdaThought.thought_id == thought.thought_id)
                .values(
                    strength=thought.strength,
                    last_accessed=thought.last_accessed,
                    access_count=thought.access_count,
                )
            )
            count += 1
        await session.commit()

    if count:
        logger.info("Flushed %d thought strengths to database", count)
    return count
