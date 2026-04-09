"""
Capability loader — boots Ada's innate capabilities at startup.

Scans capabilities/ directory, loads each capability's encoder,
exemplars, and vector space. No org_id, no model_id — just capability
names.

Internally uses the existing SDK model loader (domains.models.loader)
and ModelManager for DB persistence, but wraps them in a brain-native
interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from domains.models.loader import (
    LoadedModel as LoaderModel,
    ModelManifest,
    discover_models,
    load_model,
)

logger = logging.getLogger(__name__)

# Internal constant used for DB compat — the multi-tenant layer
# requires an org_id, so we hardcode one for Ada's brain.
ADA_ORG_ID = "ada"


@dataclass
class Capability:
    """A loaded brain capability."""
    name: str
    manifest: ModelManifest
    model_dir: Path
    encoder_config: Any = None
    encode_query_fn: Any = None
    entry_to_record_fn: Any = None
    assess_query_fn: Any = None
    mcp_tools: list = field(default_factory=list)
    handle_mcp_tool_fn: Any = None
    exemplar_count: int = 0
    loaded_at: datetime = field(default_factory=datetime.utcnow)
    is_router: bool = False  # True only for toolrouter


@dataclass
class BrainState:
    """Snapshot of all loaded capabilities."""
    capabilities: Dict[str, Capability] = field(default_factory=dict)
    router: Optional[Capability] = None

    @property
    def capability_names(self) -> list[str]:
        return list(self.capabilities.keys())

    def get(self, name: str) -> Optional[Capability]:
        return self.capabilities.get(name)


class CapabilityLoader:
    """Discovers and loads capabilities from the filesystem."""

    ROUTER_NAMES = {"toolrouter", "cognitive-router"}

    @classmethod
    def boot(cls, capabilities_dir: Path) -> BrainState:
        """Load all capabilities from the given directory.

        Returns a BrainState with all capabilities indexed by name.
        The toolrouter capability is separated as the router.
        """
        if not capabilities_dir.exists():
            logger.warning(f"Capabilities directory not found: {capabilities_dir}")
            return BrainState()

        state = BrainState()
        loader_models = discover_models(capabilities_dir)

        for lm in loader_models:
            cap = cls._to_capability(lm)
            if cap.name in cls.ROUTER_NAMES:
                cap.is_router = True
                state.router = cap
                logger.info(f"  [router] {cap.name}")
            else:
                state.capabilities[cap.name] = cap
                logger.info(f"  [capability] {cap.name}")

        total = len(state.capabilities) + (1 if state.router else 0)
        logger.info(
            f"Brain loaded: {total} capabilities "
            f"({len(state.capabilities)} routable, "
            f"{'1 router' if state.router else 'NO ROUTER'})"
        )

        return state

    @classmethod
    def _to_capability(cls, lm: LoaderModel) -> Capability:
        return Capability(
            name=lm.model_id,
            manifest=lm.manifest,
            model_dir=lm.model_dir,
            encoder_config=lm.encoder_config,
            encode_query_fn=lm.encode_query_fn,
            entry_to_record_fn=lm.entry_to_record_fn,
            assess_query_fn=lm.assess_query_fn,
            mcp_tools=lm.mcp_tools,
            handle_mcp_tool_fn=lm.handle_mcp_tool_fn,
            exemplar_count=lm.exemplar_count,
        )


async def register_capabilities_in_db(
    state: BrainState,
    model_manager: Any,
) -> None:
    """Register all capabilities with the ModelManager for DB persistence.

    This loads exemplars into the database and creates runtime encoders.
    Uses ADA_ORG_ID as the org_id for all capabilities.
    """
    all_caps = list(state.capabilities.values())
    if state.router:
        all_caps.append(state.router)

    for cap in all_caps:
        try:
            await model_manager.load_model_from_directory(
                model_dir=cap.model_dir,
                org_id=ADA_ORG_ID,
                model_id=cap.name,
            )
            logger.info(f"Registered {cap.name} in DB")
        except Exception as e:
            logger.error(f"Failed to register {cap.name}: {e}")
