"""
Runtime connection manager — tracks local and remote Ada runtimes.

Persists connections to ~/.ada/runtimes.json so they survive restarts.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".ada"
RUNTIMES_FILE = CONFIG_DIR / "runtimes.json"


@dataclass
class RuntimeConfig:
    """Configurable runtime settings (hot-reloadable, no restart needed)."""
    rate_limit_per_minute: int = 60
    cors_allow_all: bool = False
    cors_origins: list[str] = field(default_factory=lambda: [
        "http://localhost:3000",
        "http://localhost:5173",
    ])
    enable_docs: bool = False
    max_vector_dimension: int = 2000
    log_level: str = "INFO"


@dataclass
class RuntimeConnection:
    """A named connection to an Ada runtime (local or remote)."""
    name: str
    url: Optional[str] = None       # None for local
    token: Optional[str] = None     # Bearer token for remote
    org_id: str = "default"
    is_local: bool = False
    status: str = "disconnected"    # connected, disconnected, error
    last_seen: Optional[float] = None
    config: RuntimeConfig = field(default_factory=RuntimeConfig)

    # Not persisted — set at runtime
    _error: str = ""

    def api_url(self, path: str) -> str:
        """Build a full API URL for this runtime."""
        base = self.url.rstrip("/") if self.url else "http://localhost:8002"
        return f"{base}/{path}"

    def headers(self) -> dict[str, str]:
        """Build auth headers."""
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON persistence."""
        d = {
            "name": self.name,
            "url": self.url,
            "token": self.token,
            "org_id": self.org_id,
            "is_local": self.is_local,
            "config": asdict(self.config),
        }
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeConnection":
        """Deserialize from JSON."""
        config_data = data.pop("config", {})
        config = RuntimeConfig(**config_data) if config_data else RuntimeConfig()
        return cls(config=config, **data)


class RuntimeManager:
    """Manages named runtime connections with persistence."""

    def __init__(self):
        self._lock = threading.Lock()
        self._runtimes: dict[str, RuntimeConnection] = {}
        self._active: Optional[str] = None  # currently selected runtime name
        self._load()

    def _load(self) -> None:
        """Load runtimes from disk."""
        if RUNTIMES_FILE.exists():
            try:
                data = json.loads(RUNTIMES_FILE.read_text())
                for entry in data.get("runtimes", []):
                    rt = RuntimeConnection.from_dict(entry)
                    self._runtimes[rt.name] = rt
                self._active = data.get("active")
            except (json.JSONDecodeError, OSError, TypeError) as e:
                logger.warning(f"Could not load runtimes config: {e}")

        # Ensure "local" always exists
        if "local" not in self._runtimes:
            self._runtimes["local"] = RuntimeConnection(
                name="local",
                is_local=True,
                org_id="default",
            )

        if not self._active or self._active not in self._runtimes:
            self._active = "local"

    def _save(self) -> None:
        """Persist runtimes to disk."""
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                "runtimes": [rt.to_dict() for rt in self._runtimes.values()],
                "active": self._active,
            }
            RUNTIMES_FILE.write_text(json.dumps(data, indent=2))
        except OSError as e:
            logger.warning(f"Could not save runtimes config: {e}")

    def add(self, name: str, url: str, token: Optional[str] = None, org_id: str = "default") -> RuntimeConnection:
        """Add a remote runtime connection."""
        with self._lock:
            rt = RuntimeConnection(
                name=name,
                url=url.rstrip("/"),
                token=token,
                org_id=org_id,
                is_local=False,
            )
            self._runtimes[name] = rt
            self._save()
            return rt

    def remove(self, name: str) -> bool:
        """Remove a runtime connection. Cannot remove 'local'."""
        if name == "local":
            return False
        with self._lock:
            if name in self._runtimes:
                del self._runtimes[name]
                if self._active == name:
                    self._active = "local"
                self._save()
                return True
            return False

    def get(self, name: str) -> Optional[RuntimeConnection]:
        """Get a runtime by name."""
        return self._runtimes.get(name)

    def get_active(self) -> RuntimeConnection:
        """Get the currently active runtime."""
        return self._runtimes.get(self._active, self._runtimes["local"])

    def set_active(self, name: str) -> bool:
        """Set the active runtime."""
        if name in self._runtimes:
            with self._lock:
                self._active = name
                self._save()
            return True
        return False

    def list_all(self) -> list[RuntimeConnection]:
        """List all runtimes."""
        return list(self._runtimes.values())

    def update_status(self, name: str, status: str, error: str = "") -> None:
        """Update a runtime's connection status."""
        rt = self._runtimes.get(name)
        if rt:
            rt.status = status
            rt._error = error
            if status == "connected":
                rt.last_seen = time.time()

    def update_config(self, name: str, **kwargs) -> bool:
        """Update runtime config values."""
        rt = self._runtimes.get(name)
        if not rt:
            return False
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(rt.config, key):
                    setattr(rt.config, key, value)
            self._save()
        return True

    def get_config(self, name: str) -> Optional[RuntimeConfig]:
        """Get runtime config."""
        rt = self._runtimes.get(name)
        return rt.config if rt else None


# Module-level singleton
_manager: Optional[RuntimeManager] = None


def get_runtime_manager() -> RuntimeManager:
    """Get the global runtime manager."""
    global _manager
    if _manager is None:
        _manager = RuntimeManager()
    return _manager
