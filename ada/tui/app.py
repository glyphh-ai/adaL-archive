"""Ada Runtime — terminal dashboard with multi-runtime support."""

from __future__ import annotations

import asyncio
import logging
import time
import threading
from collections import deque
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Static, Input, Footer, Header, TabbedContent, TabPane
from textual.binding import Binding
from textual.timer import Timer
from rich.text import Text
from rich.table import Table
from rich.style import Style
from rich.panel import Panel

from ada.tui.banner import get_banner
from ada.tui.graph import render_graph
from ada.tui.stats import get_stats
from ada.tui.runtimes import get_runtime_manager, RuntimeConnection

# Ada's color palette
ADA_BLUE = "#6b7bff"
ADA_PINK = "#ff6b6b"
ADA_PURPLE = "#9b6bea"
TEXT_PRIMARY = "#e0e0e0"
TEXT_DIM = "#888888"
TEXT_SUCCESS = "#66cc88"
TEXT_ERROR = "#ff5555"


# ---------------------------------------------------------------------------
# Log capture
# ---------------------------------------------------------------------------

class TUILogHandler(logging.Handler):
    def __init__(self, maxlen: int = 500):
        super().__init__()
        self.records: deque[logging.LogRecord] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        with self._lock:
            self.records.append(record)

    def get_recent(self, limit: int = 50) -> list[logging.LogRecord]:
        with self._lock:
            return list(self.records)[-limit:]

    def clear(self) -> None:
        with self._lock:
            self.records.clear()

_tui_log_handler = TUILogHandler()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fetch_models_local(org_id: str) -> list[dict[str, Any]]:
    """Query models directly from DB (local runtime, no auth)."""
    from ada.server import model_manager
    from infrastructure.database import async_session_maker
    from domains.models.db_models import ModelConfig
    from domains.models.storage import GlyphStorage
    from sqlalchemy import select

    async def _query():
        async with async_session_maker() as session:
            result = await session.execute(
                select(ModelConfig).where(ModelConfig.org_id == org_id)
            )
            configs = result.scalars().all()
            models = []
            storage = GlyphStorage(session)
            for cfg in configs:
                count = await storage.count_glyphs(org_id, cfg.model_id)
                status = "active"
                if model_manager and model_manager.is_encoding(org_id, cfg.model_id):
                    status = "encoding"
                elif model_manager and model_manager.is_model_locked(org_id, cfg.model_id):
                    status = "locked"
                models.append({
                    "model_id": cfg.model_id,
                    "name": cfg.meta_name or cfg.model_id,
                    "version": cfg.model_version or "\u2014",
                    "glyphs": count,
                    "status": status,
                    "short_description": cfg.short_description or "",
                    "runtime": "local",
                })
        return models

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_query())
    finally:
        loop.close()


def _fetch_models_remote(rt: RuntimeConnection) -> list[dict[str, Any]]:
    """Fetch models from a remote runtime via API."""
    import httpx
    url = rt.api_url(f"{rt.org_id}/models")
    with httpx.Client(timeout=10) as client:
        resp = client.get(url, headers=rt.headers())
    resp.raise_for_status()
    models = resp.json().get("models", [])
    for m in models:
        m["runtime"] = rt.name
    return models


def _fetch_tokens_local(org_id: str) -> list[dict[str, Any]]:
    """Query tokens directly from DB (local runtime)."""
    from infrastructure.database import async_session_maker
    from domains.models.db_models import Token
    from sqlalchemy import select

    async def _query():
        async with async_session_maker() as session:
            result = await session.execute(
                select(Token).where(Token.org_id == org_id)
            )
            tokens = result.scalars().all()
            return [{
                "id": str(t.id),
                "name": t.name,
                "org_id": t.org_id,
                "permissions": t.permissions or ["read"],
                "status": t.status,
                "token_prefix": t.token_prefix,
                "created_at": str(t.created_at) if t.created_at else None,
            } for t in tokens]

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_query())
    finally:
        loop.close()


def _fetch_tokens_remote(rt: RuntimeConnection) -> list[dict[str, Any]]:
    """Fetch tokens from a remote runtime via API."""
    import httpx
    url = rt.api_url(f"{rt.org_id}/tokens")
    with httpx.Client(timeout=10) as client:
        resp = client.get(url, headers=rt.headers())
    resp.raise_for_status()
    return resp.json().get("tokens", [])


def _create_token_local(org_id: str, name: str, permissions: list[str] | None = None) -> dict[str, Any]:
    """Create a token directly in the DB (local runtime)."""
    import hashlib
    import secrets
    from datetime import datetime, timedelta
    from infrastructure.database import async_session_maker
    from domains.models.db_models import Token

    raw_token = f"ada_{secrets.token_urlsafe(32)}"
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    token_prefix = raw_token[:12]
    perms = permissions or ["read", "write"]
    expires_at = datetime.utcnow() + timedelta(days=365)

    async def _create():
        async with async_session_maker() as session:
            db_token = Token(
                name=name,
                token_hash=token_hash,
                token_prefix=token_prefix,
                org_id=org_id,
                permissions=perms,
                expires_at=expires_at,
            )
            session.add(db_token)
            await session.commit()
            return {
                "id": str(db_token.id),
                "name": name,
                "token": raw_token,
                "token_prefix": token_prefix,
                "org_id": org_id,
                "permissions": perms,
            }

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_create())
    finally:
        loop.close()


def _revoke_token_local(org_id: str, token_id: str) -> bool:
    """Revoke a token directly in the DB (local runtime)."""
    from datetime import datetime
    from infrastructure.database import async_session_maker
    from domains.models.db_models import Token
    from sqlalchemy import select

    async def _revoke():
        async with async_session_maker() as session:
            result = await session.execute(
                select(Token).where(Token.id == token_id, Token.org_id == org_id, Token.status == "active")
            )
            db_token = result.scalar_one_or_none()
            if not db_token:
                return False
            db_token.status = "revoked"
            db_token.revoked_at = datetime.utcnow()
            await session.commit()
            return True

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_revoke())
    finally:
        loop.close()


def _check_runtime_health(rt: RuntimeConnection) -> dict[str, Any]:
    """Check a remote runtime's health."""
    import httpx
    url = rt.api_url("health")
    with httpx.Client(timeout=5) as client:
        resp = client.get(url, headers=rt.headers())
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class BannerWidget(Static):
    def on_mount(self) -> None:
        from ada import __version__
        self.update(get_banner(version=__version__))


class ServerStatus(Static):
    def set_running(self, port: int) -> None:
        msg = Text()
        msg.append(" \u25cf ", style=f"bold {TEXT_SUCCESS}")
        msg.append(f"Server running on port {port}", style=TEXT_PRIMARY)
        self.update(msg)

    def set_stopped(self) -> None:
        msg = Text()
        msg.append(" \u25cf ", style=f"bold {TEXT_ERROR}")
        msg.append("Server stopped", style=TEXT_DIM)
        msg.append("  /start", style=f"bold {ADA_BLUE}")
        msg.append(" to launch", style=TEXT_DIM)
        self.update(msg)

    def set_starting(self) -> None:
        msg = Text()
        msg.append(" \u25cf ", style=f"bold {ADA_PURPLE}")
        msg.append("Server starting...", style=TEXT_DIM)
        self.update(msg)

    def set_error(self, error: str) -> None:
        msg = Text()
        msg.append(" \u25cf ", style=f"bold {TEXT_ERROR}")
        msg.append(f"Server error: {error}", style=TEXT_ERROR)
        self.update(msg)


class RateGraph(Static):
    def refresh_data(self) -> None:
        series = get_stats().get_rate_series(minutes=5)
        if not series:
            self.update(Panel(Text("Waiting for requests...", style=TEXT_DIM),
                title="Requests/sec", title_align="left", border_style=Style(color="#333333")))
            return
        self.update(Panel(render_graph(series, width=28, height=6, color=ADA_BLUE, axis_color="#444444", fmt=".0f"),
            title="Requests/sec", title_align="left", border_style=Style(color="#333333")))


class LatencyGraph(Static):
    def refresh_data(self) -> None:
        series = get_stats().get_latency_series(minutes=5)
        if not series:
            self.update(Panel(Text("Waiting for requests...", style=TEXT_DIM),
                title="Avg Latency (ms)", title_align="left", border_style=Style(color="#333333")))
            return
        self.update(Panel(render_graph(series, width=28, height=6, color=ADA_PINK, axis_color="#444444", fmt=".0f"),
            title="Avg Latency (ms)", title_align="left", border_style=Style(color="#333333")))


class SummaryBar(Static):
    def refresh_data(self) -> None:
        s = get_stats().get_summary()
        up = s["uptime_seconds"]
        up_s = f"{up:.0f}s" if up < 60 else f"{up/60:.0f}m" if up < 3600 else f"{up/3600:.1f}h"
        rm = get_runtime_manager()
        active = rm.get_active()

        msg = Text()
        msg.append(f" [{active.name}] ", style=f"bold {ADA_PURPLE}")
        msg.append(f"Up {up_s}", style=TEXT_DIM)
        msg.append(f"  Req {s['total_requests']:,}", style=TEXT_PRIMARY)
        msg.append(f"  RPM {s['requests_per_minute']}", style=TEXT_PRIMARY)
        msg.append(f"  Lat {s['avg_latency_ms']:.0f}ms", style=TEXT_PRIMARY)
        msg.append(f"  Err ", style=TEXT_DIM)
        msg.append(f"{s['total_errors']}", style=TEXT_ERROR if s['total_errors'] else TEXT_SUCCESS)
        msg.append(f"  Models {s['models_loaded']}", style=TEXT_PRIMARY)
        self.update(msg)


class ModelsTable(Static):
    def refresh_data(self) -> None:
        models = get_stats().get_models()
        if not models:
            self.update(Panel(Text("No models loaded", style=TEXT_DIM),
                title="Models", title_align="left", border_style=Style(color="#333333")))
            return
        table = Table(show_header=True, header_style=f"bold {ADA_BLUE}", border_style="#333333",
            title="Models", title_style=f"bold {ADA_BLUE}", expand=True, padding=(0, 1))
        table.add_column("Model", style=TEXT_PRIMARY)
        table.add_column("Org", style=TEXT_DIM)
        table.add_column("Glyphs", justify="right", style=TEXT_PRIMARY)
        table.add_column("Tools", justify="right", style=TEXT_PRIMARY)
        table.add_column("Status", justify="center")
        table.add_column("Last Query", justify="right", style=TEXT_DIM)
        for m in sorted(models, key=lambda x: x.model_id):
            st = Text("\u25cf", style=TEXT_SUCCESS if m.status == "loaded" else ADA_PURPLE if m.status == "loading" else TEXT_ERROR)
            if m.last_query_at:
                ago = time.time() - m.last_query_at
                lq = f"{ago:.0f}s ago" if ago < 60 else f"{ago/60:.0f}m ago" if ago < 3600 else f"{ago/3600:.1f}h ago"
            else:
                lq = "-"
            table.add_row(m.model_id, m.org_id, f"{m.glyph_count:,}", str(m.tool_count), st, lq)
        self.update(table)


class RecentRequests(Static):
    def refresh_data(self) -> None:
        recent = get_stats().get_recent_requests(limit=10)
        if not recent:
            self.update(Panel(Text("No requests yet", style=TEXT_DIM),
                title="Recent Requests", title_align="left", border_style=Style(color="#333333")))
            return
        table = Table(show_header=True, header_style=f"bold {ADA_BLUE}", border_style="#333333",
            title="Recent Requests", title_style=f"bold {ADA_BLUE}", expand=True, padding=(0, 1))
        table.add_column("Tool", style=TEXT_PRIMARY)
        table.add_column("Model", style=TEXT_DIM)
        table.add_column("Time", justify="right", style=TEXT_PRIMARY)
        table.add_column("Status", justify="center")
        for r in reversed(recent):
            st = Text("\u2713", style=TEXT_SUCCESS) if r.status == "ok" else Text("\u2717", style=TEXT_ERROR)
            table.add_row(r.tool, r.model_id, f"{r.duration_ms:.0f}ms", st)
        self.update(table)


class LogPanel(Static):
    def refresh_data(self) -> None:
        records = _tui_log_handler.get_recent(limit=30)
        if not records:
            self.update(Text("No log output yet.", style=TEXT_DIM))
            return
        text = Text()
        for rec in records:
            ts = time.strftime("%H:%M:%S", time.localtime(rec.created))
            text.append(f"{ts} ", style="#555555")
            if rec.levelno >= logging.ERROR:
                text.append(f"{rec.levelname:<5} ", style=f"bold {TEXT_ERROR}")
            elif rec.levelno >= logging.WARNING:
                text.append(f"{rec.levelname:<5} ", style=f"bold {ADA_PINK}")
            else:
                text.append(f"{rec.levelname:<5} ", style=TEXT_DIM)
            name = rec.name.removeprefix("ada.").removeprefix("domains.")
            text.append(f"{name:<20} ", style=ADA_PURPLE)
            text.append(f"{rec.getMessage()}\n", style=TEXT_PRIMARY)
        self.update(text)


class ModelsPanel(Static):
    PAGE_SIZE = 6

    def __init__(self, **kwargs) -> None:
        super().__init__(Text("Use /models to list deployed models.", style=TEXT_DIM), **kwargs)
        self._models: list[dict[str, Any]] = []
        self._filtered: list[dict[str, Any]] = []
        self._search: str = ""
        self._page: int = 0
        self._loading: bool = False
        self._error: str = ""
        self._runtime_filter: str = ""  # "" = all runtimes

    def set_search(self, query: str) -> None:
        self._search = query.lower().strip()
        self._page = 0
        self._apply_filter()
        self._render()

    def set_runtime_filter(self, runtime: str) -> None:
        self._runtime_filter = runtime
        self._page = 0
        self._apply_filter()
        self._render()

    def next_page(self) -> None:
        mx = max(0, (len(self._filtered) - 1) // self.PAGE_SIZE)
        if self._page < mx:
            self._page += 1
            self._render()

    def prev_page(self) -> None:
        if self._page > 0:
            self._page -= 1
            self._render()

    def set_models(self, models: list[dict[str, Any]]) -> None:
        self._models = models
        self._loading = False
        self._error = ""
        self._apply_filter()
        self._render()

    def set_loading(self) -> None:
        self._loading = True
        self._render()

    def set_error(self, error: str) -> None:
        self._error = error
        self._loading = False
        self._render()

    def _apply_filter(self) -> None:
        filtered = self._models
        if self._runtime_filter:
            filtered = [m for m in filtered if m.get("runtime", "") == self._runtime_filter]
        if self._search:
            s = self._search
            filtered = [m for m in filtered
                if s in m.get("model_id", "").lower()
                or s in m.get("name", "").lower()
                or s in m.get("version", "").lower()
                or s in m.get("status", "").lower()]
        self._filtered = filtered

    def _render(self) -> None:
        if self._loading:
            self.update(Panel(Text("Loading models...", style=TEXT_DIM),
                title="Models", title_align="left", border_style=Style(color="#333333")))
            return
        if self._error:
            msg = Text()
            msg.append("Error: ", style=f"bold {TEXT_ERROR}")
            msg.append(self._error, style=TEXT_DIM)
            self.update(Panel(msg, title="Models", title_align="left", border_style=Style(color="#333333")))
            return
        if not self._filtered:
            hint = f'No models matching "{self._search}"' if self._search else "No models deployed."
            self.update(Panel(Text(hint, style=TEXT_DIM),
                title="Models", title_align="left", border_style=Style(color="#333333")))
            return

        total = len(self._filtered)
        total_pages = max(1, (total + self.PAGE_SIZE - 1) // self.PAGE_SIZE)
        start = self._page * self.PAGE_SIZE
        page_models = self._filtered[start:start + self.PAGE_SIZE]

        content = Text()
        content.append(f" {total} model{'s' if total != 1 else ''}", style=TEXT_PRIMARY)
        if self._runtime_filter:
            content.append(f"  @{self._runtime_filter}", style=ADA_PURPLE)
        if self._search:
            content.append(f'  filter: "{self._search}"', style=TEXT_DIM)
        content.append(f"  page {self._page + 1}/{total_pages}", style=TEXT_DIM)
        content.append("\n\n")

        for card in page_models:
            rt_name = card.get("runtime", "")
            status = card.get("status", "unknown")
            st_style = TEXT_SUCCESS if status == "active" else ADA_PURPLE if status == "encoding" else ADA_PINK if status == "locked" else TEXT_ERROR
            st_icon = "\u25cf" if status in ("active", "unknown") else "\u25cb"
            name = card.get("name", card.get("model_id", "?"))
            mid = card.get("model_id", "")

            content.append(f"  {st_icon} ", style=st_style)
            content.append(name, style=f"bold {TEXT_PRIMARY}")
            if name != mid:
                content.append(f"  ({mid})", style=TEXT_DIM)
            if rt_name:
                content.append(f"  @{rt_name}", style=ADA_PURPLE)
            content.append("\n")
            content.append(f"    v{card.get('version', '\u2014')}", style=ADA_BLUE)
            content.append(f"  {status}", style=st_style)
            content.append(f"  {card.get('glyphs', 0):,} glyphs", style=TEXT_DIM)
            content.append("\n")
            desc = card.get("short_description", "")
            if desc:
                content.append(f"    {desc[:80]}{'...' if len(desc) > 80 else ''}\n", style=TEXT_DIM)

        self.update(content)


class RuntimesPanel(Static):
    """Shows all configured runtimes with status."""

    def __init__(self, **kwargs) -> None:
        super().__init__(Text("Loading runtimes...", style=TEXT_DIM), **kwargs)

    def refresh_data(self) -> None:
        rm = get_runtime_manager()
        runtimes = rm.list_all()
        active = rm.get_active()

        if not runtimes:
            self.update(Text("No runtimes configured.", style=TEXT_DIM))
            return

        content = Text()
        for rt in runtimes:
            is_active = rt.name == active.name
            # Status indicator
            if rt.status == "connected":
                content.append(" \u25cf ", style=f"bold {TEXT_SUCCESS}")
            elif rt.status == "error":
                content.append(" \u25cf ", style=f"bold {TEXT_ERROR}")
            else:
                content.append(" \u25cb ", style=TEXT_DIM)

            # Name
            if is_active:
                content.append(f"{rt.name}", style=f"bold underline {ADA_BLUE}")
                content.append(" (active)", style=ADA_BLUE)
            else:
                content.append(f"{rt.name}", style=f"bold {TEXT_PRIMARY}")

            # Details
            if rt.is_local:
                content.append("  local", style=TEXT_DIM)
            else:
                content.append(f"  {rt.url}", style=TEXT_DIM)
            content.append(f"  org={rt.org_id}", style=TEXT_DIM)
            if rt.token:
                content.append(f"  token={rt.token[:8]}...", style=TEXT_DIM)
            content.append("\n")

            # Config summary
            cfg = rt.config
            content.append(f"    rate={cfg.rate_limit_per_minute}/min", style=TEXT_DIM)
            content.append(f"  cors={'*' if cfg.cors_allow_all else 'restricted'}", style=TEXT_DIM)
            content.append(f"  docs={'on' if cfg.enable_docs else 'off'}", style=TEXT_DIM)
            content.append(f"  dim={cfg.max_vector_dimension}", style=TEXT_DIM)
            if rt._error:
                content.append(f"\n    error: {rt._error}", style=TEXT_ERROR)
            content.append("\n\n")

        self.update(content)


class TokensPanel(Static):
    """Shows API tokens for the active runtime."""

    def __init__(self, **kwargs) -> None:
        super().__init__(Text("Use /tokens to list API tokens.", style=TEXT_DIM), **kwargs)
        self._tokens: list[dict[str, Any]] = []
        self._loading: bool = False
        self._error: str = ""

    def set_tokens(self, tokens: list[dict[str, Any]]) -> None:
        self._tokens = tokens
        self._loading = False
        self._error = ""
        self._render()

    def set_loading(self) -> None:
        self._loading = True
        self._render()

    def set_error(self, error: str) -> None:
        self._error = error
        self._loading = False
        self._render()

    def _render(self) -> None:
        if self._loading:
            self.update(Text("Loading tokens...", style=TEXT_DIM))
            return
        if self._error:
            msg = Text()
            msg.append("Error: ", style=f"bold {TEXT_ERROR}")
            msg.append(self._error, style=TEXT_DIM)
            self.update(msg)
            return
        if not self._tokens:
            self.update(Text("No tokens. Use /tokens create <name> to create one.", style=TEXT_DIM))
            return

        table = Table(show_header=True, header_style=f"bold {ADA_BLUE}", border_style="#333333",
            expand=True, padding=(0, 1))
        table.add_column("Name", style=TEXT_PRIMARY)
        table.add_column("Prefix", style=TEXT_DIM)
        table.add_column("Permissions", style=TEXT_PRIMARY)
        table.add_column("Status", justify="center")
        table.add_column("Created", justify="right", style=TEXT_DIM)

        for t in self._tokens:
            st = Text("\u25cf", style=TEXT_SUCCESS if t.get("status") == "active" else TEXT_ERROR)
            perms = ", ".join(t.get("permissions", []))
            table.add_row(
                t.get("name", "?"),
                t.get("token_prefix", "?"),
                perms,
                st,
                t.get("created_at", "-"),
            )
        self.update(table)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

class AdaApp(App):
    TITLE = "Ada"
    CSS = """
    Screen { background: black; color: #e0e0e0; }
    BannerWidget { height: auto; padding: 1 2; text-align: center; }
    ServerStatus { height: 1; padding: 0 2; }
    SummaryBar { height: 1; padding: 0 2; background: #111111; }
    #graphs { height: auto; padding: 0 2; }
    RateGraph { width: 1fr; }
    LatencyGraph { width: 1fr; }
    ModelsTable { height: auto; padding: 0 2; }
    RecentRequests { height: auto; padding: 0 2; }
    ModelsPanel { height: auto; padding: 0 2; }
    RuntimesPanel { height: auto; padding: 0 2; }
    TokensPanel { height: auto; padding: 0 2; }
    LogPanel { height: auto; min-height: 8; padding: 0 2; }
    #tabs { height: 1fr; }
    TabPane { padding: 0; }
    #dashboard-scroll, #models-scroll, #runtimes-scroll, #tokens-scroll, #logs-scroll { height: 1fr; }
    #messages-scroll { height: 1fr; padding: 0 2; }
    Header { background: #111111; color: #6b7bff; }
    Footer { background: #111111; color: #888888; }
    FooterKey { background: #111111; color: #6b7bff; }
    #prompt-input { dock: bottom; padding: 0 1; background: #111111; color: #e0e0e0; border: tall #333333; }
    #prompt-input:focus { border: tall #6b7bff; }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "refresh_all", "Refresh", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._server_running = False
        self._server_thread: threading.Thread | None = None
        self._uvicorn_server = None
        self._refresh_timer: Timer | None = None
        self._server_port: int = 8002
        self._rm = get_runtime_manager()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield BannerWidget()
        yield ServerStatus()
        yield SummaryBar()
        with TabbedContent(id="tabs"):
            with TabPane("Dashboard", id="tab-dashboard"):
                with VerticalScroll(id="dashboard-scroll"):
                    with Horizontal(id="graphs"):
                        yield RateGraph()
                        yield LatencyGraph()
                    yield ModelsTable()
                    yield RecentRequests()
            with TabPane("Models", id="tab-models"):
                with VerticalScroll(id="models-scroll"):
                    yield ModelsPanel()
            with TabPane("Runtimes", id="tab-runtimes"):
                with VerticalScroll(id="runtimes-scroll"):
                    yield RuntimesPanel()
            with TabPane("Tokens", id="tab-tokens"):
                with VerticalScroll(id="tokens-scroll"):
                    yield TokensPanel()
            with TabPane("Logs", id="tab-logs"):
                with VerticalScroll(id="logs-scroll"):
                    yield LogPanel()
            with TabPane("Output", id="tab-output"):
                yield VerticalScroll(id="messages-scroll")
        yield Input(placeholder="Type a command... (/help)", id="prompt-input")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(Input).focus()
        logging.getLogger().addHandler(_tui_log_handler)
        _tui_log_handler.setLevel(logging.DEBUG)
        self.query_one(ServerStatus).set_stopped()
        self._refresh_timer = self.set_interval(2.0, self._refresh_dashboard)

    # -- Server management --

    def _start_server(self, port: int | None = None) -> None:
        if self._server_running:
            return
        import uvicorn
        from infrastructure.config import get_settings
        settings = get_settings()
        self._server_port = port or settings.port
        status = self.query_one(ServerStatus)
        status.set_starting()

        config = uvicorn.Config("ada.server:app", host=settings.host, port=self._server_port, log_level="info")
        self._uvicorn_server = uvicorn.Server(config)

        def _run():
            try:
                self._server_running = True
                self.call_from_thread(status.set_running, self._server_port)
                self._rm.update_status("local", "connected")
                self._uvicorn_server.run()
            except OSError as e:
                err = f"Port {self._server_port} already in use" if "address already in use" in str(e).lower() else str(e)
                self.call_from_thread(status.set_error, err)
                self._rm.update_status("local", "error", err)
            except Exception as e:
                self.call_from_thread(status.set_error, str(e))
                self._rm.update_status("local", "error", str(e))
            finally:
                self._server_running = False
                self._uvicorn_server = None
                self._rm.update_status("local", "disconnected")
                try:
                    self.call_from_thread(self.query_one(ServerStatus).set_stopped)
                except Exception:
                    pass

        self._server_thread = threading.Thread(target=_run, daemon=True, name="ada-server")
        self._server_thread.start()

    def _stop_server(self) -> None:
        if self._server_running and self._uvicorn_server:
            self._uvicorn_server.should_exit = True

    # -- Dashboard refresh --

    def _refresh_dashboard(self) -> None:
        try:
            if self._server_running:
                self.query_one(SummaryBar).refresh_data()
                self.query_one(RateGraph).refresh_data()
                self.query_one(LatencyGraph).refresh_data()
                self.query_one(ModelsTable).refresh_data()
                self.query_one(RecentRequests).refresh_data()
            self.query_one(LogPanel).refresh_data()
        except Exception:
            pass

    # -- Tab auto-fetch --

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        if event.pane.id == "tab-models":
            panel = self.query_one(ModelsPanel)
            if not panel._models and not panel._loading and self._server_running:
                self._fetch_models()
        elif event.pane.id == "tab-runtimes":
            self.query_one(RuntimesPanel).refresh_data()
        elif event.pane.id == "tab-tokens":
            panel = self.query_one(TokensPanel)
            if not panel._tokens and not panel._loading and self._server_running:
                self._fetch_tokens()

    # -- Data fetchers --

    def _fetch_models(self, runtime_name: str | None = None) -> None:
        panel = self.query_one(ModelsPanel)
        panel.set_loading()

        def _do():
            try:
                all_models = []
                targets = [self._rm.get(runtime_name)] if runtime_name else self._rm.list_all()
                for rt in targets:
                    if rt is None:
                        continue
                    try:
                        if rt.is_local and self._server_running:
                            models = _fetch_models_local(rt.org_id)
                        elif not rt.is_local and rt.url:
                            models = _fetch_models_remote(rt)
                        else:
                            continue
                        all_models.extend(models)
                        self._rm.update_status(rt.name, "connected")
                    except Exception as e:
                        self._rm.update_status(rt.name, "error", str(e))
                self.call_from_thread(panel.set_models, all_models)
            except Exception as e:
                self.call_from_thread(panel.set_error, str(e))

        self.run_worker(_do, name="fetch-models", exclusive=True, thread=True)

    def _fetch_tokens(self) -> None:
        panel = self.query_one(TokensPanel)
        panel.set_loading()
        rt = self._rm.get_active()

        def _do():
            try:
                if rt.is_local and self._server_running:
                    tokens = _fetch_tokens_local(rt.org_id)
                elif not rt.is_local and rt.url:
                    tokens = _fetch_tokens_remote(rt)
                else:
                    tokens = []
                self.call_from_thread(panel.set_tokens, tokens)
            except Exception as e:
                self.call_from_thread(panel.set_error, str(e))

        self.run_worker(_do, name="fetch-tokens", exclusive=True, thread=True)

    # -- Command dispatch --

    def on_input_submitted(self, event: Input.Submitted) -> None:
        message = event.value.strip()
        if not message:
            return
        event.input.clear()
        parts = message.split()
        cmd = parts[0].lower()

        if cmd == "/help":
            self._show_help()
        elif cmd == "/start":
            port = int(parts[1]) if len(parts) > 1 else None
            if self._server_running:
                self._msg(f"Server already running on port {self._server_port}.")
            else:
                self._start_server(port)
        elif cmd == "/stop":
            if not self._server_running:
                self._msg("Server is not running.")
            else:
                self._stop_server()
                self._msg("Server stopping...")
        elif cmd == "/status":
            self._msg(f"Server {'running on port ' + str(self._server_port) if self._server_running else 'stopped'}.")

        # -- Runtimes --
        elif cmd == "/runtime":
            self._handle_runtime(parts[1:])
        elif cmd == "/use":
            if len(parts) > 1:
                if self._rm.set_active(parts[1]):
                    self._msg(f"Active runtime: {parts[1]}")
                else:
                    self._msg(f"Unknown runtime: {parts[1]}")
            else:
                self._msg(f"Active runtime: {self._rm.get_active().name}")

        # -- Config --
        elif cmd == "/config":
            self._handle_config(parts[1:])

        # -- Models --
        elif cmd == "/models":
            if len(parts) > 1:
                sub = parts[1].lower()
                if sub == "next":
                    self.query_one(ModelsPanel).next_page()
                elif sub == "prev":
                    self.query_one(ModelsPanel).prev_page()
                elif sub == "refresh":
                    self._fetch_models()
                elif sub.startswith("@"):
                    self.query_one(ModelsPanel).set_runtime_filter(sub[1:])
                    self._fetch_models(sub[1:])
                else:
                    self.query_one(ModelsPanel).set_search(" ".join(parts[1:]))
            else:
                self._fetch_models()
            self.query_one(TabbedContent).active = "tab-models"

        # -- Tokens --
        elif cmd == "/tokens":
            if len(parts) > 1:
                sub = parts[1].lower()
                if sub == "create" and len(parts) >= 3:
                    self._create_token(parts[2], parts[3:])
                elif sub == "revoke" and len(parts) >= 3:
                    self._revoke_token(parts[2])
                elif sub == "refresh":
                    self._fetch_tokens()
                    self.query_one(TabbedContent).active = "tab-tokens"
                else:
                    self._msg("Usage: /tokens create <name> [--perms read,write,admin]")
                    self._msg("       /tokens revoke <token_id>")
                    self._msg("       /tokens refresh")
            else:
                self._fetch_tokens()
                self.query_one(TabbedContent).active = "tab-tokens"

        # -- Views --
        elif cmd == "/logs":
            self.query_one(TabbedContent).active = "tab-logs"
        elif cmd == "/clear":
            if len(parts) > 1 and parts[1] == "logs":
                _tui_log_handler.clear()
                self._msg("Logs cleared.")
            else:
                try:
                    self.query_one("#messages-scroll", VerticalScroll).remove_children()
                except Exception:
                    pass
                self.query_one(TabbedContent).active = "tab-dashboard"
        elif cmd in ("/exit", "/quit"):
            self.exit()
        else:
            self._msg(f"Unknown command: {cmd}. Try /help")

    # -- Runtime commands --

    def _handle_runtime(self, args: list[str]) -> None:
        if not args:
            self.query_one(TabbedContent).active = "tab-runtimes"
            self.query_one(RuntimesPanel).refresh_data()
            return

        sub = args[0].lower()
        if sub == "add" and len(args) >= 3:
            name, url = args[1], args[2]
            token = None
            org = "default"
            for i, a in enumerate(args[3:], 3):
                if a in ("--token", "-t") and i + 1 < len(args):
                    token = args[i + 1]
                elif a in ("--org", "-o") and i + 1 < len(args):
                    org = args[i + 1]
            self._rm.add(name, url, token=token, org_id=org)
            self._msg(f"Runtime '{name}' added: {url}")
        elif sub == "remove" and len(args) >= 2:
            name = args[1]
            if self._rm.remove(name):
                self._msg(f"Runtime '{name}' removed.")
            else:
                self._msg(f"Cannot remove '{name}' (local runtime cannot be removed).")
        elif sub == "list":
            self.query_one(TabbedContent).active = "tab-runtimes"
            self.query_one(RuntimesPanel).refresh_data()
        elif sub == "check" and len(args) >= 2:
            self._check_runtime(args[1])
        else:
            self._msg("Usage: /runtime add <name> <url> [--token <t>] [--org <o>]")
            self._msg("       /runtime remove <name>")
            self._msg("       /runtime list")
            self._msg("       /runtime check <name>")

    def _check_runtime(self, name: str) -> None:
        rt = self._rm.get(name)
        if not rt:
            self._msg(f"Unknown runtime: {name}")
            return
        if rt.is_local:
            if self._server_running:
                self._rm.update_status(name, "connected")
                self._msg(f"Runtime '{name}': connected (local, port {self._server_port})")
            else:
                self._rm.update_status(name, "disconnected")
                self._msg(f"Runtime '{name}': disconnected (server not running)")
            return

        def _do():
            try:
                health = _check_runtime_health(rt)
                self._rm.update_status(name, "connected")
                self.call_from_thread(self._msg, f"Runtime '{name}': connected ({health.get('status', 'ok')})")
            except Exception as e:
                self._rm.update_status(name, "error", str(e))
                self.call_from_thread(self._msg, f"Runtime '{name}': error ({e})")

        self.run_worker(_do, name=f"check-{name}", thread=True)

    # -- Config commands --

    def _handle_config(self, args: list[str]) -> None:
        active = self._rm.get_active()
        if not args:
            cfg = active.config
            self._msg(f"Config for [{active.name}]:")
            self._msg(f"  rate_limit_per_minute = {cfg.rate_limit_per_minute}")
            self._msg(f"  cors_allow_all = {cfg.cors_allow_all}")
            self._msg(f"  cors_origins = {cfg.cors_origins}")
            self._msg(f"  enable_docs = {cfg.enable_docs}")
            self._msg(f"  max_vector_dimension = {cfg.max_vector_dimension}")
            self._msg(f"  log_level = {cfg.log_level}")
            return

        if len(args) < 2:
            self._msg("Usage: /config <key> <value>")
            return

        key, raw_value = args[0], " ".join(args[1:])
        # Type coercion
        if raw_value.lower() in ("true", "false"):
            value: Any = raw_value.lower() == "true"
        elif raw_value.isdigit():
            value = int(raw_value)
        elif raw_value.startswith("["):
            import json
            try:
                value = json.loads(raw_value)
            except json.JSONDecodeError:
                self._msg(f"Invalid JSON for {key}")
                return
        else:
            value = raw_value

        if self._rm.update_config(active.name, **{key: value}):
            self._msg(f"[{active.name}] {key} = {value}")
            # Apply to running server if local
            if active.is_local and self._server_running:
                self._apply_local_config(key, value)
        else:
            self._msg(f"Unknown config key: {key}")

    def _apply_local_config(self, key: str, value: Any) -> None:
        """Apply a config change to the running local server."""
        try:
            if key == "log_level":
                logging.getLogger().setLevel(getattr(logging, str(value).upper(), logging.INFO))
                self._msg(f"  Applied: log level set to {value}")
        except Exception as e:
            self._msg(f"  Warning: could not apply {key} live: {e}")

    # -- Token management --

    def _create_token(self, name: str, extra_args: list[str]) -> None:
        if not self._server_running:
            self._msg("Server is not running. Use /start first.")
            return
        rt = self._rm.get_active()

        # Parse --perms flag
        perms = None
        for i, a in enumerate(extra_args):
            if a in ("--perms", "-p") and i + 1 < len(extra_args):
                perms = extra_args[i + 1].split(",")

        def _do():
            try:
                if rt.is_local:
                    result = _create_token_local(rt.org_id, name, perms)
                else:
                    self.call_from_thread(self._msg, "Remote token creation not yet supported. Use the API directly.")
                    return
                self.call_from_thread(self._msg, f"Token created: {result['name']}")
                self.call_from_thread(self._msg, f"  Token: {result['token']}")
                self.call_from_thread(self._msg, f"  Prefix: {result['token_prefix']}")
                self.call_from_thread(self._msg, "  Store this token securely — it won't be shown again.")
                self.call_from_thread(self._fetch_tokens)
            except Exception as e:
                self.call_from_thread(self._msg, f"Error creating token: {e}")

        self.run_worker(_do, name="create-token", thread=True)

    def _revoke_token(self, token_id: str) -> None:
        if not self._server_running:
            self._msg("Server is not running. Use /start first.")
            return
        rt = self._rm.get_active()

        def _do():
            try:
                if rt.is_local:
                    ok = _revoke_token_local(rt.org_id, token_id)
                else:
                    self.call_from_thread(self._msg, "Remote token revocation not yet supported.")
                    return
                if ok:
                    self.call_from_thread(self._msg, f"Token {token_id} revoked.")
                    self.call_from_thread(self._fetch_tokens)
                else:
                    self.call_from_thread(self._msg, f"Token {token_id} not found or already revoked.")
            except Exception as e:
                self.call_from_thread(self._msg, f"Error revoking token: {e}")

        self.run_worker(_do, name="revoke-token", thread=True)

    # -- UI helpers --

    def _msg(self, content: str) -> None:
        try:
            log = self.query_one("#messages-scroll", VerticalScroll)
        except Exception:
            return
        msg = Text()
        msg.append(f"{time.strftime('%H:%M:%S')} ", style="#555555")
        msg.append("\u2022 ", style=f"bold {ADA_PURPLE}")
        msg.append(content, style=TEXT_DIM)
        log.mount(Static(msg))
        self.query_one(TabbedContent).active = "tab-output"

    def _show_help(self) -> None:
        try:
            log = self.query_one("#messages-scroll", VerticalScroll)
        except Exception:
            return
        active = self._rm.get_active()
        h = Text()
        h.append("Server:\n", style=f"bold {TEXT_PRIMARY}")
        h.append("  /start [port]                \u2014 Start local server\n", style=TEXT_PRIMARY)
        h.append("  /stop                         \u2014 Stop local server\n", style=TEXT_PRIMARY)
        h.append("  /status                       \u2014 Server status\n", style=TEXT_PRIMARY)
        h.append("\nRuntimes:\n", style=f"bold {TEXT_PRIMARY}")
        h.append("  /runtime                      \u2014 Show runtimes tab\n", style=TEXT_PRIMARY)
        h.append("  /runtime add <name> <url>     \u2014 Add remote runtime\n", style=TEXT_PRIMARY)
        h.append("    [--token <t>] [--org <o>]\n", style=TEXT_DIM)
        h.append("  /runtime remove <name>        \u2014 Remove runtime\n", style=TEXT_PRIMARY)
        h.append("  /runtime check <name>         \u2014 Health check\n", style=TEXT_PRIMARY)
        h.append(f"  /use <name>                   \u2014 Switch active (current: {active.name})\n", style=TEXT_PRIMARY)
        h.append("\nModels:\n", style=f"bold {TEXT_PRIMARY}")
        h.append("  /models                       \u2014 List all models\n", style=TEXT_PRIMARY)
        h.append("  /models @<runtime>            \u2014 Filter by runtime\n", style=TEXT_PRIMARY)
        h.append("  /models <search>              \u2014 Search models\n", style=TEXT_PRIMARY)
        h.append("  /models next | prev           \u2014 Paginate\n", style=TEXT_PRIMARY)
        h.append("\nTokens:\n", style=f"bold {TEXT_PRIMARY}")
        h.append("  /tokens                       \u2014 List tokens\n", style=TEXT_PRIMARY)
        h.append("  /tokens create <name>         \u2014 Create token\n", style=TEXT_PRIMARY)
        h.append("    [--perms read,write,admin]\n", style=TEXT_DIM)
        h.append("  /tokens revoke <id>           \u2014 Revoke token\n", style=TEXT_PRIMARY)
        h.append("\nConfig:\n", style=f"bold {TEXT_PRIMARY}")
        h.append("  /config                       \u2014 Show config\n", style=TEXT_PRIMARY)
        h.append("  /config <key> <value>         \u2014 Set config value\n", style=TEXT_PRIMARY)
        h.append("    Keys: rate_limit_per_minute, cors_allow_all,\n", style=TEXT_DIM)
        h.append("    cors_origins, enable_docs, max_vector_dimension, log_level\n", style=TEXT_DIM)
        h.append("\nViews:\n", style=f"bold {TEXT_PRIMARY}")
        h.append("  /logs                         \u2014 Logs tab\n", style=TEXT_PRIMARY)
        h.append("  /clear                        \u2014 Clear, back to dashboard\n", style=TEXT_PRIMARY)
        h.append("  /clear logs                   \u2014 Clear logs\n", style=TEXT_PRIMARY)
        h.append("  /exit                         \u2014 Quit\n", style=TEXT_PRIMARY)
        log.mount(Static(h))
        self.query_one(TabbedContent).active = "tab-output"

    def action_refresh_all(self) -> None:
        self._refresh_dashboard()


def main() -> None:
    app = AdaApp()
    app.run()

if __name__ == "__main__":
    main()
