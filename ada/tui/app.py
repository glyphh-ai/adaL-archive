"""Ada Runtime — terminal dashboard."""

from __future__ import annotations

import asyncio
import logging
import time
import threading
from collections import deque

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Static, Input, Footer, Header, TabbedContent, TabPane
from textual.binding import Binding
from textual.timer import Timer
from rich.text import Text
from rich.table import Table
from rich.style import Style
from rich.panel import Panel

from ada.tui.banner import get_banner
from ada.tui.graph import render_graph
from ada.tui.stats import get_stats, RequestRecord, ModelStats

# Ada's color palette — derived from the banner gradient
ADA_BLUE = "#6b7bff"
ADA_PINK = "#ff6b6b"
ADA_PURPLE = "#9b6bea"
TEXT_PRIMARY = "#e0e0e0"
TEXT_DIM = "#888888"
TEXT_SUCCESS = "#66cc88"
TEXT_ERROR = "#ff5555"


# ---------------------------------------------------------------------------
# Log capture — routes Python logging into the TUI
# ---------------------------------------------------------------------------

class TUILogHandler(logging.Handler):
    """Captures log records into a deque for the TUI to display."""

    def __init__(self, maxlen: int = 500):
        super().__init__()
        self.records: deque[logging.LogRecord] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        with self._lock:
            self.records.append(record)

    def get_recent(self, limit: int = 50) -> list[logging.LogRecord]:
        with self._lock:
            items = list(self.records)
            return items[-limit:]

    def clear(self) -> None:
        with self._lock:
            self.records.clear()


# Global log handler — attached during app startup
_tui_log_handler = TUILogHandler()


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class BannerWidget(Static):
    """Displays the Ada braille banner."""

    def on_mount(self) -> None:
        from ada import __version__
        self.update(get_banner(version=__version__))


class ServerStatus(Static):
    """Shows server running/stopped status."""

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
    """Live braille graph showing requests per second."""

    def refresh_data(self) -> None:
        stats = get_stats()
        series = stats.get_rate_series(minutes=5)
        if not series:
            self.update(Panel(
                Text("Waiting for requests...", style=TEXT_DIM),
                title="Requests/sec",
                title_align="left",
                border_style=Style(color="#333333"),
            ))
            return
        graph = render_graph(
            series, width=28, height=6,
            color=ADA_BLUE, axis_color="#444444", fmt=".0f",
        )
        self.update(Panel(
            graph,
            title="Requests/sec",
            title_align="left",
            border_style=Style(color="#333333"),
        ))


class LatencyGraph(Static):
    """Live braille graph showing average response time."""

    def refresh_data(self) -> None:
        stats = get_stats()
        series = stats.get_latency_series(minutes=5)
        if not series:
            self.update(Panel(
                Text("Waiting for requests...", style=TEXT_DIM),
                title="Avg Latency (ms)",
                title_align="left",
                border_style=Style(color="#333333"),
            ))
            return
        graph = render_graph(
            series, width=28, height=6,
            color=ADA_PINK, axis_color="#444444", fmt=".0f",
        )
        self.update(Panel(
            graph,
            title="Avg Latency (ms)",
            title_align="left",
            border_style=Style(color="#333333"),
        ))


class SummaryBar(Static):
    """One-line summary of runtime stats."""

    def refresh_data(self) -> None:
        stats = get_stats()
        summary = stats.get_summary()

        uptime = summary["uptime_seconds"]
        if uptime < 60:
            uptime_str = f"{uptime:.0f}s"
        elif uptime < 3600:
            uptime_str = f"{uptime / 60:.0f}m"
        else:
            uptime_str = f"{uptime / 3600:.1f}h"

        msg = Text()
        msg.append(" Uptime ", style=f"bold {ADA_BLUE}")
        msg.append(uptime_str, style=TEXT_PRIMARY)
        msg.append("  Requests ", style=f"bold {ADA_BLUE}")
        msg.append(f"{summary['total_requests']:,}", style=TEXT_PRIMARY)
        msg.append("  RPM ", style=f"bold {ADA_BLUE}")
        msg.append(f"{summary['requests_per_minute']}", style=TEXT_PRIMARY)
        msg.append("  Avg Latency ", style=f"bold {ADA_BLUE}")
        msg.append(f"{summary['avg_latency_ms']:.0f}ms", style=TEXT_PRIMARY)
        msg.append("  Errors ", style=f"bold {ADA_BLUE}")
        if summary["total_errors"] > 0:
            msg.append(f"{summary['total_errors']}", style=f"bold {TEXT_ERROR}")
        else:
            msg.append("0", style=TEXT_SUCCESS)
        msg.append("  Models ", style=f"bold {ADA_BLUE}")
        msg.append(f"{summary['models_loaded']}", style=TEXT_PRIMARY)

        self.update(msg)


class ModelsTable(Static):
    """Table showing loaded models."""

    def refresh_data(self) -> None:
        stats = get_stats()
        models = stats.get_models()

        if not models:
            self.update(Panel(
                Text("No models loaded", style=TEXT_DIM),
                title="Models",
                title_align="left",
                border_style=Style(color="#333333"),
            ))
            return

        table = Table(
            show_header=True,
            header_style=f"bold {ADA_BLUE}",
            border_style="#333333",
            title="Models",
            title_style=f"bold {ADA_BLUE}",
            expand=True,
            padding=(0, 1),
        )
        table.add_column("Model", style=TEXT_PRIMARY)
        table.add_column("Org", style=TEXT_DIM)
        table.add_column("Glyphs", justify="right", style=TEXT_PRIMARY)
        table.add_column("Tools", justify="right", style=TEXT_PRIMARY)
        table.add_column("Status", justify="center")
        table.add_column("Last Query", justify="right", style=TEXT_DIM)

        for m in sorted(models, key=lambda x: x.model_id):
            if m.status == "loaded":
                status = Text("\u25cf", style=TEXT_SUCCESS)
            elif m.status == "loading":
                status = Text("\u25cf", style=ADA_PURPLE)
            else:
                status = Text("\u25cf", style=TEXT_ERROR)

            if m.last_query_at:
                ago = time.time() - m.last_query_at
                if ago < 60:
                    last_q = f"{ago:.0f}s ago"
                elif ago < 3600:
                    last_q = f"{ago / 60:.0f}m ago"
                else:
                    last_q = f"{ago / 3600:.1f}h ago"
            else:
                last_q = "-"

            table.add_row(
                m.model_id,
                m.org_id,
                f"{m.glyph_count:,}",
                str(m.tool_count),
                status,
                last_q,
            )

        self.update(table)


class RecentRequests(Static):
    """Rolling log of recent MCP requests."""

    def refresh_data(self) -> None:
        stats = get_stats()
        recent = stats.get_recent_requests(limit=10)

        if not recent:
            self.update(Panel(
                Text("No requests yet", style=TEXT_DIM),
                title="Recent Requests",
                title_align="left",
                border_style=Style(color="#333333"),
            ))
            return

        table = Table(
            show_header=True,
            header_style=f"bold {ADA_BLUE}",
            border_style="#333333",
            title="Recent Requests",
            title_style=f"bold {ADA_BLUE}",
            expand=True,
            padding=(0, 1),
        )
        table.add_column("Tool", style=TEXT_PRIMARY)
        table.add_column("Model", style=TEXT_DIM)
        table.add_column("Time", justify="right", style=TEXT_PRIMARY)
        table.add_column("Status", justify="center")

        for r in reversed(recent):
            status = Text("\u2713", style=TEXT_SUCCESS) if r.status == "ok" else Text("\u2717", style=TEXT_ERROR)
            table.add_row(
                r.tool,
                r.model_id,
                f"{r.duration_ms:.0f}ms",
                status,
            )

        self.update(table)


class LogPanel(Static):
    """Displays captured log output."""

    def refresh_data(self) -> None:
        records = _tui_log_handler.get_recent(limit=30)
        if not records:
            self.update(Text("No log output yet.", style=TEXT_DIM))
            return

        text = Text()
        for rec in records:
            # Timestamp
            ts = time.strftime("%H:%M:%S", time.localtime(rec.created))
            text.append(f"{ts} ", style="#555555")

            # Level
            if rec.levelno >= logging.ERROR:
                text.append(f"{rec.levelname:<5} ", style=f"bold {TEXT_ERROR}")
            elif rec.levelno >= logging.WARNING:
                text.append(f"{rec.levelname:<5} ", style=f"bold {ADA_PINK}")
            else:
                text.append(f"{rec.levelname:<5} ", style=TEXT_DIM)

            # Logger name (shortened)
            name = rec.name
            if name.startswith("ada."):
                name = name[4:]
            elif name.startswith("domains."):
                name = name[8:]
            text.append(f"{name:<20} ", style=ADA_PURPLE)

            # Message
            text.append(f"{rec.getMessage()}\n", style=TEXT_PRIMARY)

        self.update(text)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

class AdaApp(App):
    """Ada Runtime terminal dashboard."""

    TITLE = "Ada"
    CSS = """
    Screen {
        background: black;
        color: #e0e0e0;
    }

    BannerWidget {
        height: auto;
        padding: 1 2;
        text-align: center;
    }

    ServerStatus {
        height: 1;
        padding: 0 2;
    }

    SummaryBar {
        height: 1;
        padding: 0 2;
        background: #111111;
    }

    #graphs {
        height: auto;
        padding: 0 2;
    }

    RateGraph {
        width: 1fr;
    }

    LatencyGraph {
        width: 1fr;
    }

    ModelsTable {
        height: auto;
        padding: 0 2;
    }

    RecentRequests {
        height: auto;
        padding: 0 2;
    }

    LogPanel {
        height: auto;
        min-height: 8;
        padding: 0 2;
    }

    #tabs {
        height: 1fr;
    }

    TabPane {
        padding: 0;
    }

    #dashboard-scroll {
        height: 1fr;
    }

    #logs-scroll {
        height: 1fr;
    }

    #messages-scroll {
        height: 1fr;
        padding: 0 2;
    }

    Header {
        background: #111111;
        color: #6b7bff;
    }

    Footer {
        background: #111111;
        color: #888888;
    }

    FooterKey {
        background: #111111;
        color: #6b7bff;
    }

    #prompt-input {
        dock: bottom;
        padding: 0 1;
        background: #111111;
        color: #e0e0e0;
        border: tall #333333;
    }

    #prompt-input:focus {
        border: tall #6b7bff;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "refresh_all", "Refresh", show=True),
    ]

    def __init__(self, server_started: bool = False) -> None:
        super().__init__()
        self._server_started = server_started
        self._server_running = server_started
        self._server_thread: threading.Thread | None = None
        self._uvicorn_server = None
        self._refresh_timer: Timer | None = None
        self._server_port: int = 8002

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
            with TabPane("Logs", id="tab-logs"):
                with VerticalScroll(id="logs-scroll"):
                    yield LogPanel()
            with TabPane("Output", id="tab-output"):
                yield VerticalScroll(id="messages-scroll")
        yield Input(placeholder="Type a command... (/help)", id="prompt-input")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(Input).focus()

        # Attach log handler to root logger to capture everything
        root_logger = logging.getLogger()
        root_logger.addHandler(_tui_log_handler)
        _tui_log_handler.setLevel(logging.DEBUG)

        # Show initial server status
        status_widget = self.query_one(ServerStatus)
        if self._server_started:
            from infrastructure.config import get_settings
            self._server_port = get_settings().port
            status_widget.set_running(self._server_port)
        else:
            status_widget.set_stopped()

        # Refresh dashboard every 2 seconds
        self._refresh_timer = self.set_interval(2.0, self._refresh_dashboard)

    def _start_server(self, port: int | None = None) -> None:
        """Start the FastAPI server in a background thread."""
        if self._server_running:
            return

        import uvicorn
        from infrastructure.config import get_settings

        settings = get_settings()
        self._server_port = port or settings.port

        status_widget = self.query_one(ServerStatus)
        status_widget.set_starting()

        config = uvicorn.Config(
            "ada.server:app",
            host=settings.host,
            port=self._server_port,
            log_level="info",
        )
        self._uvicorn_server = uvicorn.Server(config)

        def _run():
            # uvicorn.Server.run() creates its own event loop
            try:
                self._server_running = True
                self.call_from_thread(status_widget.set_running, self._server_port)
                self._uvicorn_server.run()
            except OSError as e:
                error_msg = str(e)
                if "address already in use" in error_msg.lower():
                    error_msg = f"Port {self._server_port} already in use"
                self.call_from_thread(status_widget.set_error, error_msg)
                self.call_from_thread(
                    self._show_message, "system",
                    f"Failed to start: {error_msg}. Try /start <different-port>"
                )
            except Exception as e:
                self.call_from_thread(status_widget.set_error, str(e))
            finally:
                self._server_running = False
                self._uvicorn_server = None
                try:
                    self.call_from_thread(status_widget.set_stopped)
                except Exception:
                    pass

        self._server_thread = threading.Thread(target=_run, daemon=True, name="ada-server")
        self._server_thread.start()

    def _stop_server(self) -> None:
        """Stop the running server."""
        if not self._server_running or not self._uvicorn_server:
            return
        self._uvicorn_server.should_exit = True

    def _refresh_dashboard(self) -> None:
        """Update all dashboard widgets with latest data."""
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
                self._show_message("system", f"Server is already running on port {self._server_port}.")
            else:
                self._start_server(port)
        elif cmd == "/stop":
            if not self._server_running:
                self._show_message("system", "Server is not running.")
            else:
                self._stop_server()
                self._show_message("system", "Server stopping...")
        elif cmd == "/status":
            if self._server_running:
                self._show_message("system", f"Server running on port {self._server_port}.")
            else:
                self._show_message("system", "Server is stopped.")
        elif cmd == "/logs":
            self.query_one(TabbedContent).active = "tab-logs"
        elif cmd == "/clear":
            if len(parts) > 1 and parts[1] == "logs":
                _tui_log_handler.clear()
                self._show_message("system", "Logs cleared.")
            else:
                # Clear output messages and switch to dashboard
                try:
                    scroll = self.query_one("#messages-scroll", VerticalScroll)
                    scroll.remove_children()
                except Exception:
                    pass
                self.query_one(TabbedContent).active = "tab-dashboard"
        elif cmd in ("/exit", "/quit"):
            self.exit()
        else:
            self._show_message("system", f"Unknown command: {cmd}. Try /help")

    def _show_message(self, role: str, content: str) -> None:
        """Show a message in the Output tab."""
        try:
            log = self.query_one("#messages-scroll", VerticalScroll)
        except Exception:
            return
        msg = Text()
        ts = time.strftime("%H:%M:%S")
        msg.append(f"{ts} ", style="#555555")
        if role == "system":
            msg.append("\u2022 ", style=f"bold {ADA_PURPLE}")
            msg.append(content, style=TEXT_DIM)
        else:
            msg.append(content, style=TEXT_PRIMARY)
        log.mount(Static(msg))
        # Switch to output tab to show the message
        self.query_one(TabbedContent).active = "tab-output"

    def _show_help(self) -> None:
        try:
            log = self.query_one("#messages-scroll", VerticalScroll)
        except Exception:
            return
        help_text = Text()
        help_text.append("Server:\n", style=f"bold {TEXT_PRIMARY}")
        help_text.append("  /start", style=f"bold {ADA_BLUE}")
        help_text.append(" [port]              \u2014 Start the server\n", style=TEXT_PRIMARY)
        help_text.append("  /stop", style=f"bold {ADA_BLUE}")
        help_text.append("                      \u2014 Stop the server\n", style=TEXT_PRIMARY)
        help_text.append("  /status", style=f"bold {ADA_BLUE}")
        help_text.append("                    \u2014 Server status\n", style=TEXT_PRIMARY)
        help_text.append("\nViews:\n", style=f"bold {TEXT_PRIMARY}")
        help_text.append("  /logs", style=f"bold {ADA_BLUE}")
        help_text.append("                      \u2014 Switch to logs tab\n", style=TEXT_PRIMARY)
        help_text.append("  /clear", style=f"bold {ADA_BLUE}")
        help_text.append("                     \u2014 Clear output, back to dashboard\n", style=TEXT_PRIMARY)
        help_text.append("  /clear logs", style=f"bold {ADA_BLUE}")
        help_text.append("                \u2014 Clear log output\n", style=TEXT_PRIMARY)
        help_text.append("\nGeneral:\n", style=f"bold {TEXT_PRIMARY}")
        help_text.append("  /exit", style=f"bold {ADA_BLUE}")
        help_text.append("                      \u2014 Quit Ada\n", style=TEXT_PRIMARY)
        help_text.append("  Ctrl+L", style=f"bold {ADA_BLUE}")
        help_text.append("                     \u2014 Refresh dashboard\n", style=TEXT_PRIMARY)
        help_text.append("  Ctrl+C", style=f"bold {ADA_BLUE}")
        help_text.append("                     \u2014 Quit", style=TEXT_PRIMARY)
        log.mount(Static(help_text))
        self.query_one(TabbedContent).active = "tab-output"

    def action_refresh_all(self) -> None:
        """Refresh all dashboard data."""
        self._refresh_dashboard()


def main() -> None:
    """Launch the Ada TUI dashboard."""
    app = AdaApp()
    app.run()


if __name__ == "__main__":
    main()
