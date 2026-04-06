"""Ada Runtime — terminal dashboard."""

from __future__ import annotations

import time
import threading

from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Static, Input, Footer, Header
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
            # Status indicator
            if m.status == "loaded":
                status = Text("\u25cf", style=TEXT_SUCCESS)
            elif m.status == "loading":
                status = Text("\u25cf", style=ADA_PURPLE)
            else:
                status = Text("\u25cf", style=TEXT_ERROR)

            # Last query relative time
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

    #dashboard {
        height: 1fr;
        padding: 0 0;
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
        Binding("ctrl+l", "clear", "Clear", show=True),
    ]

    def __init__(self, server_started: bool = False) -> None:
        super().__init__()
        self._server_started = server_started
        self._server_running = server_started
        self._server_thread: threading.Thread | None = None
        self._uvicorn_server = None
        self._refresh_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield BannerWidget()
        yield ServerStatus()
        yield SummaryBar()
        with VerticalScroll(id="dashboard"):
            with Horizontal(id="graphs"):
                yield RateGraph()
                yield LatencyGraph()
            yield ModelsTable()
            yield RecentRequests()
        yield Input(placeholder="Type a command... (/help)", id="prompt-input")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(Input).focus()

        # Show initial server status
        status_widget = self.query_one(ServerStatus)
        if self._server_started:
            from infrastructure.config import get_settings
            status_widget.set_running(get_settings().port)
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
        actual_port = port or settings.port

        status_widget = self.query_one(ServerStatus)
        status_widget.set_starting()

        config = uvicorn.Config(
            "ada.server:app",
            host=settings.host,
            port=actual_port,
            log_level="warning",
        )
        self._uvicorn_server = uvicorn.Server(config)

        def _run():
            try:
                self._server_running = True
                self.call_from_thread(status_widget.set_running, actual_port)
                self._uvicorn_server.run()
            except Exception as e:
                self.call_from_thread(status_widget.set_error, str(e))
            finally:
                self._server_running = False
                self.call_from_thread(status_widget.set_stopped)

        self._server_thread = threading.Thread(target=_run, daemon=True)
        self._server_thread.start()

    def _stop_server(self) -> None:
        """Stop the running server."""
        if not self._server_running or not self._uvicorn_server:
            return

        self._uvicorn_server.should_exit = True
        self._server_running = False

    def _refresh_dashboard(self) -> None:
        """Update all dashboard widgets with latest data."""
        if not self._server_running:
            return
        try:
            self.query_one(SummaryBar).refresh_data()
            self.query_one(RateGraph).refresh_data()
            self.query_one(LatencyGraph).refresh_data()
            self.query_one(ModelsTable).refresh_data()
            self.query_one(RecentRequests).refresh_data()
        except Exception:
            pass  # Widget not mounted yet

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
                self._show_message("system", "Server is already running.")
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
                self._show_message("system", "Server is running.")
            else:
                self._show_message("system", "Server is stopped.")
        else:
            self._show_message("system", f"Unknown command: {cmd}. Try /help")

    def _show_message(self, role: str, content: str) -> None:
        """Show a message in the dashboard area."""
        log = self.query_one("#dashboard", VerticalScroll)
        msg = Text()
        if role == "system":
            msg.append("\u2022 ", style=f"bold {ADA_PURPLE}")
            msg.append(content, style=TEXT_DIM)
        else:
            msg.append(content, style=TEXT_PRIMARY)
        log.mount(Static(msg))

    def _show_help(self) -> None:
        log = self.query_one("#dashboard", VerticalScroll)
        help_text = Text()
        help_text.append("Commands:\n", style=f"bold {TEXT_PRIMARY}")
        help_text.append("  /start", style=f"bold {ADA_BLUE}")
        help_text.append(" [port]              \u2014 Start the server\n", style=TEXT_PRIMARY)
        help_text.append("  /stop", style=f"bold {ADA_BLUE}")
        help_text.append("                      \u2014 Stop the server\n", style=TEXT_PRIMARY)
        help_text.append("  /status", style=f"bold {ADA_BLUE}")
        help_text.append("                    \u2014 Show server status\n", style=TEXT_PRIMARY)
        help_text.append("  /help", style=f"bold {ADA_BLUE}")
        help_text.append("                      \u2014 Show this help\n", style=TEXT_PRIMARY)
        help_text.append("  Ctrl+L", style=f"bold {ADA_BLUE}")
        help_text.append("                     \u2014 Refresh dashboard\n", style=TEXT_PRIMARY)
        help_text.append("  Ctrl+C", style=f"bold {ADA_BLUE}")
        help_text.append("                     \u2014 Quit", style=TEXT_PRIMARY)
        log.mount(Static(help_text))

    def action_clear(self) -> None:
        """Refresh all dashboard data."""
        self._refresh_dashboard()


def main() -> None:
    """Launch the Ada TUI dashboard."""
    app = AdaApp()
    app.run()


if __name__ == "__main__":
    main()
