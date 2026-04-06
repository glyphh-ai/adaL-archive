"""
Ada CLI — entry point for TUI dashboard and headless server.

Usage:
    ada              Start the TUI dashboard (server runs in background thread)
    ada serve        Start the server headless (no TUI)
    ada serve -p 9000  Custom port
"""

import sys


def main() -> None:
    args = sys.argv[1:]

    if args and args[0] == "serve":
        _serve(args[1:])
    else:
        _dashboard()


def _dashboard() -> None:
    """Launch the TUI dashboard."""
    from ada.tui.app import AdaApp

    app = AdaApp()
    app.run()


def _serve(args: list[str]) -> None:
    """Run the server headless (no TUI)."""
    import uvicorn
    from infrastructure.config import get_settings

    settings = get_settings()

    # Parse --port / -p flag
    port = settings.port
    host = settings.host
    for i, arg in enumerate(args):
        if arg in ("-p", "--port") and i + 1 < len(args):
            port = int(args[i + 1])
        elif arg in ("-h", "--host") and i + 1 < len(args):
            host = args[i + 1]

    print(f"Ada Runtime — serving on {host}:{port}")
    uvicorn.run(
        "ada.server:app",
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
