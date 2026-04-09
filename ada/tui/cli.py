"""
Ada CLI — interactive shell + headless server.

Usage:
    ada              Launch interactive shell
    ada serve        Start headless (MCP endpoint only)
    ada serve -p 9000  Custom port
    ada setup        Show how to connect to Claude/ChatGPT
    ada setup claude Auto-configure Claude Code
"""

import sys
import threading
import time

# ANSI colors
P = "\033[38;5;141m"   # purple
B = "\033[38;5;111m"   # blue
G = "\033[38;5;114m"   # green
D = "\033[38;5;244m"   # dim
W = "\033[37m"         # white
R = "\033[0m"          # reset
BOLD = "\033[1m"
PINK = "\033[38;5;204m"
CYAN = "\033[38;5;116m"

# Banner gradient
_GRAD = ["\033[38;5;204m", "\033[38;5;177m", "\033[38;5;141m", "\033[38;5;111m"]
_BANNER = [
    "⣼⢻⡄ ⣿⠛⠛⢻⡄ ⣼⢻⡄",
    "⣼⠃ ⢻⡄⣿  ⢸⡇⣼⠃ ⢻⡄",
    "⣿⠛⠛⢻⡇⣿  ⢸⡇⣿⠛⠛⢻⡇",
    "⠛  ⠘⠃⠛⠛⠛⠛ ⠛  ⠘⠃",
]


def _print_banner(version: str = "0.2.0") -> None:
    for i, line in enumerate(_BANNER):
        c = _GRAD[i] if i < len(_GRAD) else _GRAD[-1]
        print(f"  {c}{BOLD}{line}{R}")
    print(f"  {D}v{version}{R}")
    print()


def _print_config(host: str, port: int) -> None:
    import os
    display_host = "localhost" if host == "0.0.0.0" else host
    model = os.environ.get("ADA_MODEL", "claude-haiku-4-5-20251001")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

    print(f"  {D}MCP{R}    http://{display_host}:{port}/mcp")
    print(f"  {D}Tool{R}   think(input)")
    print(f"  {D}LLM{R}    {model}")
    if has_key:
        print(f"  {D}Key{R}    {G}●{R} set")
    else:
        print(f"  {D}Key{R}    {PINK}○{R} not set {D}(run{R} setup key{D}){R}")
    print()


def _print_capabilities(brain) -> None:
    if not brain:
        return
    caps = brain.capabilities
    router_name = None
    if brain._state.router:
        router_name = brain._state.router.name

    if router_name:
        print(f"  {D}router{R} {G}{router_name}{R}")
    for name in caps:
        print(f"  {D}  ├─{R}   {P}{name}{R}")
    print()


def _print_help() -> None:
    print(f"""
  {BOLD}Commands{R}

  {B}help{R}           Show this message
  {B}setup{R}          How to connect Ada to an LLM
  {B}setup claude{R}   Auto-configure Claude Code
  {B}setup key{R}      Set Anthropic API key
  {B}setup model{R}    Change Ada's internal LLM model
  {B}status{R}         Show brain status
  {B}config{R}         Show current configuration
  {B}exit{R}           Quit
""")


def _print_status(brain) -> None:
    if not brain:
        print(f"  {D}Brain not loaded yet{R}")
        return

    caps = brain.capabilities
    obs = brain.observations
    llm = brain.llm

    print(f"\n  {BOLD}Brain{R}")
    print(f"  {D}Capabilities:{R}  {len(caps)} loaded")
    print(f"  {D}Observations:{R}  {obs.count} total, {obs.llm_fallback_count} LLM fallbacks")
    print(f"  {D}Thoughts:{R}      {brain.cognitive.thought_space.count}")

    if llm.available:
        print(f"  {D}LLM:{R}           {G}online{R} ({llm.usage.calls} calls, {llm.usage.input_tokens + llm.usage.output_tokens} tokens)")
    else:
        print(f"  {D}LLM:{R}           {D}offline (deterministic only){R}")
    print()


def _print_setup(port: int) -> None:
    url = f"http://localhost:{port}/mcp"
    print(f"""
  {P}{BOLD}Connect Ada to your LLM{R}

  {B}{BOLD}Claude Code{R} {G}(one command){R}
  {W}ada setup claude{R}

  {B}{BOLD}Claude Desktop{R}
  {D}Add to ~/Library/Application Support/Claude/claude_desktop_config.json:{R}
  {W}{{"mcpServers": {{"ada": {{"transport": "http", "url": "{url}"}}}}}}{R}

  {B}{BOLD}Cursor / VS Code{R}
  {D}Add to MCP settings:{R}
  {W}{{"ada": {{"transport": "http", "url": "{url}"}}}}{R}

  {B}{BOLD}Any MCP Client{R}
  {W}Endpoint: {url}{R}
  {W}Tool: think(input: string){R}
""")


def _get_ada_home() -> str:
    """~/.ada/ — Ada's config directory."""
    import os
    from pathlib import Path
    home = Path.home() / ".ada"
    home.mkdir(exist_ok=True)
    return str(home)


def _get_env_path() -> str:
    """Path to Ada's .env file."""
    return f"{_get_ada_home()}/env"


def _load_ada_env() -> None:
    """Load Ada's stored env vars from ~/.ada/env into os.environ."""
    import os
    env_path = _get_env_path()
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def _setup_key() -> None:
    """Set or update the Anthropic API key."""
    import os
    import getpass

    env_path = _get_env_path()
    current = os.environ.get("ANTHROPIC_API_KEY", "")

    if current:
        masked = current[:8] + "..." + current[-4:]
        print(f"\n  {D}Current key:{R} {masked}")
        print(f"  {D}Enter new key or press Enter to keep:{R}")
    else:
        print(f"\n  {D}No API key set. Ada needs this for her internal LLM.{R}")
        print(f"  {D}Get one at:{R} {B}https://console.anthropic.com/settings/keys{R}")
        print()

    try:
        key = getpass.getpass(f"  {P}API key{D}>{R} ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if not key:
        if current:
            print(f"  {D}Keeping existing key.{R}\n")
        else:
            print(f"  {D}No key set.{R}\n")
        return

    if not key.startswith("sk-"):
        print(f"  {PINK}Doesn't look like an Anthropic key (should start with sk-){R}\n")
        return

    # Write to ~/.ada/env
    lines = []
    if os.path.exists(env_path):
        with open(env_path) as f:
            lines = [l for l in f.readlines() if not l.strip().startswith("ANTHROPIC_API_KEY=")]

    lines.append(f"ANTHROPIC_API_KEY={key}\n")
    with open(env_path, "w") as f:
        f.writelines(lines)
    os.chmod(env_path, 0o600)  # Owner-only read/write

    os.environ["ANTHROPIC_API_KEY"] = key
    print(f"  {G}Key saved to ~/.ada/env{R}")
    print(f"  {D}Restart Ada to activate.{R}\n")


def _setup_model() -> None:
    """Change which LLM model Ada uses internally."""
    import os

    env_path = _get_env_path()
    current = os.environ.get("ADA_MODEL", "claude-haiku-4-5-20251001")

    models = [
        ("claude-haiku-4-5-20251001", "Haiku 4.5", f"{G}fast, cheap — recommended{R}"),
        ("claude-sonnet-4-6", "Sonnet 4.6", f"{B}balanced{R}"),
        ("claude-opus-4-6", "Opus 4.6", f"{P}powerful, expensive{R}"),
    ]

    print(f"\n  {BOLD}Ada's internal LLM{R}")
    print(f"  {D}Current:{R} {current}\n")

    for i, (model_id, name, desc) in enumerate(models, 1):
        marker = f" {G}←{R}" if model_id == current else ""
        print(f"  {B}{i}{R}  {W}{name}{R} {D}({model_id}){R} {desc}{marker}")

    print(f"\n  {D}Enter number or model ID (Enter to keep current):{R}")

    try:
        choice = input(f"  {P}model{D}>{R} ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if not choice:
        print(f"  {D}Keeping {current}{R}\n")
        return

    # Resolve choice
    model_id = None
    if choice.isdigit() and 1 <= int(choice) <= len(models):
        model_id = models[int(choice) - 1][0]
    elif choice.startswith("claude-"):
        model_id = choice
    else:
        print(f"  {PINK}Invalid choice.{R}\n")
        return

    # Write to ~/.ada/env
    lines = []
    if os.path.exists(env_path):
        with open(env_path) as f:
            lines = [l for l in f.readlines() if not l.strip().startswith("ADA_MODEL=")]

    lines.append(f"ADA_MODEL={model_id}\n")
    with open(env_path, "w") as f:
        f.writelines(lines)

    os.environ["ADA_MODEL"] = model_id
    print(f"  {G}Model set to {model_id}{R}")
    print(f"  {D}Restart Ada to activate.{R}\n")


def _setup_claude_code(port: int) -> None:
    import subprocess
    import shutil

    url = f"http://localhost:{port}/mcp"

    if not shutil.which("claude"):
        print(f"\n  {D}claude CLI not found. Install it first:{R}")
        print(f"  {W}npm install -g @anthropic-ai/claude-code{R}\n")
        return

    try:
        result = subprocess.run(
            ["claude", "mcp", "add", "ada", "--transport", "http", url],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            print(f"\n  {G}{BOLD}Done.{R} Ada is connected to Claude Code.\n")
        elif "already exists" in (result.stderr or "").lower():
            print(f"\n  {G}Already configured.{R}\n")
        else:
            print(f"\n  {D}Try manually: claude mcp add ada --transport http {url}{R}\n")
    except Exception as e:
        print(f"\n  {D}Error: {e}{R}")
        print(f"  {D}Try manually: claude mcp add ada --transport http {url}{R}\n")


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

def _start_server_background(host: str, port: int) -> threading.Thread:
    """Start Ada's server in a background thread."""
    import uvicorn

    def _run():
        uvicorn.run(
            "ada.server:app",
            host=host,
            port=port,
            log_level="error",
        )

    t = threading.Thread(target=_run, daemon=True, name="ada-server")
    t.start()
    return t


def _wait_for_server(host: str, port: int, timeout: float = 30.0) -> bool:
    """Wait for the server to be ready by polling the health endpoint."""
    import socket

    check_host = "127.0.0.1" if host == "0.0.0.0" else host
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            # First check if port is open (fast)
            with socket.create_connection((check_host, port), timeout=1):
                pass
            # Then check health endpoint
            import httpx
            r = httpx.get(f"http://{check_host}:{port}/health", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _get_brain():
    """Get the brain instance from the running server."""
    try:
        from ada.server import brain
        return brain
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Interactive shell
# ---------------------------------------------------------------------------

def _shell(host: str, port: int) -> None:
    """Interactive Ada shell."""
    _print_banner()

    # Suppress all logging — this is a clean REPL
    import logging
    logging.disable(logging.CRITICAL)
    # Also kill any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.root.addHandler(logging.NullHandler())

    print(f"  {D}Starting...{R}", end="", flush=True)
    _start_server_background(host, port)

    if _wait_for_server(host, port):
        print(f"\r  {G}● Online{R}   ")
    else:
        print(f"\r  {PINK}● Failed to start{R}   ")
        return

    _print_config(host, port)

    brain = _get_brain()
    _print_capabilities(brain)

    print(f"  {D}Type{R} help {D}for commands,{R} setup {D}to connect an LLM{R}")
    print()

    while True:
        try:
            line = input(f"  {P}ada{D}>{R} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        if cmd in ("exit", "quit", "q"):
            break
        elif cmd == "help":
            _print_help()
        elif cmd == "setup":
            if len(parts) > 1 and parts[1] in ("claude", "claude-code"):
                _setup_claude_code(port)
            elif len(parts) > 1 and parts[1] == "key":
                _setup_key()
            elif len(parts) > 1 and parts[1] == "model":
                _setup_model()
            else:
                _print_setup(port)
        elif cmd == "status":
            brain = _get_brain()
            _print_status(brain)
        elif cmd == "config":
            _print_config(host, port)
        else:
            print(f"  {D}Unknown command: {cmd}. Type{R} help{D} for options.{R}")

    print(f"  {D}Goodbye.{R}")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def _serve(args: list[str]) -> None:
    """Run Ada headless (no shell, MCP endpoint only)."""
    import uvicorn
    from infrastructure.config import get_settings

    settings = get_settings()

    port = settings.port
    host = settings.host
    for i, arg in enumerate(args):
        if arg in ("-p", "--port") and i + 1 < len(args):
            port = int(args[i + 1])
        elif arg in ("-h", "--host") and i + 1 < len(args):
            host = args[i + 1]

    _print_banner()
    _print_config(host, port)

    uvicorn.run(
        "ada.server:app",
        host=host,
        port=port,
        log_level="info",
    )


def main() -> None:
    from infrastructure.config import get_settings
    settings = get_settings()

    args = sys.argv[1:]

    # Parse global flags
    host = settings.host
    port = settings.port
    for i, arg in enumerate(args):
        if arg in ("-p", "--port") and i + 1 < len(args):
            port = int(args[i + 1])
        elif arg in ("-h", "--host") and i + 1 < len(args):
            host = args[i + 1]

    # Load stored env vars (API key, model, etc.)
    _load_ada_env()

    if not args:
        _shell(host, port)
    elif args[0] == "serve":
        _load_ada_env()  # also for headless
        _serve(args[1:])
    elif args[0] == "setup":
        if len(args) > 1 and args[1] in ("claude", "claude-code"):
            _setup_claude_code(port)
        elif len(args) > 1 and args[1] == "key":
            _setup_key()
        elif len(args) > 1 and args[1] == "model":
            _setup_model()
        else:
            _print_setup(port)
    elif args[0] == "help":
        _print_banner()
        _print_help()
    else:
        print(f"Unknown command: {args[0]}")
        print("Usage: ada [serve|setup|help]")
        sys.exit(1)


if __name__ == "__main__":
    main()
