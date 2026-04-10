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
    " ⣼⢻⡄ ⣿⠛⠛⢻⡄ ⣼⢻⡄",
    "⣼⠃ ⢻⡄⣿  ⢸⡇⣼⠃ ⢻⡄",
    "⣿⠛⠛⢻⡇⣿  ⢸⡇⣿⠛⠛⢻⡇",
    "⠛  ⠘⠃⠛⠛⠛⠛ ⠛  ⠘⠃",
]


def _print_banner(version: str = "0.2.0") -> None:
    width = max(len(line) for line in _BANNER)
    for i, line in enumerate(_BANNER):
        c = _GRAD[i] if i < len(_GRAD) else _GRAD[-1]
        print(f"  {c}{BOLD}{line}{R}")
    ver = f"v{version}"
    padding = width - len(ver)
    print(f"  {' ' * padding}{D}{ver}{R}")
    print()


class _BrailleSpinner:
    """Braille dot spinner: ⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"""
    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, message: str = "thinking..."):
        self._message = message
        self._running = False
        self._thread = None
        # Grab a direct handle to the terminal — survives logging shenanigans
        import io, os
        self._tty = io.TextIOWrapper(
            io.FileIO(os.dup(sys.stderr.fileno()), mode="w"),
            encoding="utf-8",
        )

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        self._tty.write("\r\033[2K")
        self._tty.flush()

    def _spin(self):
        i = 0
        while self._running:
            frame = self._FRAMES[i % len(self._FRAMES)]
            self._tty.write(f"\r  \033[38;5;141m{frame}\033[0m \033[38;5;244m{self._message}\033[0m")
            self._tty.flush()
            i += 1
            time.sleep(0.08)


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


def _require_api_key() -> bool:
    """Check for API key. If missing, prompt to set it. Returns True if key is set."""
    import os
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True

    print(f"\n  {PINK}Ada needs an Anthropic API key to start.{R}")
    print(f"  {D}Get one at:{R} {B}https://console.anthropic.com/settings/keys{R}\n")
    _setup_key(quiet=True)

    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _print_help() -> None:
    print(f"""
  {D}Just type anything to talk to Ada.{R}

  {BOLD}Commands{R}

  {B}help{R}           Show this message
  {B}setup{R}          How to connect Ada to an LLM
  {B}setup claude{R}   Auto-configure Claude Code
  {B}setup key{R}      Set Anthropic API key
  {B}setup model{R}    Change Ada's internal LLM model
  {B}setup license{R}   Install license file from glyphh.ai
  {B}setup pipedream{R} Connect to Pipedream (3000+ APIs)
  {B}connect{R} <app>   Authenticate an app (slack, gmail, etc.)
  {B}connect{R}         List connected apps
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
    """Load Ada's secrets from vault + env file into os.environ."""
    import os

    # Load vault secrets first
    try:
        from domains.brain.skills.vault import get_vault
        vault = get_vault()

        # Map vault keys to env vars
        _VAULT_TO_ENV = {
            "anthropic_api_key": "ANTHROPIC_API_KEY",
            "pipedream_client_id": "PIPEDREAM_CLIENT_ID",
            "pipedream_client_secret": "PIPEDREAM_CLIENT_SECRET",
            "pipedream_project_id": "PIPEDREAM_PROJECT_ID",
        }
        for vault_key, env_key in _VAULT_TO_ENV.items():
            val = vault.get(vault_key)
            if val:
                os.environ.setdefault(env_key, val)
    except Exception:
        pass

    # Then load ~/.ada/env (lower priority — vault wins)
    env_path = _get_env_path()
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def _setup_key(quiet: bool = False) -> None:
    """Set or update the Anthropic API key."""
    import os
    import getpass
    from domains.brain.skills.vault import get_vault

    vault = get_vault()
    current = os.environ.get("ANTHROPIC_API_KEY") or vault.get("anthropic_api_key") or ""

    if current:
        masked = current[:8] + "..." + current[-4:]
        print(f"\n  {D}Current key:{R} {masked}")
        print(f"  {D}Enter new key or press Enter to keep:{R}")
    elif not quiet:
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

    # Store encrypted in vault + set in env for current session
    vault.set("anthropic_api_key", key)
    os.environ["ANTHROPIC_API_KEY"] = key
    print(f"  {G}Key encrypted and saved.{R}")
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


def _setup_pipedream() -> None:
    """Set up Pipedream Connect credentials."""
    from domains.brain.skills.vault import get_vault

    vault = get_vault()

    print(f"\n  {P}{BOLD}Pipedream Connect{R}")
    print(f"  {D}This connects Ada to 3,000+ APIs.{R}")
    print(f"  {D}Get your project credentials at:{R} {B}https://pipedream.com/connect{R}\n")

    if vault.has("pipedream_project_id"):
        print(f"  {D}Already configured. Enter new values to update, or press Enter to keep.{R}\n")

    try:
        client_id = input(f"  {P}Client ID{D}>{R} ").strip()
        client_secret = input(f"  {P}Client Secret{D}>{R} ").strip()
        project_id = input(f"  {P}Project ID{D}>{R} ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if not all([client_id, client_secret, project_id]):
        if vault.has("pipedream_project_id"):
            print(f"  {D}Keeping existing config.{R}\n")
            return
        print(f"  {PINK}All three values required.{R}\n")
        return

    # Store encrypted in vault
    vault.set("pipedream_client_id", client_id)
    vault.set("pipedream_client_secret", client_secret)
    vault.set("pipedream_project_id", project_id)

    print(f"  {G}Pipedream credentials encrypted and saved.{R}")
    print(f"  {D}Now run{R} connect slack {D}to authenticate your first app.{R}\n")


def _connect_app(app_slug: str) -> None:
    """Generate a Pipedream Connect URL for the user to authenticate an app."""
    from domains.brain.skills.pipedream_connect import PipedreamConnect

    pd = PipedreamConnect()
    if not pd.configured:
        print(f"\n  {PINK}Pipedream not configured.{R} Run: {W}setup pipedream{R}\n")
        return

    url = pd.get_connect_url(app_slug)
    if url:
        print(f"\n  {D}Open this URL to connect {B}{app_slug}{D}:{R}")
        print(f"  {W}{url}{R}")
        print(f"\n  {D}After authenticating, {app_slug} is ready to use.{R}\n")

        # Try to open in browser
        try:
            import webbrowser
            webbrowser.open(url)
            print(f"  {D}(Opened in your browser){R}\n")
        except Exception:
            pass
    else:
        print(f"\n  {PINK}Failed to generate connect URL.{R}\n")


def _list_connected_apps() -> None:
    """List apps the user has connected via Pipedream."""
    from domains.brain.skills.pipedream_connect import PipedreamConnect

    pd = PipedreamConnect()
    if not pd.configured:
        print(f"\n  {PINK}Pipedream not configured.{R} Run: {W}setup pipedream{R}\n")
        return

    apps = pd.list_connected_apps()
    if apps:
        print(f"\n  {BOLD}Connected apps{R}")
        for app in apps:
            name = app.get("name", app.get("app_slug", "unknown"))
            print(f"  {G}●{R} {name}")
        print()
    else:
        print(f"\n  {D}No apps connected yet.{R} Run: {W}connect slack{R}\n")


def _setup_license(path: str) -> None:
    """Install a license file."""
    from domains.brain.skills.license import install_license

    status = install_license(path)
    if status.valid:
        print(f"  {G}License installed: {status.message}{R}\n")
    else:
        print(f"  {PINK}{status.message}{R}\n")


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

_server_error: str | None = None


def _start_server_background(host: str, port: int) -> threading.Thread:
    """Start Ada's server in a background thread."""
    import uvicorn

    def _run():
        global _server_error
        try:
            uvicorn.run(
                "ada.server:app",
                host=host,
                port=port,
                log_level="error",
            )
        except Exception as e:
            _server_error = str(e)

    t = threading.Thread(target=_run, daemon=True, name="ada-server")
    t.start()
    return t


def _wait_for_server(host: str, port: int, timeout: float = 300.0) -> bool:
    """Wait for the server to be ready. First boot can take minutes (encoding exemplars)."""
    import socket

    check_host = "127.0.0.1" if host == "0.0.0.0" else host
    start = time.time()
    dots = 0
    while time.time() - start < timeout:
        try:
            with socket.create_connection((check_host, port), timeout=1):
                pass
            import httpx
            r = httpx.get(f"http://{check_host}:{port}/health", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass

        # Show progress dots so user knows it's working
        elapsed = int(time.time() - start)
        if elapsed > 5 and elapsed % 5 == 0 and elapsed // 5 > dots:
            dots = elapsed // 5
            sys.stdout.write(".")
            sys.stdout.flush()

        time.sleep(1.0)
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

def _check_license_or_exit() -> bool:
    """Check license. Returns True if valid."""
    from domains.brain.skills.license import check_license

    status = check_license()

    if status.valid:
        if status.plan == "trial":
            print(f"  {D}Trial:{R} {status.days_remaining} days remaining")
        else:
            print(f"  {G}●{R} {status.plan} license")
        return True
    else:
        print(f"\n  {PINK}{status.message}{R}")
        print(f"  {D}Purchase at:{R} {B}https://glyphh.ai{R}")
        print(f"  {D}Then run:{R} {W}ada setup license <path-to-file>{R}\n")
        return False


def _shell(host: str, port: int) -> None:
    """Interactive Ada shell."""
    _print_banner()

    # Check license
    if not _check_license_or_exit():
        return

    # Require API key before starting
    if not _require_api_key():
        print(f"  {D}Cannot start without an API key.{R}\n")
        return

    # Suppress Ada's logging — keep uvicorn functional
    import logging
    for name in ["ada", "domains", "infrastructure", "shared", "api", "mcp"]:
        logging.getLogger(name).setLevel(logging.CRITICAL)
    # Suppress root handler output (structured JSON logs)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.root.addHandler(logging.NullHandler())

    print(f"  {D}Starting...{R}", end="", flush=True)
    _start_server_background(host, port)

    if _wait_for_server(host, port):
        print(f"\r  {G}● Online{R}   ")
    else:
        print(f"\r  {PINK}● Failed to start{R}   ")
        if _server_error:
            print(f"  {PINK}{_server_error}{R}")
        else:
            print(f"  {D}Try: ada serve (to see full error output){R}")
        return

    _print_config(host, port)

    print(f"  {D}Type{R} help {D}for commands,{R} setup {D}to connect an LLM{R}")
    print()

    while True:
        try:
            line = input(f"  \x01{P}\x02ada\x01{D}\x02>\x01{R}\x02 ").strip()
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
            elif len(parts) > 1 and parts[1] == "pipedream":
                _setup_pipedream()
            elif len(parts) > 1 and parts[1] == "license":
                if len(parts) > 2:
                    _setup_license(parts[2])
                else:
                    print(f"  {D}Usage:{R} setup license <path-to-license-file>\n")
            else:
                _print_setup(port)
        elif cmd == "connect":
            # connect <app> — authenticate an app via Pipedream
            if len(parts) > 1:
                _connect_app(parts[1])
            else:
                _list_connected_apps()
        elif cmd == "status":
            brain = _get_brain()
            _print_status(brain)
        elif cmd == "config":
            _print_config(host, port)
        else:
            # Everything else goes to Ada's brain
            brain = _get_brain()
            if not brain:
                print(f"  {D}Brain not ready yet.{R}")
                continue

            spinner = _BrailleSpinner()
            spinner.start()
            try:
                # Run think() in a new thread since the server owns the event loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    import asyncio
                    future = pool.submit(lambda: asyncio.run(brain.think(line)))
                    result = future.result(timeout=30)
                spinner.stop()
                print(f"  {CYAN}{result.response}{R}")
                tags = []
                if result.capability:
                    tags.append(f"{result.capability}")
                    tags.append(f"{result.confidence:.0%}")
                if result.llm_fallback:
                    tags.append("llm")
                tags.append(f"{result.elapsed_ms:.0f}ms")
                print(f"  {D}[{' · '.join(tags)}]{R}")
            except Exception as e:
                spinner.stop()
                print(f"  {PINK}Error: {e}{R}")
            print()

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

    if not _check_license_or_exit():
        return

    if not _require_api_key():
        print(f"  {D}Cannot start without an API key.{R}\n")
        return

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
        elif len(args) > 1 and args[1] == "license":
            if len(args) > 2:
                _setup_license(args[2])
            else:
                print(f"Usage: ada setup license <path-to-license-file>")
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
