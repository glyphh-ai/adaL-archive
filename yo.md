# Yo

**One word. Every AI tool. Shared context. Prompt security.**

## The Problem

Developers bounce between 5+ AI tools daily — Claude Code, Gemini, OpenAI, Cowork, Grok. Each one is a silo. No session crosses between them. The developer is the glue — copy-pasting context, re-explaining, losing flow.

## The Solution

Ada is the shared brain across all your AI tools. "Yo" is how you talk to it.

From anywhere — terminal, browser, any AI tool — you say:

- **"yo code"** → Claude Code slides in, session-aware
- **"yo gemini"** → Gemini slides in, context-rich
- **"yo openai"** → OpenAI slides in, knows everything
- **"yo ada"** → Ada herself slides out, her TUI running on your box
- **"see ya [tool]"** → slides away, nothing lost

Every tool has Ada wired in via MCP. Every prompt runs through `ada.think()`. Context flows. Tools come and go. Ada remembers everything.

## How It Works

Every "yo" triggers the same pipeline:

```
user prompt
  → ada.think()
    → FIREWALL (HDC prompt injection check, microseconds, no LLM)
    → CLASSIFY (RECALL, STORE, FEEL, WONDER...)
    → CONTEXT (session thought glyphs loaded)
    → ROUTE (right capability)
  → tool receives clean, enriched prompt
```

The tools never talk to each other. They all just call `ada.think()` and get session context back. Ada is the invisible bridge.

## Architecture

```
Your machine                          Cloud
┌──────────────┐                ┌──────────────┐
│ Ada (local)  │───pushes───▶   │ glyphh API   │
│              │  context       │ (platform)   │
│ Claude Code  │  (glyphs,      │              │
│ your code    │   not code)    │ session      │
│ think (MCP)  │                │ context      │
│              │◀──pulls────    │              │
└──────────────┘  updates       └──────┬───────┘
                                       │
                                ┌──────┴───────┐
                                │ Browser      │
                                │ yo.ai        │
                                │ Gemini/OpenAI│
                                │ slide-outs   │
                                └──────────────┘
```

- **Ada runs local** — your code stays on your machine
- **Glyphs go up, code never does** — HDC vectors are lossy, semantic representations. Context without content. Privacy by architecture.
- **Web tools proxy through glyphh-platform** to reach local Ada
- **Terminal tools (Claude Code)** hit Ada directly via MCP

## The UX

There is no home screen. Ada is everywhere. Wherever you are, you're wired in.

**In Claude Code terminal:**
```
> yo gemini build me a deck on this
(ada.think() → context flows → gemini does it)
```

**In Gemini browser:**
```
> yo code fix that bug we talked about
(ada.think() → context flows → code slides in)
```

**In Cowork:**
```
> yo ada what did I work on today?
(ada.think() → "you worked on a, b, c")
```

**In glyphh.ai:**
```
> yo code open session c
(ada.think() → code slides in with session loaded)
```

Tools slide in and out. Context compounds. The session gets richer with every interaction.

## What Ada Already Has

- **`think` MCP tool** — single entry point, 7-step cognitive loop
- **Firewall** — 200+ prompt injection patterns, HDC similarity (microseconds)
- **ThoughtGlyphSpace** — persistent memory with Hebbian reinforcement
- **Cognitive gradient** — multi-step reasoning through HDC space
- **Dream loop** — background pattern discovery (REM + slow-wave)
- **Session isolation** — per-session runtimes
- **Claude Haiku** — internal LLM for disambiguation
- **Capability system** — extensible routing (code, voice, custom)

## What Needs to Be Built

### Browser layer (yo.ai)
- **Slide-out panels** — "yo [tool]" / "see ya [tool]" UX
- **xterm.js** — terminal emulator for Claude Code in the browser
- **LLM API integrations** — Gemini, OpenAI, Meta, Grok driven via APIs
- **WebSocket bridge** — browser ↔ Ada local runtime via platform proxy

### Ada additions
- **PTY bridge** — WebSocket endpoint for hosting Claude Code sessions
- **Context sync** — push thought glyphs to platform API for web access
- **"yo" routing** — HDC-based tool router for yo commands

### Platform (glyphh)
- **Context proxy** — relay between browser and local Ada
- **Auth/billing** — per-user, per-session
- **Team sessions** — shared context across developers

## The Moat

No LLM vendor will build this. Anthropic won't help Gemini. Google won't help Claude. Only a third party can own the space between them.

Ada is that third party. The session layer nobody else can build.

## Go-to-Market

1. **Free: local Ada** — `pip install ada-runtime && ada setup claude`. Your Claude Code is smarter. That's the hook.
2. **Aha moment** — "yo gemini build a deck on what I just coded." Gemini knows everything without you explaining. Holy shit.
3. **Viral** — the output is visible to others (Slack posts, decks, PRs). "How did you do that?" "Yo."
4. **Paid: hosted** — team sessions, persistent memory, cross-device, yo.ai web UI.

## The Pitch

"Your tools finally know about each other."
