"""
Ada — a self-evolving cognitive brain for LLMs.

Single FastAPI server exposing one MCP endpoint (/mcp) with one tool (think).
Ada loads her innate capabilities at startup, owns her own LLM (Haiku),
and grows new capabilities over time via the dream loop.

Importable as ``ada.server:app``.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from infrastructure.config import get_settings, validate_settings
from infrastructure.database import init_db, close_db, async_session_maker
from shared.exceptions import AdaRuntimeException
from shared.middleware import (
    CorrelationIDMiddleware,
    LoggingMiddleware,
)
from domains.models.manager import ModelManager

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

settings = get_settings()
_start_time = datetime.utcnow()

# ── Global brain state ───────────────────────────────────────────────────────

model_manager: Optional[ModelManager] = None
brain: Optional[object] = None  # domains.brain.think.Brain — set in lifespan


def get_model_manager() -> ModelManager:
    return model_manager


def get_brain():
    return brain


# ── Capabilities directory ───────────────────────────────────────────────────

CAPABILITIES_DIR = Path(__file__).resolve().parent.parent / "capabilities"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Boot Ada's brain."""
    global model_manager, brain

    logger.info("Waking up...")

    # Validate config
    try:
        validate_settings()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise

    # Database
    await init_db()
    logger.info("Memory banks online")

    # Model manager (internal — used by capabilities for DB persistence)
    model_manager = ModelManager(async_session_maker)

    # ── Load innate capabilities ─────────────────────────────────────────
    from domains.brain.loader import CapabilityLoader, register_capabilities_in_db

    brain_state = CapabilityLoader.boot(CAPABILITIES_DIR)
    app.state.brain_state = brain_state

    # Register capabilities in DB (loads exemplars, creates encoders)
    await register_capabilities_in_db(brain_state, model_manager)

    # ── Initialize Ada's LLM ────────────────────────────────────────────
    from domains.brain.llm import AdaLLM

    llm = AdaLLM()
    app.state.llm = llm

    # ── Initialize the think pipeline ───────────────────────────────────
    from domains.brain.think import Brain

    brain = Brain(
        brain_state=brain_state,
        model_manager=model_manager,
        llm=llm,
        session_factory=async_session_maker,
    )
    app.state.brain = brain
    logger.info("Think pipeline online")

    # ── Load persistent memories ────────────────────────────────────────
    from ada.memory.thought_persistence import load_thoughts
    loaded = await load_thoughts(async_session_maker, brain.cognitive.thought_space)
    if loaded:
        logger.info(f"Restored {loaded} memories from database")

    # ── Initialize MCP ──────────────────────────────────────────────────
    from domains.auth.service import AuthService
    from domains.mcp.app import create_mcp_session_managers

    auth_service = AuthService()
    json_manager, sse_manager = create_mcp_session_managers(brain, auth_service)
    app.state.mcp_session_managers = (json_manager, sse_manager)
    logger.info("MCP endpoint online at /mcp")

    # ── Resume any incomplete encoding from previous boot ─────────────
    try:
        await model_manager.resume_staged_encoding()
    except Exception as e:
        logger.warning(f"Staged encoding resume: {e}")

    # ── Wait for ALL encoding to finish before accepting requests ────
    # Ada doesn't serve until every capability (especially firewall)
    # has all its exemplars loaded. First boot can take minutes.
    import asyncio
    if model_manager._encoding_in_progress:
        logger.info(
            f"Waiting for exemplar encoding to finish: "
            f"{[k[1] for k in model_manager._encoding_in_progress]}"
        )
        while model_manager._encoding_in_progress:
            await asyncio.sleep(1.0)
        logger.info("All exemplars encoded")

    # ── Start dream loop ────────────────────────────────────────────────
    brain.start_dreaming()
    logger.info("Dream loop active")

    logger.info("Ada is awake.")

    # Run MCP session managers
    async with json_manager.run():
        async with sse_manager.run():
            yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("Going to sleep...")
    brain.stop_dreaming()

    # Flush memory strengths to DB
    from ada.memory.thought_persistence import flush_all_strengths
    await flush_all_strengths(async_session_maker, brain.cognitive.thought_space)

    await close_db()
    logger.info("Ada is asleep.")


# Create FastAPI application
app = FastAPI(
    title="Ada",
    description="A self-evolving cognitive brain for LLMs",
    version="0.1.0",
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url="/redoc" if settings.enable_docs else None,
    lifespan=lifespan,
)

# CORS middleware
if settings.cors_allow_all:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    origins = settings.cors_origins_production or settings.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        allow_headers=["*"],
    )

# Custom middleware
app.add_middleware(CorrelationIDMiddleware)
app.add_middleware(LoggingMiddleware)


# Global exception handlers
@app.exception_handler(AdaRuntimeException)
async def runtime_exception_handler(request: Request, exc: AdaRuntimeException) -> JSONResponse:
    """Handle custom runtime exceptions with CORS headers"""
    response = JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "details": exc.details,
                "correlation_id": getattr(request.state, "correlation_id", "unknown"),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
    )
    origin = request.headers.get("origin")
    if origin and _is_allowed_origin(origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions with CORS headers"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    response = JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "correlation_id": getattr(request.state, "correlation_id", "unknown"),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
    )
    origin = request.headers.get("origin")
    if origin and _is_allowed_origin(origin):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


def _is_allowed_origin(origin: str) -> bool:
    """Check if origin is in allowed CORS origins list."""
    if settings.cors_allow_all:
        return True
    import fnmatch
    origins = settings.cors_origins_production or settings.cors_origins
    for allowed in origins:
        if allowed == "*" or allowed == origin:
            return True
        if fnmatch.fnmatch(origin, allowed):
            return True
    return False


# Import and include routers
from api.routes import health_router, tokens_router

app.include_router(health_router)
app.include_router(tokens_router)


# MCP routing middleware — intercepts /mcp requests and forwards to the
# MCP SDK's Streamable HTTP handler. Single endpoint, no org/model.
from domains.mcp.app import MCPRoutingMiddleware

app.add_middleware(MCPRoutingMiddleware, mcp_app_getter=lambda: getattr(app.state, "mcp_session_managers", None))
