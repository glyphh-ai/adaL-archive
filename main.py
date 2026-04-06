"""Thin shim — the real app lives in ada.server."""

from ada.server import app  # noqa: F401

if __name__ == "__main__":
    import uvicorn
    from infrastructure.config import get_settings

    settings = get_settings()
    uvicorn.run("ada.server:app", host=settings.host, port=settings.port, log_level="info")
