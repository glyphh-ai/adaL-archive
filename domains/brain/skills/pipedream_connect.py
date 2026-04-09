"""
Pipedream Connect — wires Ada to 3,145 APIs.

Three parts:
  1. Setup: user provides Pipedream project credentials (once)
  2. Connect: user authenticates their apps via Pipedream's connect URL
  3. Execute: Ada runs actions on behalf of the user

The OAuth tokens live in Pipedream, not in Ada. Ada only stores:
  - Project credentials (client_id, secret, project_id)
  - The user's external_user_id (maps to their Pipedream accounts)

Flow:
  ada> setup pipedream
  → stores credentials in ~/.ada/env

  ada> connect slack
  → generates a Pipedream Connect URL
  → user opens in browser, authenticates
  → Pipedream stores the OAuth token

  ada> send a slack message to #general saying hello
  → cognitive router → pipedream capability
  → HDC routes to slack-send-message-to-channel
  → PipedreamExecutor runs the action
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

PIPEDREAM_API = "https://api.pipedream.com/v1"


class PipedreamConnect:
    """Manages Pipedream Connect integration for Ada."""

    def __init__(self):
        from domains.brain.skills.vault import get_vault
        vault = get_vault()

        # Vault first, env fallback
        self.client_id = vault.get("pipedream_client_id") or os.environ.get("PIPEDREAM_CLIENT_ID")
        self.client_secret = vault.get("pipedream_client_secret") or os.environ.get("PIPEDREAM_CLIENT_SECRET")
        self.project_id = vault.get("pipedream_project_id") or os.environ.get("PIPEDREAM_PROJECT_ID")
        self.environment = os.environ.get("PIPEDREAM_ENVIRONMENT", "production")
        self.user_id = os.environ.get("ADA_USER_ID", "default")

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret and self.project_id)

    def get_connect_url(self, app_slug: str) -> Optional[str]:
        """Generate a Pipedream Connect URL for the user to authenticate an app.

        The user opens this in their browser, authenticates with the app
        (Slack, Gmail, etc.), and Pipedream stores the OAuth token.

        Returns the URL, or None if not configured.
        """
        if not self.configured:
            return None

        try:
            import httpx

            # Get a connect token from Pipedream
            token = self._get_server_token()
            if not token:
                return None

            # The connect URL
            return (
                f"https://pipedream.com/connect/{self.project_id}"
                f"?token={token}"
                f"&app={app_slug}"
                f"&external_user_id={self.user_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to generate connect URL: {e}")
            return None

    def _get_server_token(self) -> Optional[str]:
        """Get a server-side token for Pipedream Connect."""
        try:
            import httpx

            resp = httpx.post(
                f"{PIPEDREAM_API}/connect/{self.project_id}/tokens",
                json={
                    "external_user_id": self.user_id,
                },
                headers={
                    "Authorization": f"Bearer {self._get_access_token()}",
                    "Content-Type": "application/json",
                },
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json().get("token")
        except Exception as e:
            logger.warning(f"Failed to get connect token: {e}")
        return None

    def _get_access_token(self) -> str:
        """Get OAuth access token via client credentials."""
        import httpx

        resp = httpx.post(
            f"{PIPEDREAM_API}/oauth/token",
            json={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["access_token"]

    def list_connected_apps(self) -> list[dict]:
        """List apps the user has connected."""
        if not self.configured:
            return []

        try:
            import httpx

            resp = httpx.get(
                f"{PIPEDREAM_API}/connect/{self.project_id}/accounts",
                headers={
                    "Authorization": f"Bearer {self._get_access_token()}",
                },
                params={"external_user_id": self.user_id},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json().get("data", [])
        except Exception as e:
            logger.warning(f"Failed to list connected apps: {e}")
        return []

    async def execute_action(
        self,
        action_key: str,
        props: dict[str, Any] | None = None,
    ) -> dict:
        """Execute a Pipedream action on behalf of the user.

        The HDC router already determined the action_key. This just
        executes it via the Connect API.
        """
        if not self.configured:
            return {"success": False, "error": "Pipedream not configured. Run: ada setup pipedream"}

        try:
            import httpx

            resp = httpx.post(
                f"{PIPEDREAM_API}/connect/{self.project_id}/actions/run",
                headers={
                    "Authorization": f"Bearer {self._get_access_token()}",
                    "X-PD-Environment": self.environment,
                    "Content-Type": "application/json",
                },
                json={
                    "id": action_key,
                    "external_user_id": self.user_id,
                    "configured_props": props or {},
                },
                timeout=30,
            )

            if resp.status_code == 200:
                return {"success": True, "data": resp.json()}
            else:
                return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:300]}"}

        except Exception as e:
            return {"success": False, "error": str(e)}
