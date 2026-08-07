"""GEE connection and authentication (project brief Section 7).

Handles two of the three user-facing modes:
  1. Automated GEE — credentials already cached or a service account is
     configured, so no interactive prompt is needed.
  2. Automated GEE requiring a one-time interactive auth prompt.

The third mode (fully offline/manual) never touches this module at all —
it lives in a separate ManualFileRetriever (added in Stage 3.3) that
implements the same SatelliteVariableRetriever interface without any GEE
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import ee
except ImportError:  # pragma: no cover - exercised only when ee isn't installed
    ee = None


def ensure_ee_installed() -> None:
    if ee is None:
        raise ImportError(
            "The 'earthengine-api' package is required for GEE-based "
            "satellite retrieval. Install it with: pip install earthengine-api\n"
            "(Not needed if you use satellite.source: manual in your config.)"
        )


@dataclass
class GEEConfig:
    """Connection settings."""

    project_id: str | None = None  # your GEE Cloud project id
    service_account_key_path: str | None = None  # for unattended/CI auth


class GEEAuthenticationRequired(Exception):
    """Raised when GEE needs one-time interactive authentication.

    Carries a human-readable `instructions` string the caller (CLI today,
    a future GUI later) should display to the user — this class does NOT
    block on input() itself, so it stays usable from any calling context.
    """

    def __init__(self, instructions: str):
        self.instructions = instructions
        super().__init__(instructions)


def connect(config: GEEConfig | None = None) -> None:
    """Establish a GEE session.

    Tries silent/cached authentication first (mode 1). If that fails and
    no service account is configured, raises GEEAuthenticationRequired
    (mode 2) instead of prompting directly — the caller decides how to
    surface that to the user. After the user completes
    `complete_interactive_authentication()` once, future `connect()`
    calls succeed silently (credentials are cached locally by `ee`).
    """
    ensure_ee_installed()
    config = config or GEEConfig()

    if config.service_account_key_path:
        credentials = ee.ServiceAccountCredentials(
            email=None, key_file=config.service_account_key_path
        )
        ee.Initialize(credentials, project=config.project_id)
        return

    try:
        ee.Initialize(project=config.project_id)  # uses cached credentials only
    except Exception as exc:
        raise GEEAuthenticationRequired(
            "Google Earth Engine needs one-time authentication.\n"
            "Run this once in a Python shell:\n\n"
            "    import ee; ee.Authenticate()\n\n"
            "This opens a browser link — follow it, sign in, and approve "
            "access. Then re-run your command; it will connect silently "
            "from then on."
        ) from exc


def complete_interactive_authentication(config: GEEConfig | None = None) -> None:
    """Runs the interactive ee.Authenticate() browser flow, then connects.
    Call this once — credentials are cached locally afterward, so future
    connect() calls won't need this again.
    """
    ensure_ee_installed()
    ee.Authenticate()
    connect(config)