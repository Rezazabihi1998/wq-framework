"""Resolves which of the three satellite-data modes to use (project brief
Section 7): config file first, interactive CLI prompt as a fallback.

This module has no GEE dependency — it only decides *which* source to use;
gee_client.py and the manual retriever (Stage 3.3) handle the rest.
"""

from __future__ import annotations

VALID_SOURCES = ("gee", "manual")


def resolve_satellite_source(config: dict, prompt_if_missing: bool = True) -> str:
    """Return "gee" or "manual".

    Looks for config["satellite"]["source"] first. If absent (or the
    config section itself is absent) and `prompt_if_missing` is True,
    asks interactively via stdin. Pass `prompt_if_missing=False` for
    non-interactive contexts (e.g. tests, or a future GUI that supplies
    its own selection mechanism instead of a terminal prompt).
    """
    configured = (config.get("satellite") or {}).get("source")
    if configured is not None:
        if configured not in VALID_SOURCES:
            raise ValueError(
                f"config satellite.source must be one of {VALID_SOURCES}, "
                f"got '{configured}'"
            )
        return configured

    if not prompt_if_missing:
        raise ValueError(
            "satellite.source is not set in the config, and interactive "
            "prompting is disabled in this context."
        )

    return _prompt_for_source()


def _prompt_for_source() -> str:
    print("How should satellite data be obtained?")
    print("  1) gee    - connect to Google Earth Engine automatically")
    print("  2) manual - I will provide a pre-prepared data file myself")
    while True:
        choice = input("Enter 1 or 2 (or 'gee'/'manual'): ").strip().lower()
        if choice in ("1", "gee"):
            return "gee"
        if choice in ("2", "manual"):
            return "manual"
        print("Not understood — please enter 1, 2, 'gee', or 'manual'.")