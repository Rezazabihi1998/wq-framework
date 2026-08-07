from unittest.mock import patch

import pytest

from wq_framework.satellite.source_selection import resolve_satellite_source


def test_reads_from_config_when_present():
    config = {"satellite": {"source": "manual"}}
    assert resolve_satellite_source(config) == "manual"


def test_invalid_config_value_raises():
    config = {"satellite": {"source": "carrier_pigeon"}}
    with pytest.raises(ValueError, match="must be one of"):
        resolve_satellite_source(config)


def test_missing_config_prompts_interactively():
    config = {}
    with patch("builtins.input", return_value="gee"):
        assert resolve_satellite_source(config) == "gee"


def test_missing_config_accepts_numeric_choice():
    config = {}
    with patch("builtins.input", return_value="2"):
        assert resolve_satellite_source(config) == "manual"


def test_missing_config_reprompts_on_invalid_input():
    config = {}
    with patch("builtins.input", side_effect=["banana", "gee"]):
        assert resolve_satellite_source(config) == "gee"


def test_missing_config_raises_when_prompting_disabled():
    config = {}
    with pytest.raises(ValueError, match="interactive prompting is disabled"):
        resolve_satellite_source(config, prompt_if_missing=False)