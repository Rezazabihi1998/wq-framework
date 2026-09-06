"""Loading and representing schema.yaml (the parameter registry).

Design rationale: schema validation is a *sanity gate* on raw input, used to
reject physically impossible values. It is independent from, and
complementary to, the later statistical (IQR) outlier detection stage, which
looks for anomalies *within* the sanity-valid range.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass(frozen=True)
class ParameterRule:
    """Validation rule for a single water quality parameter column."""

    name: str
    unit: str
    required: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def is_out_of_range(self, value: float) -> bool:
        """True if `value` violates this parameter's sanity bounds."""
        if value is None:
            return False
        if self.min_value is not None and value < self.min_value:
            return True
        if self.max_value is not None and value > self.max_value:
            return True
        return False

    def reason(self, value: float) -> str:
        """Human-readable reason a value was flagged, for reporting."""
        if self.min_value is not None and value < self.min_value:
            return f"below minimum ({self.min_value})"
        if self.max_value is not None and value > self.max_value:
            return f"above maximum ({self.max_value})"
        return "out of range"


@dataclass(frozen=True)
class ValidationPolicy:
    on_out_of_range: str = "null_cell"
    on_missing_optional: str = "allow"
    on_unknown_column: str = "reject_file"


@dataclass(frozen=True)
class Schema:
    """The full parameter registry loaded from schema.yaml."""

    parameters: dict[str, ParameterRule] = field(default_factory=dict)
    required_columns: tuple[str, ...] = ("station_id", "date")
    date_format: str = "%Y-%m-%d"
    policy: ValidationPolicy = field(default_factory=ValidationPolicy)

    @classmethod
    def load(cls, path: str | Path) -> "Schema":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        parameters = {
            name: ParameterRule(
                name=name,
                unit=rules.get("unit", "-"),
                required=bool(rules.get("required", False)),
                min_value=rules.get("min_value"),
                max_value=rules.get("max_value"),
            )
            for name, rules in (raw.get("parameters") or {}).items()
        }

        required_cols = tuple((raw.get("required_columns") or {}).keys()) or (
            "station_id",
            "date",
        )
        date_fmt = (
            (raw.get("required_columns") or {}).get("date", {}).get("format", "%Y-%m-%d")
        )

        policy_raw = raw.get("validation_policy") or {}
        policy = ValidationPolicy(
            on_out_of_range=policy_raw.get("on_out_of_range", "null_cell"),
            on_missing_optional=policy_raw.get("on_missing_optional", "allow"),
            on_unknown_column=policy_raw.get("on_unknown_column", "reject_file"),
        )

        return cls(
            parameters=parameters,
            required_columns=required_cols,
            date_format=date_fmt,
            policy=policy,
        )

    @property
    def known_columns(self) -> set[str]:
        """All column names schema.yaml knows about (parameters + required)."""
        return set(self.parameters.keys()) | set(self.required_columns)
