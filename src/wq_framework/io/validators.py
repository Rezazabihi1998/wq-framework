"""Per-cell schema validation and the confirm-or-halt gate.

Implements the flow confirmed in project brief Section 5:

  1. All invalid cells are identified in one pass (not stopped at the first).
  2. Each invalid cell is nulled individually (NaN) — the rest of that row
     is preserved untouched. No row/file is ever dropped for this reason.
  3. A summary report is shown to the user.
  4. The user confirms to continue (execution proceeds with those cells as
     NaN) or declines (execution halts so the user can fix the source file).

NaN values produced here are deliberately *not* dropped anywhere in this
module — downstream stages (feature selection / modeling) are responsible
for dropping rows locally, scoped only to the columns they actually use
(see project brief Section 5, "Confirmed NaN-handling strategy"), so a
NaN in one parameter doesn't cost data for models that don't use it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from .schema import Schema


@dataclass(frozen=True)
class ValidationIssue:
    station_id: str
    date: str
    parameter: str
    value: float
    reason: str

    def __str__(self) -> str:
        return f"{self.station_id}, {self.date}, {self.parameter}: {self.value} ({self.reason})"


@dataclass
class ValidationReport:
    issues: list[ValidationIssue]
    cleaned_df: pd.DataFrame

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0

    def summary(self) -> str:
        if not self.has_issues:
            return "No invalid values found."
        lines = [f"{len(self.issues)} invalid value(s) found and will be removed:"]
        lines += [f"  - {issue}" for issue in self.issues]
        return "\n".join(lines)


class SchemaValidator:
    """Applies each parameter's min/max sanity bounds, cell by cell."""

    def __init__(self, schema: Schema):
        self.schema = schema

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        issues: list[ValidationIssue] = []
        cleaned = df.copy()

        for param_name, rule in self.schema.parameters.items():
            if param_name not in cleaned.columns:
                continue  # optional parameter, absent entirely — allowed

            col = cleaned[param_name]
            for idx in col.index:
                value = col.at[idx]
                if pd.isna(value):
                    continue
                if rule.is_out_of_range(value):
                    issues.append(
                        ValidationIssue(
                            station_id=str(cleaned.at[idx, "station_id"]),
                            date=str(pd.Timestamp(cleaned.at[idx, "date"]).date()),
                            parameter=param_name,
                            value=value,
                            reason=rule.reason(value),
                        )
                    )
                    cleaned.at[idx, param_name] = pd.NA

        return ValidationReport(issues=issues, cleaned_df=cleaned)


def _default_confirm(report: ValidationReport) -> bool:
    """Default interactive confirmation via stdin (CLI usage)."""
    print(report.summary())
    answer = input("Continue with these cells removed? [y/N]: ").strip().lower()
    return answer in ("y", "yes")


def run_validation_gate(
    df: pd.DataFrame,
    schema: Schema,
    confirm_callback: Optional[Callable[[ValidationReport], bool]] = None,
) -> pd.DataFrame:
    """Run schema validation and enforce the confirm-or-halt gate.

    `confirm_callback` receives the ValidationReport and returns True to
    proceed or False to halt. Defaults to an interactive stdin prompt; a
    future GUI can pass its own callback (e.g. a dialog box) without any
    change to this function.

    Raises RuntimeError if the user does not confirm (or if there are
    issues and no way to confirm was provided in a non-interactive
    context) — the caller is expected to let this propagate and stop the
    pipeline, per the confirmed design ("execution halts so the user can
    correct the input file and re-run").
    """
    report = SchemaValidator(schema).validate(df)

    if not report.has_issues:
        return report.cleaned_df

    confirm = confirm_callback or _default_confirm
    if confirm(report):
        return report.cleaned_df

    raise RuntimeError(
        "Execution halted: invalid values were found and not approved for "
        "removal. Please correct the input file and re-run.\n" + report.summary()
    )
