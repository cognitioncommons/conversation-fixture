"""
Diff utilities for comparing conversation fixture runs.

This module provides tools to compare two runs of the same fixture
and highlight differences in responses and validation results.
"""

import difflib
from dataclasses import dataclass, field
from typing import Any, Optional

from conversation_fixture.runner import RunResult, TurnResult


@dataclass
class TurnDiff:
    """Difference between two turn results."""

    turn_index: int
    role: str
    content_diff: Optional[list[str]] = None
    validation_changed: bool = False
    old_passed: Optional[bool] = None
    new_passed: Optional[bool] = None
    latency_delta_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "turn_index": self.turn_index,
            "role": self.role,
            "content_diff": self.content_diff,
            "validation_changed": self.validation_changed,
            "old_passed": self.old_passed,
            "new_passed": self.new_passed,
            "latency_delta_ms": self.latency_delta_ms,
        }


@dataclass
class RunDiff:
    """Difference between two fixture runs."""

    fixture_name: str
    old_model: str
    new_model: str
    old_passed: bool
    new_passed: bool
    turn_diffs: list[TurnDiff] = field(default_factory=list)
    latency_delta_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fixture_name": self.fixture_name,
            "old_model": self.old_model,
            "new_model": self.new_model,
            "old_passed": self.old_passed,
            "new_passed": self.new_passed,
            "turn_diffs": [t.to_dict() for t in self.turn_diffs],
            "latency_delta_ms": self.latency_delta_ms,
        }

    @property
    def has_differences(self) -> bool:
        """Check if there are any differences."""
        return (
            self.old_passed != self.new_passed
            or any(d.content_diff or d.validation_changed for d in self.turn_diffs)
        )

    @property
    def regression(self) -> bool:
        """Check if the new run is a regression (passed -> failed)."""
        return self.old_passed and not self.new_passed

    @property
    def improvement(self) -> bool:
        """Check if the new run is an improvement (failed -> passed)."""
        return not self.old_passed and self.new_passed


def diff_content(old: Optional[str], new: Optional[str]) -> Optional[list[str]]:
    """Generate a unified diff between two content strings.

    Args:
        old: The old content.
        new: The new content.

    Returns:
        List of diff lines, or None if content is identical.
    """
    if old == new:
        return None

    old_lines = (old or "").splitlines(keepends=True)
    new_lines = (new or "").splitlines(keepends=True)

    diff = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="old",
            tofile="new",
            lineterm="",
        )
    )

    return diff if diff else None


def diff_turns(old: TurnResult, new: TurnResult) -> TurnDiff:
    """Compare two turn results.

    Args:
        old: The old turn result.
        new: The new turn result.

    Returns:
        A TurnDiff describing the differences.
    """
    # Compare output content for assistant turns
    content_diff = None
    if old.role == "assistant":
        content_diff = diff_content(old.output_content, new.output_content)

    # Compare validation results
    old_passed = old.validation.passed if old.validation else None
    new_passed = new.validation.passed if new.validation else None
    validation_changed = old_passed != new_passed

    # Calculate latency delta
    latency_delta = new.latency_ms - old.latency_ms

    return TurnDiff(
        turn_index=old.turn_index,
        role=old.role,
        content_diff=content_diff,
        validation_changed=validation_changed,
        old_passed=old_passed,
        new_passed=new_passed,
        latency_delta_ms=latency_delta,
    )


def diff_runs(old: RunResult, new: RunResult) -> RunDiff:
    """Compare two fixture runs.

    Args:
        old: The old run result.
        new: The new run result.

    Returns:
        A RunDiff describing the differences.

    Raises:
        ValueError: If the runs are for different fixtures.
    """
    if old.fixture_name != new.fixture_name:
        raise ValueError(
            f"Cannot diff runs for different fixtures: "
            f"{old.fixture_name} vs {new.fixture_name}"
        )

    # Compare turns
    turn_diffs = []
    max_turns = max(len(old.turns), len(new.turns))

    for i in range(max_turns):
        if i < len(old.turns) and i < len(new.turns):
            turn_diffs.append(diff_turns(old.turns[i], new.turns[i]))
        elif i < len(old.turns):
            # Turn was removed in new run
            turn_diffs.append(
                TurnDiff(
                    turn_index=i,
                    role=old.turns[i].role,
                    content_diff=["- " + (old.turns[i].output_content or "")],
                    validation_changed=True,
                    old_passed=old.turns[i].validation.passed
                    if old.turns[i].validation
                    else None,
                    new_passed=None,
                )
            )
        else:
            # Turn was added in new run
            turn_diffs.append(
                TurnDiff(
                    turn_index=i,
                    role=new.turns[i].role,
                    content_diff=["+ " + (new.turns[i].output_content or "")],
                    validation_changed=True,
                    old_passed=None,
                    new_passed=new.turns[i].validation.passed
                    if new.turns[i].validation
                    else None,
                )
            )

    return RunDiff(
        fixture_name=old.fixture_name,
        old_model=old.model,
        new_model=new.model,
        old_passed=old.passed,
        new_passed=new.passed,
        turn_diffs=turn_diffs,
        latency_delta_ms=new.total_latency_ms - old.total_latency_ms,
    )


def format_diff(diff: RunDiff, verbose: bool = False) -> str:
    """Format a run diff as a human-readable string.

    Args:
        diff: The run diff to format.
        verbose: If True, include full content diffs.

    Returns:
        Formatted string representation.
    """
    lines = []

    # Header
    lines.append(f"Fixture: {diff.fixture_name}")
    lines.append(f"Models: {diff.old_model} -> {diff.new_model}")

    # Overall status
    old_status = "PASSED" if diff.old_passed else "FAILED"
    new_status = "PASSED" if diff.new_passed else "FAILED"

    if diff.regression:
        status_indicator = "[REGRESSION]"
    elif diff.improvement:
        status_indicator = "[IMPROVEMENT]"
    else:
        status_indicator = ""

    lines.append(f"Status: {old_status} -> {new_status} {status_indicator}")

    # Latency
    latency_sign = "+" if diff.latency_delta_ms >= 0 else ""
    lines.append(f"Latency: {latency_sign}{diff.latency_delta_ms:.0f}ms")

    # Turn differences
    if diff.has_differences:
        lines.append("")
        lines.append("Turn Differences:")
        lines.append("-" * 40)

        for turn_diff in diff.turn_diffs:
            if not turn_diff.content_diff and not turn_diff.validation_changed:
                continue

            lines.append(f"  Turn {turn_diff.turn_index} ({turn_diff.role}):")

            if turn_diff.validation_changed:
                old_v = "PASS" if turn_diff.old_passed else "FAIL"
                new_v = "PASS" if turn_diff.new_passed else "FAIL"
                if turn_diff.old_passed is None:
                    old_v = "N/A"
                if turn_diff.new_passed is None:
                    new_v = "N/A"
                lines.append(f"    Validation: {old_v} -> {new_v}")

            if turn_diff.content_diff and verbose:
                lines.append("    Content diff:")
                for diff_line in turn_diff.content_diff:
                    lines.append(f"      {diff_line.rstrip()}")

            if turn_diff.latency_delta_ms != 0:
                lat_sign = "+" if turn_diff.latency_delta_ms >= 0 else ""
                lines.append(f"    Latency: {lat_sign}{turn_diff.latency_delta_ms:.0f}ms")

    else:
        lines.append("")
        lines.append("No significant differences.")

    return "\n".join(lines)


def format_diff_summary(diffs: list[RunDiff]) -> str:
    """Format a summary of multiple run diffs.

    Args:
        diffs: List of run diffs.

    Returns:
        Summary string.
    """
    lines = []

    total = len(diffs)
    regressions = sum(1 for d in diffs if d.regression)
    improvements = sum(1 for d in diffs if d.improvement)
    unchanged = sum(1 for d in diffs if not d.has_differences)
    changed = total - unchanged

    lines.append(f"Summary: {total} fixtures compared")
    lines.append(f"  Regressions:  {regressions}")
    lines.append(f"  Improvements: {improvements}")
    lines.append(f"  Changed:      {changed}")
    lines.append(f"  Unchanged:    {unchanged}")

    if regressions > 0:
        lines.append("")
        lines.append("Regressions:")
        for d in diffs:
            if d.regression:
                lines.append(f"  - {d.fixture_name}")

    if improvements > 0:
        lines.append("")
        lines.append("Improvements:")
        for d in diffs:
            if d.improvement:
                lines.append(f"  - {d.fixture_name}")

    return "\n".join(lines)
