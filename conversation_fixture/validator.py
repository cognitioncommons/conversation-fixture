"""
Validation for conversation fixture responses.

This module validates assistant responses against expected criteria
defined in fixture expectations.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

from conversation_fixture.fixture import Expectation, Turn


@dataclass
class CheckResult:
    """Result of a single validation check."""

    check_type: str
    passed: bool
    expected: Any
    actual: Any
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "check_type": self.check_type,
            "passed": self.passed,
            "expected": self.expected,
            "actual": self.actual,
            "message": self.message,
        }


@dataclass
class ValidationResult:
    """Result of validating a response against expectations."""

    passed: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "checks": self.checks,
            "errors": self.errors,
        }

    def add_check(self, check: CheckResult) -> None:
        """Add a check result."""
        self.checks.append(check.to_dict())
        if not check.passed:
            self.passed = False
            self.errors.append(check.message)


def check_contains(response: str, expected: list[str]) -> list[CheckResult]:
    """Check that response contains all expected strings.

    Args:
        response: The response text to check.
        expected: List of strings that must be present.

    Returns:
        List of CheckResult objects.
    """
    results = []
    response_lower = response.lower()

    for item in expected:
        item_lower = item.lower()
        found = item_lower in response_lower
        results.append(
            CheckResult(
                check_type="contains",
                passed=found,
                expected=item,
                actual=found,
                message=f"Response {'contains' if found else 'does not contain'} '{item}'",
            )
        )

    return results


def check_not_contains(response: str, forbidden: list[str]) -> list[CheckResult]:
    """Check that response does not contain forbidden strings.

    Args:
        response: The response text to check.
        forbidden: List of strings that must not be present.

    Returns:
        List of CheckResult objects.
    """
    results = []
    response_lower = response.lower()

    for item in forbidden:
        item_lower = item.lower()
        found = item_lower in response_lower
        results.append(
            CheckResult(
                check_type="not_contains",
                passed=not found,
                expected=f"not contain '{item}'",
                actual=found,
                message=f"Response {'contains' if found else 'does not contain'} forbidden string '{item}'",
            )
        )

    return results


def check_starts_with(response: str, prefix: str) -> CheckResult:
    """Check that response starts with a prefix.

    Args:
        response: The response text to check.
        prefix: The expected prefix.

    Returns:
        A CheckResult object.
    """
    matches = response.lower().startswith(prefix.lower())
    return CheckResult(
        check_type="starts_with",
        passed=matches,
        expected=prefix,
        actual=response[:len(prefix)] if len(response) >= len(prefix) else response,
        message=f"Response {'starts' if matches else 'does not start'} with '{prefix}'",
    )


def check_ends_with(response: str, suffix: str) -> CheckResult:
    """Check that response ends with a suffix.

    Args:
        response: The response text to check.
        suffix: The expected suffix.

    Returns:
        A CheckResult object.
    """
    matches = response.lower().endswith(suffix.lower())
    return CheckResult(
        check_type="ends_with",
        passed=matches,
        expected=suffix,
        actual=response[-len(suffix):] if len(response) >= len(suffix) else response,
        message=f"Response {'ends' if matches else 'does not end'} with '{suffix}'",
    )


def check_matches(response: str, pattern: str) -> CheckResult:
    """Check that response matches a regex pattern.

    Args:
        response: The response text to check.
        pattern: The regex pattern to match.

    Returns:
        A CheckResult object.
    """
    try:
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        matches = match is not None
        return CheckResult(
            check_type="matches",
            passed=matches,
            expected=pattern,
            actual=match.group(0) if match else None,
            message=f"Response {'matches' if matches else 'does not match'} pattern '{pattern}'",
        )
    except re.error as e:
        return CheckResult(
            check_type="matches",
            passed=False,
            expected=pattern,
            actual=None,
            message=f"Invalid regex pattern '{pattern}': {e}",
        )


def check_length(
    response: str,
    min_length: int | None = None,
    max_length: int | None = None,
) -> list[CheckResult]:
    """Check that response length is within bounds.

    Args:
        response: The response text to check.
        min_length: Minimum length (optional).
        max_length: Maximum length (optional).

    Returns:
        List of CheckResult objects.
    """
    results = []
    actual_length = len(response)

    if min_length is not None:
        passes = actual_length >= min_length
        results.append(
            CheckResult(
                check_type="min_length",
                passed=passes,
                expected=min_length,
                actual=actual_length,
                message=f"Response length ({actual_length}) {'>=':s if passes else '<'} minimum ({min_length})",
            )
        )

    if max_length is not None:
        passes = actual_length <= max_length
        results.append(
            CheckResult(
                check_type="max_length",
                passed=passes,
                expected=max_length,
                actual=actual_length,
                message=f"Response length ({actual_length}) {'<=' if passes else '>'} maximum ({max_length})",
            )
        )

    return results


def check_json_schema(response: str, schema: dict[str, Any]) -> CheckResult:
    """Check that response is valid JSON matching a schema.

    Args:
        response: The response text to check.
        schema: JSON schema to validate against.

    Returns:
        A CheckResult object.
    """
    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        return CheckResult(
            check_type="json_schema",
            passed=False,
            expected="valid JSON",
            actual=str(e),
            message=f"Response is not valid JSON: {e}",
        )

    # Basic schema validation (type checking)
    # For full JSON Schema validation, would need jsonschema library
    if "type" in schema:
        expected_type = schema["type"]
        actual_type = type(data).__name__

        type_map = {
            "object": "dict",
            "array": "list",
            "string": "str",
            "number": ("int", "float"),
            "integer": "int",
            "boolean": "bool",
            "null": "NoneType",
        }

        expected_python_type = type_map.get(expected_type, expected_type)

        if isinstance(expected_python_type, tuple):
            matches = actual_type in expected_python_type
        else:
            matches = actual_type == expected_python_type

        if not matches:
            return CheckResult(
                check_type="json_schema",
                passed=False,
                expected=f"type '{expected_type}'",
                actual=actual_type,
                message=f"JSON type mismatch: expected {expected_type}, got {actual_type}",
            )

    # Check required properties for objects
    if schema.get("type") == "object" and "required" in schema:
        missing = [key for key in schema["required"] if key not in data]
        if missing:
            return CheckResult(
                check_type="json_schema",
                passed=False,
                expected=f"required properties: {schema['required']}",
                actual=list(data.keys()) if isinstance(data, dict) else data,
                message=f"Missing required properties: {missing}",
            )

    return CheckResult(
        check_type="json_schema",
        passed=True,
        expected=schema,
        actual=data,
        message="Response matches JSON schema",
    )


def validate_response(response: str, expectation: Expectation) -> ValidationResult:
    """Validate a response against an expectation.

    Args:
        response: The response text to validate.
        expectation: The expectation to check against.

    Returns:
        A ValidationResult with all check results.
    """
    result = ValidationResult(passed=True)

    # Check contains
    if expectation.contains:
        for check in check_contains(response, expectation.contains):
            result.add_check(check)

    # Check not_contains
    if expectation.not_contains:
        for check in check_not_contains(response, expectation.not_contains):
            result.add_check(check)

    # Check starts_with
    if expectation.starts_with:
        result.add_check(check_starts_with(response, expectation.starts_with))

    # Check ends_with
    if expectation.ends_with:
        result.add_check(check_ends_with(response, expectation.ends_with))

    # Check matches
    if expectation.matches:
        result.add_check(check_matches(response, expectation.matches))

    # Check length
    if expectation.min_length is not None or expectation.max_length is not None:
        for check in check_length(
            response, expectation.min_length, expectation.max_length
        ):
            result.add_check(check)

    # Check JSON schema
    if expectation.json_schema:
        result.add_check(check_json_schema(response, expectation.json_schema))

    return result


def validate_turn(response: str, turn: Turn) -> ValidationResult:
    """Validate a response against a turn's expectations.

    Args:
        response: The response text to validate.
        turn: The turn containing expectations.

    Returns:
        A ValidationResult, or a passed result if no expectations.
    """
    if not turn.expect:
        return ValidationResult(passed=True)

    return validate_response(response, turn.expect)
