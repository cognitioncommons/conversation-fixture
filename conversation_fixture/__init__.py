"""
Conversation Fixture - Multi-turn conversation test fixture manager.

A tool for managing, running, and validating multi-turn conversation tests
against language models.
"""

from conversation_fixture.fixture import (
    ConversationFixture,
    Turn,
    Expectation,
    load_fixture,
    save_fixture,
)
from conversation_fixture.runner import FixtureRunner, RunResult
from conversation_fixture.validator import (
    ValidationResult,
    validate_response,
    validate_turn,
)
from conversation_fixture.diff import (
    RunDiff,
    TurnDiff,
    diff_runs,
    format_diff,
)

__version__ = "0.1.0"

__all__ = [
    # Fixture
    "ConversationFixture",
    "Turn",
    "Expectation",
    "load_fixture",
    "save_fixture",
    # Runner
    "FixtureRunner",
    "RunResult",
    # Validator
    "ValidationResult",
    "validate_response",
    "validate_turn",
    # Diff
    "RunDiff",
    "TurnDiff",
    "diff_runs",
    "format_diff",
]
