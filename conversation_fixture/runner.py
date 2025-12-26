"""
Fixture runner for executing conversation fixtures against models.

This module handles sending conversation fixtures to language model APIs
and collecting responses.
"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx

from conversation_fixture.fixture import ConversationFixture, Turn
from conversation_fixture.validator import ValidationResult, validate_turn


@dataclass
class TurnResult:
    """Result of executing a single turn."""

    turn_index: int
    role: str
    input_content: Optional[str]
    output_content: Optional[str]
    validation: Optional[ValidationResult]
    latency_ms: float
    tokens_used: Optional[dict[str, int]] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "turn_index": self.turn_index,
            "role": self.role,
            "input_content": self.input_content,
            "output_content": self.output_content,
            "validation": self.validation.to_dict() if self.validation else None,
            "latency_ms": self.latency_ms,
            "tokens_used": self.tokens_used,
            "error": self.error,
        }


@dataclass
class RunResult:
    """Result of executing a complete fixture."""

    fixture_name: str
    model: str
    timestamp: datetime
    turns: list[TurnResult]
    total_latency_ms: float
    passed: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fixture_name": self.fixture_name,
            "model": self.model,
            "timestamp": self.timestamp.isoformat(),
            "turns": [t.to_dict() for t in self.turns],
            "total_latency_ms": self.total_latency_ms,
            "passed": self.passed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunResult":
        """Create a RunResult from a dictionary."""
        turns = []
        for t in data.get("turns", []):
            validation = None
            if t.get("validation"):
                validation = ValidationResult(
                    passed=t["validation"]["passed"],
                    checks=t["validation"]["checks"],
                    errors=t["validation"]["errors"],
                )
            turns.append(
                TurnResult(
                    turn_index=t["turn_index"],
                    role=t["role"],
                    input_content=t.get("input_content"),
                    output_content=t.get("output_content"),
                    validation=validation,
                    latency_ms=t["latency_ms"],
                    tokens_used=t.get("tokens_used"),
                    error=t.get("error"),
                )
            )

        return cls(
            fixture_name=data["fixture_name"],
            model=data["model"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            turns=turns,
            total_latency_ms=data["total_latency_ms"],
            passed=data["passed"],
            metadata=data.get("metadata", {}),
        )

    def summary(self) -> str:
        """Get a summary of the run result."""
        status = "PASSED" if self.passed else "FAILED"
        failed_turns = [t for t in self.turns if t.validation and not t.validation.passed]
        return (
            f"{self.fixture_name}: {status} "
            f"({len(self.turns)} turns, {len(failed_turns)} failed, "
            f"{self.total_latency_ms:.0f}ms)"
        )


def save_run_result(result: RunResult, path: str | Path) -> None:
    """Save a run result to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)


def load_run_result(path: str | Path) -> RunResult:
    """Load a run result from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return RunResult.from_dict(data)


class FixtureRunner:
    """Runs conversation fixtures against language model APIs."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """Initialize the fixture runner.

        Args:
            api_url: Base URL for the API (default: OpenAI-compatible).
            api_key: API key (default: from OPENAI_API_KEY env var).
            model: Default model to use.
            timeout: Request timeout in seconds.
        """
        self.api_url = api_url or os.environ.get(
            "OPENAI_API_BASE", "https://api.openai.com/v1"
        )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

    def _build_messages(
        self,
        fixture: ConversationFixture,
        up_to_index: int,
        responses: dict[int, str],
    ) -> list[dict[str, str]]:
        """Build the messages array for an API call.

        Args:
            fixture: The conversation fixture.
            up_to_index: Build messages up to this turn index.
            responses: Map of turn index to collected response.

        Returns:
            List of message dicts for the API.
        """
        messages = []

        # Add system prompt if present
        if fixture.system_prompt:
            messages.append({"role": "system", "content": fixture.system_prompt})

        # Add turns up to the specified index
        for i, turn in enumerate(fixture.turns[:up_to_index]):
            if turn.role == "user":
                if turn.content:
                    messages.append({"role": "user", "content": turn.content})
            elif turn.role == "assistant":
                # Use collected response if available, otherwise skip
                if i in responses:
                    messages.append({"role": "assistant", "content": responses[i]})
            elif turn.role == "system":
                if turn.content:
                    messages.append({"role": "system", "content": turn.content})

        return messages

    def _call_api(
        self,
        messages: list[dict[str, str]],
        model: str,
    ) -> tuple[str, dict[str, int], float]:
        """Call the chat completion API.

        Args:
            messages: The messages to send.
            model: The model to use.

        Returns:
            Tuple of (response content, token usage, latency_ms).
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": messages,
        }

        start = time.time()
        response = self.client.post(
            f"{self.api_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        latency_ms = (time.time() - start) * 1000

        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return content, usage, latency_ms

    def run(
        self,
        fixture: ConversationFixture,
        model: Optional[str] = None,
        dry_run: bool = False,
    ) -> RunResult:
        """Execute a conversation fixture.

        Args:
            fixture: The fixture to run.
            model: Model to use (overrides fixture and default).
            dry_run: If True, don't actually call the API.

        Returns:
            A RunResult with all turn results and validation.
        """
        model = model or fixture.model or self.model
        turn_results: list[TurnResult] = []
        responses: dict[int, str] = {}
        total_latency = 0.0
        all_passed = True
        start_time = datetime.now()

        for i, turn in enumerate(fixture.turns):
            if turn.role == "user":
                # User turns don't need API calls
                turn_results.append(
                    TurnResult(
                        turn_index=i,
                        role="user",
                        input_content=turn.content,
                        output_content=None,
                        validation=None,
                        latency_ms=0,
                    )
                )
            elif turn.role == "assistant":
                # Assistant turns need API calls
                messages = self._build_messages(fixture, i, responses)

                if dry_run:
                    # Simulate response
                    response_content = f"[DRY RUN] Response for turn {i}"
                    tokens = {}
                    latency = 0.0
                    error = None
                else:
                    try:
                        response_content, tokens, latency = self._call_api(
                            messages, model
                        )
                        error = None
                    except Exception as e:
                        response_content = ""
                        tokens = {}
                        latency = 0.0
                        error = str(e)

                responses[i] = response_content
                total_latency += latency

                # Validate if expectations are defined
                validation = None
                if turn.expect:
                    validation = validate_turn(response_content, turn)
                    if not validation.passed:
                        all_passed = False

                turn_results.append(
                    TurnResult(
                        turn_index=i,
                        role="assistant",
                        input_content=None,
                        output_content=response_content,
                        validation=validation,
                        latency_ms=latency,
                        tokens_used=tokens if tokens else None,
                        error=error,
                    )
                )

                if error:
                    all_passed = False

            elif turn.role == "system":
                # System turns are context-only
                turn_results.append(
                    TurnResult(
                        turn_index=i,
                        role="system",
                        input_content=turn.content,
                        output_content=None,
                        validation=None,
                        latency_ms=0,
                    )
                )

        return RunResult(
            fixture_name=fixture.name,
            model=model,
            timestamp=start_time,
            turns=turn_results,
            total_latency_ms=total_latency,
            passed=all_passed,
            metadata=fixture.metadata,
        )

    def run_fixtures(
        self,
        fixtures: list[ConversationFixture],
        model: Optional[str] = None,
        dry_run: bool = False,
    ) -> list[RunResult]:
        """Execute multiple fixtures.

        Args:
            fixtures: List of fixtures to run.
            model: Model to use for all fixtures.
            dry_run: If True, don't actually call the API.

        Returns:
            List of RunResult objects.
        """
        results = []
        for fixture in fixtures:
            result = self.run(fixture, model=model, dry_run=dry_run)
            results.append(result)
        return results
