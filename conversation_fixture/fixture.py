"""
Fixture loading and parsing for conversation test fixtures.

This module handles YAML fixture files that define multi-turn conversation tests.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class Expectation:
    """Defines expected properties of an assistant response."""

    contains: list[str] = field(default_factory=list)
    not_contains: list[str] = field(default_factory=list)
    starts_with: Optional[str] = None
    ends_with: Optional[str] = None
    matches: Optional[str] = None  # Regex pattern
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    json_schema: Optional[dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Expectation":
        """Create an Expectation from a dictionary."""
        return cls(
            contains=data.get("contains", []),
            not_contains=data.get("not_contains", []),
            starts_with=data.get("starts_with"),
            ends_with=data.get("ends_with"),
            matches=data.get("matches"),
            min_length=data.get("min_length"),
            max_length=data.get("max_length"),
            json_schema=data.get("json_schema"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {}
        if self.contains:
            result["contains"] = self.contains
        if self.not_contains:
            result["not_contains"] = self.not_contains
        if self.starts_with:
            result["starts_with"] = self.starts_with
        if self.ends_with:
            result["ends_with"] = self.ends_with
        if self.matches:
            result["matches"] = self.matches
        if self.min_length is not None:
            result["min_length"] = self.min_length
        if self.max_length is not None:
            result["max_length"] = self.max_length
        if self.json_schema:
            result["json_schema"] = self.json_schema
        return result


@dataclass
class Turn:
    """A single turn in a conversation."""

    role: str  # "user", "assistant", or "system"
    content: Optional[str] = None
    expect: Optional[Expectation] = None
    name: Optional[str] = None  # Optional name for the turn

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Turn":
        """Create a Turn from a dictionary."""
        expect = None
        if "expect" in data:
            expect = Expectation.from_dict(data["expect"])

        return cls(
            role=data["role"],
            content=data.get("content"),
            expect=expect,
            name=data.get("name"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {"role": self.role}
        if self.content is not None:
            result["content"] = self.content
        if self.expect:
            result["expect"] = self.expect.to_dict()
        if self.name:
            result["name"] = self.name
        return result


@dataclass
class ConversationFixture:
    """A multi-turn conversation test fixture."""

    name: str
    turns: list[Turn]
    description: Optional[str] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationFixture":
        """Create a ConversationFixture from a dictionary."""
        turns = [Turn.from_dict(t) for t in data.get("turns", [])]

        return cls(
            name=data["name"],
            turns=turns,
            description=data.get("description"),
            model=data.get("model"),
            system_prompt=data.get("system_prompt"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "name": self.name,
            "turns": [t.to_dict() for t in self.turns],
        }
        if self.description:
            result["description"] = self.description
        if self.model:
            result["model"] = self.model
        if self.system_prompt:
            result["system_prompt"] = self.system_prompt
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def get_user_turns(self) -> list[Turn]:
        """Get all user turns."""
        return [t for t in self.turns if t.role == "user"]

    def get_assistant_turns(self) -> list[Turn]:
        """Get all assistant turns (expected responses)."""
        return [t for t in self.turns if t.role == "assistant"]


def load_fixture(path: str | Path) -> ConversationFixture:
    """Load a conversation fixture from a YAML file.

    Args:
        path: Path to the YAML fixture file.

    Returns:
        A ConversationFixture object.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        KeyError: If required fields are missing.
    """
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError(f"Empty fixture file: {path}")

    if "name" not in data:
        # Use filename as name if not specified
        data["name"] = path.stem

    return ConversationFixture.from_dict(data)


def load_fixtures(path: str | Path) -> list[ConversationFixture]:
    """Load all fixtures from a directory or a single file.

    Args:
        path: Path to a YAML file or directory containing fixtures.

    Returns:
        List of ConversationFixture objects.
    """
    path = Path(path)

    if path.is_file():
        return [load_fixture(path)]

    fixtures = []
    for yaml_file in path.glob("**/*.yaml"):
        try:
            fixtures.append(load_fixture(yaml_file))
        except Exception as e:
            print(f"Warning: Failed to load {yaml_file}: {e}")

    for yml_file in path.glob("**/*.yml"):
        try:
            fixtures.append(load_fixture(yml_file))
        except Exception as e:
            print(f"Warning: Failed to load {yml_file}: {e}")

    return fixtures


def save_fixture(fixture: ConversationFixture, path: str | Path) -> None:
    """Save a conversation fixture to a YAML file.

    Args:
        fixture: The fixture to save.
        path: Path to save the fixture to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(
            fixture.to_dict(),
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


def create_fixture_template(
    name: str,
    description: Optional[str] = None,
    model: Optional[str] = None,
) -> ConversationFixture:
    """Create a basic fixture template with a sample turn.

    Args:
        name: Name for the fixture.
        description: Optional description.
        model: Optional model name.

    Returns:
        A ConversationFixture with a sample user/assistant exchange.
    """
    return ConversationFixture(
        name=name,
        description=description or "A conversation test fixture",
        model=model,
        turns=[
            Turn(role="user", content="Hello"),
            Turn(
                role="assistant",
                expect=Expectation(contains=["hello", "hi"]),
            ),
        ],
    )
