"""
CLI for conversation fixture management.

Commands:
  new     - Create a new fixture template
  run     - Execute fixtures against a model
  diff    - Compare two run results
  replay  - Replay a fixture interactively
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

from conversation_fixture.diff import diff_runs, format_diff, format_diff_summary
from conversation_fixture.fixture import (
    ConversationFixture,
    Expectation,
    Turn,
    create_fixture_template,
    load_fixture,
    load_fixtures,
    save_fixture,
)
from conversation_fixture.runner import (
    FixtureRunner,
    RunResult,
    load_run_result,
    save_run_result,
)


@click.group()
@click.version_option()
def cli():
    """Conversation Fixture - Multi-turn conversation test fixture manager.

    Manage, run, and validate multi-turn conversation tests against LLMs.
    """
    pass


@cli.command()
@click.argument("name")
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output file path. Default: <name>.yaml",
)
@click.option(
    "-d",
    "--description",
    help="Description for the fixture.",
)
@click.option(
    "-m",
    "--model",
    help="Default model to use for this fixture.",
)
@click.option(
    "--system-prompt",
    help="System prompt for the conversation.",
)
def new(
    name: str,
    output: Optional[str],
    description: Optional[str],
    model: Optional[str],
    system_prompt: Optional[str],
):
    """Create a new conversation fixture template.

    NAME is the name of the fixture to create.

    Example:

        conversation-fixture new greeting_test -d "Test greeting flow"
    """
    fixture = create_fixture_template(name, description, model)

    if system_prompt:
        fixture.system_prompt = system_prompt

    output_path = Path(output) if output else Path(f"{name}.yaml")
    save_fixture(fixture, output_path)

    click.echo(f"Created fixture template: {output_path}")
    click.echo(f"Edit the file to add your conversation turns and expectations.")


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "-m",
    "--model",
    help="Model to use (overrides fixture and env default).",
)
@click.option(
    "--api-url",
    envvar="OPENAI_API_BASE",
    help="API base URL (default: OpenAI).",
)
@click.option(
    "--api-key",
    envvar="OPENAI_API_KEY",
    help="API key.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output directory for run results.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Don't actually call the API.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show detailed output.",
)
@click.option(
    "--timeout",
    type=float,
    default=60.0,
    help="Request timeout in seconds.",
)
def run(
    path: str,
    model: Optional[str],
    api_url: Optional[str],
    api_key: Optional[str],
    output: Optional[str],
    dry_run: bool,
    verbose: bool,
    timeout: float,
):
    """Run conversation fixtures against a model.

    PATH can be a single YAML fixture file or a directory containing fixtures.

    Example:

        conversation-fixture run fixtures/ -m gpt-4 -o results/

        conversation-fixture run test.yaml --dry-run
    """
    # Load fixtures
    fixtures = load_fixtures(path)

    if not fixtures:
        click.echo(f"No fixtures found in: {path}", err=True)
        sys.exit(1)

    click.echo(f"Found {len(fixtures)} fixture(s)")

    # Create runner
    with FixtureRunner(
        api_url=api_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
    ) as runner:
        results = []
        passed = 0
        failed = 0

        for fixture in fixtures:
            if verbose:
                click.echo(f"\nRunning: {fixture.name}")

            try:
                result = runner.run(fixture, model=model, dry_run=dry_run)
                results.append(result)

                if result.passed:
                    passed += 1
                    status = click.style("PASSED", fg="green")
                else:
                    failed += 1
                    status = click.style("FAILED", fg="red")

                click.echo(
                    f"  {fixture.name}: {status} "
                    f"({result.total_latency_ms:.0f}ms)"
                )

                if verbose and not result.passed:
                    for turn in result.turns:
                        if turn.validation and not turn.validation.passed:
                            click.echo(f"    Turn {turn.turn_index}: FAILED")
                            for error in turn.validation.errors:
                                click.echo(f"      - {error}")

            except Exception as e:
                failed += 1
                click.echo(
                    f"  {fixture.name}: "
                    f"{click.style('ERROR', fg='red')} - {e}"
                )

        # Summary
        click.echo(f"\n{'='*40}")
        click.echo(f"Results: {passed} passed, {failed} failed")

        # Save results
        if output:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)

            for result in results:
                timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
                filename = f"{result.fixture_name}_{timestamp}.json"
                save_run_result(result, output_dir / filename)

            click.echo(f"Results saved to: {output_dir}")

        # Exit with failure if any tests failed
        if failed > 0:
            sys.exit(1)


@cli.command()
@click.argument("old_result", type=click.Path(exists=True))
@click.argument("new_result", type=click.Path(exists=True))
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show full content diffs.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output as JSON.",
)
def diff(
    old_result: str,
    new_result: str,
    verbose: bool,
    as_json: bool,
):
    """Compare two fixture run results.

    OLD_RESULT and NEW_RESULT are paths to JSON run result files.

    Example:

        conversation-fixture diff results/old.json results/new.json -v
    """
    old = load_run_result(old_result)
    new = load_run_result(new_result)

    try:
        run_diff = diff_runs(old, new)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(run_diff.to_dict(), indent=2))
    else:
        click.echo(format_diff(run_diff, verbose=verbose))

    # Exit with failure if regression
    if run_diff.regression:
        sys.exit(1)


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "-m",
    "--model",
    help="Model to use.",
)
@click.option(
    "--api-url",
    envvar="OPENAI_API_BASE",
    help="API base URL.",
)
@click.option(
    "--api-key",
    envvar="OPENAI_API_KEY",
    help="API key.",
)
@click.option(
    "--timeout",
    type=float,
    default=60.0,
    help="Request timeout in seconds.",
)
def replay(
    path: str,
    model: Optional[str],
    api_url: Optional[str],
    api_key: Optional[str],
    timeout: float,
):
    """Replay a fixture interactively.

    PATH is a YAML fixture file to replay.

    This command runs the fixture turn by turn, showing responses
    and waiting for user confirmation between turns.

    Example:

        conversation-fixture replay test.yaml -m gpt-4
    """
    fixture = load_fixture(path)

    click.echo(f"Replaying: {fixture.name}")
    if fixture.description:
        click.echo(f"Description: {fixture.description}")
    click.echo(f"Turns: {len(fixture.turns)}")
    click.echo("-" * 40)

    if fixture.system_prompt:
        click.echo(f"\n[System]: {fixture.system_prompt}")

    with FixtureRunner(
        api_url=api_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
    ) as runner:
        responses: dict[int, str] = {}
        effective_model = model or fixture.model or runner.model

        for i, turn in enumerate(fixture.turns):
            if turn.role == "user":
                click.echo(f"\n[User]: {turn.content}")
                click.echo("")

            elif turn.role == "assistant":
                click.echo(f"[Waiting for {effective_model} response...]")

                # Build messages up to this point
                messages = runner._build_messages(fixture, i, responses)

                try:
                    content, usage, latency = runner._call_api(
                        messages, effective_model
                    )
                    responses[i] = content

                    click.echo(f"\n[Assistant] ({latency:.0f}ms):")
                    click.echo(content)

                    # Validate if expectations exist
                    if turn.expect:
                        from conversation_fixture.validator import validate_turn

                        validation = validate_turn(content, turn)

                        if validation.passed:
                            click.echo(
                                click.style("\n[Validation: PASSED]", fg="green")
                            )
                        else:
                            click.echo(
                                click.style("\n[Validation: FAILED]", fg="red")
                            )
                            for error in validation.errors:
                                click.echo(f"  - {error}")

                except Exception as e:
                    click.echo(click.style(f"\n[Error]: {e}", fg="red"))
                    if not click.confirm("Continue?", default=False):
                        sys.exit(1)

            elif turn.role == "system":
                click.echo(f"\n[System]: {turn.content}")

            # Wait for user to continue (except on last turn)
            if i < len(fixture.turns) - 1:
                if not click.confirm("\nContinue to next turn?", default=True):
                    click.echo("Replay stopped.")
                    return

    click.echo("\n" + "=" * 40)
    click.echo("Replay complete.")


@cli.command("list")
@click.argument("path", type=click.Path(exists=True))
def list_fixtures(path: str):
    """List fixtures in a directory.

    PATH is a directory containing fixture files.

    Example:

        conversation-fixture list fixtures/
    """
    fixtures = load_fixtures(path)

    if not fixtures:
        click.echo(f"No fixtures found in: {path}")
        return

    click.echo(f"Found {len(fixtures)} fixture(s):\n")

    for fixture in sorted(fixtures, key=lambda f: f.name):
        click.echo(f"  {fixture.name}")
        if fixture.description:
            click.echo(f"    {fixture.description}")
        click.echo(f"    Turns: {len(fixture.turns)}")
        if fixture.model:
            click.echo(f"    Model: {fixture.model}")
        click.echo()


@cli.command()
@click.argument("path", type=click.Path(exists=True))
def validate(path: str):
    """Validate fixture file(s) without running.

    PATH is a fixture file or directory.

    Example:

        conversation-fixture validate fixtures/
    """
    path_obj = Path(path)
    errors = []

    if path_obj.is_file():
        files = [path_obj]
    else:
        files = list(path_obj.glob("**/*.yaml")) + list(path_obj.glob("**/*.yml"))

    for file in files:
        try:
            fixture = load_fixture(file)
            click.echo(f"  {click.style('OK', fg='green')} {file}")

            # Check for common issues
            if not fixture.turns:
                click.echo(f"    Warning: No turns defined")
            else:
                user_turns = [t for t in fixture.turns if t.role == "user"]
                assistant_turns = [t for t in fixture.turns if t.role == "assistant"]

                if not user_turns:
                    click.echo(f"    Warning: No user turns")
                if not assistant_turns:
                    click.echo(f"    Warning: No assistant turns")

                # Check for assistant turns without expectations
                for i, turn in enumerate(fixture.turns):
                    if turn.role == "assistant" and not turn.expect:
                        click.echo(
                            f"    Warning: Turn {i} (assistant) has no expectations"
                        )

        except Exception as e:
            click.echo(f"  {click.style('FAIL', fg='red')} {file}")
            click.echo(f"    {e}")
            errors.append((file, e))

    click.echo()
    if errors:
        click.echo(f"Validation failed: {len(errors)} error(s)")
        sys.exit(1)
    else:
        click.echo(f"Validation passed: {len(files)} file(s)")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
