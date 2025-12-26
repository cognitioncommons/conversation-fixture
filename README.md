# Conversation Fixture

Multi-turn conversation test fixture manager for LLMs.

Define conversation test fixtures in YAML, run them against language models, validate responses against expectations, and compare runs to detect regressions.

## Installation

```bash
pip install conversation-fixture
```

Or install from source:

```bash
pip install -e .
```

## Quick Start

### 1. Create a Fixture

Create a YAML file defining your conversation test:

```yaml
name: greeting_test
description: Test basic greeting flow
model: gpt-4

turns:
  - role: user
    content: "Hello, how are you?"

  - role: assistant
    expect:
      contains: ["hello", "hi"]
      not_contains: ["error", "sorry"]

  - role: user
    content: "Can you help me with Python?"

  - role: assistant
    expect:
      contains: ["python", "help"]
      min_length: 50
```

Or generate a template:

```bash
conversation-fixture new my_test -d "Test description"
```

### 2. Run the Fixture

```bash
# Run with environment variables
export OPENAI_API_KEY="your-key"
conversation-fixture run greeting_test.yaml -m gpt-4

# Run with options
conversation-fixture run fixtures/ --api-key sk-xxx -o results/
```

### 3. Compare Runs

```bash
conversation-fixture diff results/old.json results/new.json -v
```

## Fixture Format

```yaml
name: string                    # Required: fixture name
description: string             # Optional: description
model: string                   # Optional: default model
system_prompt: string           # Optional: system message

turns:                          # Required: list of conversation turns
  - role: user                  # "user", "assistant", or "system"
    content: string             # Message content (for user/system)
    name: string                # Optional: turn name
    expect:                     # Expectations (for assistant turns)
      contains: [string]        # Must contain all (case-insensitive)
      not_contains: [string]    # Must not contain any
      starts_with: string       # Must start with
      ends_with: string         # Must end with
      matches: string           # Regex pattern to match
      min_length: int           # Minimum response length
      max_length: int           # Maximum response length
      json_schema:              # JSON schema for response
        type: object
        required: [field1]

metadata:                       # Optional: arbitrary metadata
  author: string
  tags: [string]
```

## CLI Commands

### new

Create a new fixture template:

```bash
conversation-fixture new my_fixture -d "Description" -m gpt-4
conversation-fixture new complex_flow --system-prompt "You are a helpful assistant"
```

### run

Execute fixtures against a model:

```bash
# Run single fixture
conversation-fixture run test.yaml -m gpt-4

# Run all fixtures in directory
conversation-fixture run fixtures/ -o results/

# Dry run (no API calls)
conversation-fixture run fixtures/ --dry-run

# Verbose output
conversation-fixture run test.yaml -v
```

Options:
- `-m, --model`: Model to use
- `--api-url`: API base URL (default: OpenAI)
- `--api-key`: API key
- `-o, --output`: Save results to directory
- `--dry-run`: Skip API calls
- `-v, --verbose`: Detailed output
- `--timeout`: Request timeout (seconds)

### diff

Compare two run results:

```bash
conversation-fixture diff old.json new.json
conversation-fixture diff old.json new.json -v    # Verbose
conversation-fixture diff old.json new.json --json  # JSON output
```

### replay

Interactive fixture replay:

```bash
conversation-fixture replay test.yaml -m gpt-4
```

Step through turns with validation feedback.

### list

List fixtures in a directory:

```bash
conversation-fixture list fixtures/
```

### validate

Validate fixture files:

```bash
conversation-fixture validate fixtures/
```

## Python API

```python
from conversation_fixture import (
    ConversationFixture,
    FixtureRunner,
    load_fixture,
    validate_response,
    diff_runs,
)

# Load a fixture
fixture = load_fixture("test.yaml")

# Run against a model
with FixtureRunner(model="gpt-4") as runner:
    result = runner.run(fixture)
    print(result.summary())

    if not result.passed:
        for turn in result.turns:
            if turn.validation and not turn.validation.passed:
                print(f"Turn {turn.turn_index} failed:")
                for error in turn.validation.errors:
                    print(f"  - {error}")

# Compare runs
old_result = load_run_result("old.json")
new_result = load_run_result("new.json")
diff = diff_runs(old_result, new_result)

if diff.regression:
    print("Regression detected!")
```

## Expectations Reference

### contains

Check that response contains all specified strings (case-insensitive):

```yaml
expect:
  contains:
    - "hello"
    - "help"
```

### not_contains

Check that response does not contain forbidden strings:

```yaml
expect:
  not_contains:
    - "error"
    - "cannot"
```

### starts_with / ends_with

Check response prefix/suffix:

```yaml
expect:
  starts_with: "Hello"
  ends_with: "?"
```

### matches

Check against a regex pattern:

```yaml
expect:
  matches: "\\d{4}-\\d{2}-\\d{2}"  # Date pattern
```

### min_length / max_length

Constrain response length:

```yaml
expect:
  min_length: 100
  max_length: 500
```

### json_schema

Validate JSON responses:

```yaml
expect:
  json_schema:
    type: object
    required:
      - name
      - value
```

## Environment Variables

- `OPENAI_API_KEY`: API key for OpenAI-compatible APIs
- `OPENAI_API_BASE`: Base URL for API (default: `https://api.openai.com/v1`)
- `OPENAI_MODEL`: Default model (default: `gpt-3.5-turbo`)

## License

MIT
