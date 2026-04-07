"""Safe JSON file loading."""

import json
from pathlib import Path


def load_json_file(path: Path) -> dict | list:
    """Load and parse a JSON file.

    Raises ValueError with file path context on parse errors.
    """
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
