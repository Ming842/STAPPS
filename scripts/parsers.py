"""Parsers for different data formats."""

import csv
import json

from typing import Any
from pathlib import Path

def parse_csv(file_path: Path) -> list[dict[str, Any]]:
    """Parse a CSV file and return its contents as a list of dictionaries."""
    with file_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)

def parse_json(file_path: Path) -> dict[str, Any]:
    """Parse a JSON file and return its contents as a dictionary."""
    with file_path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)

def parse_txt(file_path: Path) -> str:
    """Parse a TXT file and return its contents as a string."""
    with file_path.open("r", encoding="utf-8-sig") as f:
        return f.read()
