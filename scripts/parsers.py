"""Parsers for different data formats."""

import csv
import json
import re

import pandas as pd

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


def parse_dates(text):
    """Parse dates from a given text and return them as a list of datetime objects."""
    date_pat = r"\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}:\d{2})?"
    pattern = rf"({date_pat})\s*(?:tm|t\s*/\s*m)\s*({date_pat})|({date_pat})"
    if pd.isna(text):
        return None

    result = []

    for m in re.finditer(pattern, str(text), flags=re.IGNORECASE):
        if m.group(1):  # range
            result.append([
                pd.to_datetime(m.group(1)),
                pd.to_datetime(m.group(2))
            ])
        else:  # single date
            if isinstance(text, pd.Timestamp):
                result.append(text)
            else:
                result.append(
                    pd.to_datetime(m.group(3))
                )

    if not result:
        return None
    return result