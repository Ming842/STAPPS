"""Parsers for different data formats."""

import csv
import json
import re
import logging
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from config import DATA_DIR

def parse_csv(file_path: Path) -> list[dict[str, Any]]:
    """Parse a CSV file and return its contents as a list of dictionaries.

    Args:
        file_path (Path): Path to the CSV file to parse.

    Returns:
        list[dict[str, Any]]: Rows from the CSV file keyed by column name.
    """
    with file_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader)

def parse_json(file_path: Path) -> dict[str, Any]:
    """Parse a JSON file and return its contents as a dictionary.

    Args:
        file_path (Path): Path to the JSON file to parse.

    Returns:
        dict[str, Any]: Parsed JSON object.
    """
    with file_path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)

def parse_txt(file_path: Path) -> str:
    """Parse a text file and return its contents as a string.

    Args:
        file_path (Path): Path to the text file to parse.

    Returns:
        str: File contents as a string.
    """
    with file_path.open("r", encoding="utf-8-sig") as f:
        return f.read()
    
def get_timestamp_column(df):
    """Identify the timestamp column in a DataFrame."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df)}")

    for col in ["dateTime", "Date", "date", "timestamp", "time"]:
        if col in df.columns:
            return col

    raise KeyError(
        f"No timestamp column found. Available columns: {list(df.columns)}"
    )

def normalize_part_ids(part_ids: int | str | Sequence[int | str]) -> list[int | str]:
    """Normalize participant identifiers into a list.

    Args:
        part_ids (int | str | Sequence[int | str]): A single participant ID or a sequence of IDs.

    Returns:
        list[int | str]: A list containing the normalized participant IDs.
    """
    path = Path(DATA_DIR)
    if isinstance(part_ids, (int, str)):
        part_ids = [part_ids]

    available_pt_ids: list[int] = []

    for item in path.iterdir():
        m = re.match(r"^ST(\d+)", item.name)
        if m:
            available_pt_ids.append(int(m.group(1)))

    part_ids_final = []
    for pt_id in part_ids:
        if pt_id in available_pt_ids:
            part_ids_final.append(pt_id)
        else:
            logging.warning(f"Participant ID {pt_id} not found in data directory.")
    return list(part_ids_final)

def parse_dates(text: Any) -> list[Any] | None:
    """Parse date strings from a value and return the parsed datetime objects.

    Args:
        text (Any): A scalar or text value containing one or more dates.

    Returns:
        list[Any] | None: Parsed date values or ranges, or None when no dates can be parsed.
    """
    date_pat = r"\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}:\d{2})?"
    pattern = rf"({date_pat})\s*(?:tm|t\s*/\s*m)\s*({date_pat})|({date_pat})"
    if pd.isna(text):
        return None

    result: list[Any] = []

    for m in re.finditer(pattern, str(text), flags=re.IGNORECASE):
        if m.group(1):
            result.append([
                pd.to_datetime(m.group(1)),
                pd.to_datetime(m.group(2))
            ])
        else:
            if isinstance(text, pd.Timestamp):
                result.append(text)
            else:
                result.append(pd.to_datetime(m.group(3)))

    if not result:
        return None
    return result

def align_metric_against_ref(
    metric_df: pd.DataFrame,
    value_date: str,
    value_col: str,
    ref_df: pd.DataFrame,
    ref_date: str,
    ref_col: str,
) -> pd.DataFrame:
    """Align one metric DataFrame to a reference DataFrame using a common datetime column.

    Args:
        metric_df (pd.DataFrame): DataFrame containing the metric values.
        value_date (str): Column name for the metric timestamps.
        value_col (str): Column name for the metric values.
        ref_df (pd.DataFrame): Reference DataFrame used to align the metric against.
        ref_date (str): Column name for the reference timestamps.
        ref_col (str): Column name for the reference values.

    Returns:
        pd.DataFrame: Metric rows aligned by datetime to the reference data.
    """
    aligned_df = (
        metric_df[[value_date, value_col]]
        .assign(dateTime=lambda df: pd.to_datetime(df[value_date]))
        .merge(
            ref_df[[ref_date, ref_col]]
            .assign(dateTime=lambda df: pd.to_datetime(df[ref_date]))[["dateTime", ref_col]]
            .dropna()
            .drop_duplicates(subset="dateTime"),
            on="dateTime",
            how="inner",
        )
        .sort_values("dateTime")
    )
    return aligned_df

