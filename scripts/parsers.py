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

def normalize_part_ids(part_ids):
    if isinstance(part_ids, (int, str)):
        return [part_ids]
    return list(part_ids)

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

def align_metric_against_ref(metric_df, value_date, value_col, ref_df, ref_date, ref_col):
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

def get_moving_mean_valid(data: pd.DataFrame, windows: list[int]) -> dict[int, pd.Series]:
    """Return rolling means for each window, while allowing at most one invalid day.

    The input may be a DataFrame with a ``date``/``dateTime`` column and a numeric
    value column (such as ``cadence`` or ``rmssd``), or a Series. For a Series,
    the index is treated as the date axis when it is datetime-like.
    """
    if isinstance(windows, int):
        windows = [windows]
    else:
        windows = list(windows)

    if isinstance(data, pd.Series):
        series = data.copy()
        value_name = series.name if series.name not in {None, "date", "dateTime"} else "value"
        if isinstance(series.index, pd.DatetimeIndex):
            data = pd.DataFrame({"date": series.index, value_name: series.to_numpy()})
        else:
            data = pd.DataFrame({"date": pd.to_datetime(series.index, errors="coerce"), value_name: series.to_numpy()})
    else:
        data = data.copy()

    if data.empty:
        return {window: pd.Series(dtype=float) for window in windows}

    if "date" not in data.columns:
        if "dateTime" in data.columns:
            data["date"] = pd.to_datetime(data["dateTime"], errors="coerce")
        elif isinstance(data.index, pd.DatetimeIndex):
            data["date"] = data.index
        else:
            data["date"] = pd.to_datetime(data.index, errors="coerce")

    if "cadence" in data.columns:
        value_col = "cadence"
    elif "rmssd" in data.columns:
        value_col = "rmssd"
    elif "value" in data.columns:
        value_col = "value"
    else:
        value_cols = [c for c in data.columns if c not in {"date", "dateTime", "zone"}]
        if not value_cols:
            raise ValueError("No numeric value column found for moving mean computation")
        value_col = value_cols[0]

    data = data[["date", value_col, *(["zone"] if "zone" in data.columns else [])]].copy()
    data = data.sort_values("date").reset_index(drop=True)

    if "zone" in data.columns:
        grouped = []
        for _, zone_df in data.groupby("zone", sort=False):
            zone_df = zone_df.sort_values("date").reset_index(drop=True)
            valid_series = zone_df[value_col].notna()
            zone_output = []
            for window in windows:
                rolling = zone_df[value_col].rolling(window=window, min_periods=1).mean()
                invalid_days = (~valid_series).rolling(window=window, min_periods=1).sum()
                zone_output.append(rolling.where(invalid_days <= 1))
            grouped.append(pd.concat(zone_output, axis=1, keys=windows))
        result = {}
        for window in windows:
            result[window] = pd.concat([g[window] for g in grouped], axis=0)
        return result

    valid_series = data[value_col].notna()
    result = {}
    for window in windows:
        rolling = data[value_col].rolling(window=window, min_periods=1).mean()
        invalid_days = (~valid_series).rolling(window=window, min_periods=1).sum()
        result[window] = rolling.where(invalid_days <= 1)
    return result