import matplotlib.pyplot as plt
import pandas as pd
import warnings
import dateutil.parser

from preprocessing import Preprocessor
from loaders import ParticipantDataLoader


def z_score(series) -> pd.Series:
    """Calculate the z-score of a pandas Series."""
    return (series - series.mean()) / series.std()

def z_score_diff(series1, series2)-> pd.Series:
    """Calculate the z-score of the difference of a pandas Series."""
    return z_score(series1)- z_score(series2)

def get_hr_zones(hr_data: pd.DataFrame, cycle_data: pd.DataFrame) -> pd.Series:
    """Categorize heart rate into zones based on thresholds."""
    
    if cycle_data.empty:
        raise ValueError("Cycle data is empty. Cannot determine heart rate zones without cycle information.")

    required_hr_columns = {"dateTime", "bpm"}
    if not required_hr_columns.issubset(hr_data.columns):
        raise ValueError(f"hr_data must contain the columns {sorted(required_hr_columns)}.")

    if not {"Period", "Date"}.issubset(cycle_data.columns):
        raise ValueError("cycle_data must contain the columns 'Period' and 'Date'.")

    hr_data = hr_data.copy()

    c1_rows = cycle_data.loc[cycle_data["Period"] == "C1", "Date"]
    if c1_rows.empty:
        c1_rows = cycle_data.loc[cycle_data["Period"].astype(str).str.startswith("Week"), "Date"]
    if c1_rows.empty:
        raise ValueError("cycle_data does not contain a C1 date.")

    C1_date = c1_rows.iloc[0]
    if C1_date == "Dagelijks":
        end_baseline = pd.to_datetime(hr_data["dateTime"].min()) + pd.Timedelta(days=3)
        warnings.warn("C1 date is 'Dagelijks'. Using the first three days as the baseline.", UserWarning)
    else:
        end_baseline = pd.to_datetime(C1_date, errors="coerce")
        if pd.isna(end_baseline):
            raise ValueError("C1 date could not be parsed as a valid datetime.")

    start_baseline = pd.to_datetime(hr_data["dateTime"].min())

    baseline_days = end_baseline - start_baseline
    if baseline_days <= pd.Timedelta(days=3):
        if "C2" in cycle_data["Period"].values:
            end_baseline = pd.to_datetime(cycle_data.loc[cycle_data["Period"] == "C2", "Date"].iloc[0], errors="coerce")
            if pd.isna(end_baseline):
                raise ValueError("C2 date could not be parsed as a valid datetime.")
            start_baseline = end_baseline - pd.Timedelta(days=3)
            warnings.warn("Baseline period is less than 3 days; using 3 days before C2 as baseline period.", UserWarning)
        else:
            end_baseline = pd.to_datetime(hr_data["dateTime"].min()) + pd.Timedelta(days=3)
            if end_baseline > pd.to_datetime(hr_data["dateTime"].max()):
                raise ValueError("Not enough data to establish a baseline period. Ensure there are at least 3 days of data before C1 or C2.")
            warnings.warn("Baseline period before chemo is less than 3 days; C2 non existent. Using the first three days as baseline period.", UserWarning)

    baseline = hr_data[hr_data["dateTime"] < end_baseline].copy()
    if baseline.empty:
        raise ValueError("No baseline data available before the calculated end date.")

    baseline_stats = {
        "min": baseline["bpm"].min(),
        "median": baseline["bpm"].median(),
        "max": baseline["bpm"].max(),
    }
    max_bpm = baseline_stats["max"]
    if pd.isna(max_bpm) or max_bpm <= 0:
        raise ValueError("Baseline heart rate data must contain positive bpm values.")

    zone_thresholds = {
        "zero": 0,
        "zone1": max_bpm * 0.6,  # Zone 1: <60% of max baseline
        "zone2": max_bpm * 0.7,  # Zone 2: <70% of max baseline
        "zone3": max_bpm * 0.8,  # Zone 3: <80% of max baseline
        "zone4": max_bpm * 0.9,  # Zone 4: <90% of max baseline
        "zone5": float("inf"),   # Zone 5: >=90% of max baseline
    }

    zones = pd.cut(
        hr_data["bpm"],
        bins=list(zone_thresholds.values()),
        labels=list(zone_thresholds.keys())[1:],
        right=False
    )
    hr_data.insert(hr_data.columns.get_loc("bpm") + 1, "zone", zones)
    return hr_data, zone_thresholds


if __name__ == "__main__":
    from plots import load_plot_data

    pat_ids, data = load_plot_data([1], data_types=["steps", "hr", "cycles"])

    steps = data.get("steps")
    hr = data.get("hr")
    cycles = data.get("cycles")

    hr_data, zone_thresholds = get_hr_zones(hr, cycles)
    data_merge = merge_cadence_with_hr(steps[["dateTime", "cadence"]], hr_data)
    # Median cadence per HR zone over time
    data_merge["date"] = data_merge["dateTime"].dt.date
    median_cadence_by_zone = data_merge.groupby(["date","zone"])["cadence"].mean()
    print(median_cadence_by_zone)

