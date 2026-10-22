import matplotlib.pyplot as plt
import pandas as pd
import warnings
import dateutil.parser
import logging
from preprocessing import load_plot_data, add_side_effect_column, get_before_and_after_chemo, get_baseline_data

def z_score(series: pd.Series) -> pd.Series:
    """Calculate the z-score of a pandas Series.

    Args:
        series (pd.Series): Input data to standardize.

    Returns:
        pd.Series: Standardized values with zero mean and unit variance.
    """
    return (series - series.mean()) / series.std()


def z_score_diff(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """Calculate the z-score of the difference between two series.

    Args:
        series1 (pd.Series): First series to compare.
        series2 (pd.Series): Second series to compare.

    Returns:
        pd.Series: Z-score difference between the two series.
    """
    return z_score(series1) - z_score(series2)


def get_hr_zones(hr_data: pd.DataFrame, cycle_data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Categorize heart rate into zones based on baseline threshold values.

    Args:
        hr_data (pd.DataFrame): Heart-rate records containing a dateTime and bpm column.
        cycle_data (pd.DataFrame): Cycle metadata used to identify the baseline window.

    Returns:
        tuple[pd.DataFrame, dict[str, float]]: The heart-rate data with a zone column appended and the zone thresholds used.
    """
    if cycle_data.empty:
        raise ValueError("Cycle data is empty. Cannot determine heart rate zones without cycle information.")

    required_hr_columns = {"dateTime", "bpm"}
    if not required_hr_columns.issubset(hr_data.columns):
        raise ValueError(f"hr_data must contain the columns {sorted(required_hr_columns)}.")

    if not {"Period", "Date"}.issubset(cycle_data.columns):
        raise ValueError("cycle_data must contain the columns 'Period' and 'Date'.")

    hr_data = hr_data.copy()
    baseline = get_baseline_data(hr_data, "dateTime", cycle_data) 

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

def mixed_effects_model(data: pd.DataFrame):
    """Fit a mixed-effects model to the data.

    Args:
        data (pd.DataFrame): The dataset containing the dependent and independent variables.

    Returns:
        MixedLMResults: The fitted mixed-effects model results.
    """
    import statsmodels.formula.api as smf
    df = data.get("steps").copy()
    # Reduce to one row per participant per day (sum daily steps)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["date"] = df["date"].dt.normalize()

    df = (
        df.groupby(["participant_id", "date", "days_after_chemo"], as_index=False)["value"]
        .sum()
    )

    df = df.loc[
        df["days_after_chemo"].between(-3, 7)
    ].copy()
    
    df.loc[
        df["days_after_chemo"].between(-3, -1),
        "period"
    ] = "baseline"

    df.loc[
        df["days_after_chemo"].eq(0),
        "period"
    ] = "administration_day"

    df.loc[
        df["days_after_chemo"].between(1, 3),
        "period"
    ] = "post_1_3"

    df.loc[
        df["days_after_chemo"].between(4, 7),
        "period"
    ] = "post_4_7"

    # Keep participants with baseline and post-treatment RHR data.
    valid = df.dropna(subset=["participant_id", "value"])

    has_baseline = set(
        valid.loc[valid["period"].eq("baseline"), "participant_id"]
    )

    has_post = set(
        valid.loc[
            valid["period"].isin(["post_1_3", "post_4_7"]),
            "participant_id"
        ]
    )
    
    logging.info(f"Participants with baseline data: {has_baseline}")
    logging.info(f"Participants with post-treatment data: {has_post}")
    logging.info(f"Participants with both baseline and post-treatment data: {has_baseline & has_post}")

    analysis = valid.loc[
        valid["participant_id"].isin(has_baseline & has_post)
    ].copy()

    pat_ids = analysis["participant_id"].unique()
    print(pat_ids)
    model = smf.mixedlm('value ~ C(period, Treatment(reference="baseline"))', 
                        data=analysis, 
                        groups=analysis["participant_id"], 
                        re_formula="1")
    result =    model.fit()
    return result

if __name__ == "__main__":
    pat_ids = list(range(1, 31))
    pat_ids, data = load_plot_data(pat_ids, data_types=("steps", "cycles")) 
    data = add_side_effect_column(pat_ids, data)

    before_after_chemo_data = get_before_and_after_chemo(data)

    results = mixed_effects_model(before_after_chemo_data)
    print(results.summary())
    print(len(pat_ids))

