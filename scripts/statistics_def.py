import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import statsmodels.formula.api as smf

from lifelines.exceptions import ConvergenceError
from numpy.linalg import LinAlgError
from math import log
from tqdm import tqdm
from config import DATA_DIR
from pathlib import Path
from loaders import load_from_pickle
from collections.abc import Callable
from parsers import get_timestamp_column
from preprocessing import (load_plot_data,
                           add_side_effect_column,
                           get_before_and_after_chemo,
                           get_baseline_data,
                           get_baseline_stats,
                           get_smoothed_cadence_by_zone)
from lifelines import CoxTimeVaryingFitter
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

def prepare_mixed_effects_data(
    data: dict[str, pd.DataFrame],
    data_type: str = "steps",
    value_column: str | None = None,
    aggregation: str | Callable | None = None,
) -> pd.DataFrame:
    """Prepare daily observations for a mixed-effects model.

    The measurements loaded by :mod:`preprocessing` use different value
    columns.  This helper provides one common preprocessing path for steps,
    HRV, heart rate, resting heart rate, and future metrics.

    Args:
        data: Mapping of data type names to data frames.
        data_type: Key in ``data`` identifying the metric to prepare.
        value_column: Measurement column. Defaults to ``value`` for steps and
            resting heart rate, ``rmssd`` for HRV,``bpm`` for heart rate and
            ``cadence`` for smoothed cadence.

        aggregation: Aggregation used when multiple records occur on one
            participant-day. Defaults to ``"sum"`` for steps and ``"mean"``
            for other metrics.

    Returns:
        A daily data frame containing ``participant_id``, ``date``, ``value``,
        ``days_after_chemo``, and ``period``.

    Raises:
        ValueError: If the metric or required columns are missing.
    """
    default_value_columns = {
        "steps": "value",
        "hrv": "rmssd",
        "hr": "bpm",
        "rhr": "value",
        "smoothed_cadence_by_zone1": "cadence",
        "smoothed_cadence_by_zone2": "cadence",
        "smoothed_cadence_by_zone3": "cadence",
        "smoothed_cadence_by_zone4": "cadence",
        "smoothed_cadence_by_zone5": "cadence",
    }
    default_aggregations = {"steps": "sum"}
    if data_type not in data:
        raise ValueError(f"Data does not contain the metric {data_type!r}.")

    metric = data[data_type].copy()
    value_column = value_column or default_value_columns.get(data_type, "value")
    aggregation = aggregation or default_aggregations.get(data_type, "median")
    required_columns = {"participant_id", "days_after_chemo", value_column}
    missing_columns = required_columns - set(metric.columns)
    if missing_columns:
        raise ValueError(
            f"{data_type!r} data is missing required columns: {sorted(missing_columns)}."
        )

    metric_timestamp = get_timestamp_column(metric)
    date_source = metric[metric_timestamp]

    metric["date"] = pd.to_datetime(date_source, errors="coerce").dt.normalize()
    metric["value"] = pd.to_numeric(metric[value_column], errors="coerce")
    metric["days_after_chemo"] = pd.to_numeric(
        metric["days_after_chemo"], errors="coerce"
    )
    metric = metric.dropna(
        subset=["participant_id", "date", "days_after_chemo", "value"]
    )
    metric = metric.loc[metric["days_after_chemo"].between(-3, 7)]

    daily = (
        metric.groupby(
            ["participant_id", "date", "days_after_chemo"], as_index=False
        )["value"]
        .agg(aggregation)
    )
    daily["period"] = pd.NA
    daily.loc[daily["days_after_chemo"].between(-3, -1), "period"] = "baseline"
    daily.loc[daily["days_after_chemo"].eq(0), "period"] = "administration_day"
    daily.loc[daily["days_after_chemo"].between(1, 3), "period"] = "post_1_3"
    daily.loc[daily["days_after_chemo"].between(4, 7), "period"] = "post_4_7"
    return daily

def mixed_effects_model(
    data: dict[str, pd.DataFrame],
    data_type: str = "steps",
    value_column: str | None = None,
    aggregation: str | Callable | None = None,
    ratios: bool = False,
):
    """Fit a mixed-effects model to the data for a given metric.

    Args:
        data: Mapping of data type names to data frames.
        data_type: Key in ``data`` identifying the metric to analyze.
        value_column: Measurement column. Defaults to ``value`` for steps and
            resting heart rate, ``rmssd`` for HRV, and ``bpm`` for heart rate.
        aggregation: Aggregation used when multiple records occur on one
            participant-day. Defaults to ``"sum"`` for steps and ``"mean"``
            for other metrics.
        ratios: If True, compute log ratios relative to participant-specific
            baseline median values.

    Returns:
        A fitted mixed-effects model object.
    """
    df = prepare_mixed_effects_data(
        data, data_type, value_column, aggregation
    )

    valid = df.dropna(subset=["participant_id", "value"])

    has_baseline = set(
        valid.loc[valid["period"].eq("baseline"), "participant_id"]
    )
    has_post = set(
        valid.loc[
            valid["period"].isin(["post_1_3", "post_4_7"]),
            "participant_id",
        ]
    )
    included_ids = has_baseline & has_post

    logging.info("Participants with baseline data: %d", len(has_baseline))
    logging.info("Participants with post-treatment data: %d", len(has_post))
    logging.info(
        "Participants with both baseline and post-treatment data: %d",
        len(included_ids),
    )

    analysis = valid.loc[
        valid["participant_id"].isin(included_ids)
    ].copy()

    if ratios:
        baseline_values = (
            analysis.loc[analysis["period"].eq("baseline")]
            .groupby("participant_id", as_index=False)["value"]
            .median()
            .rename(columns={"value": "baseline_value"})
        )

        analysis = analysis.merge(
            baseline_values,
            on="participant_id",
            how="left",
            validate="many_to_one",
        )

        positive = (
            analysis["value"].gt(0)
            & analysis["baseline_value"].gt(0)
        )

        dropped = int((~positive).sum())
        dropped_ids = (
            analysis.loc[~positive, "participant_id"].unique().tolist()
        )

        if dropped:
            logging.warning(
                "Dropping %d observations for participants %s because log "
                "ratios require positive values and positive baselines.",
                dropped,
                dropped_ids,
            )

        analysis = analysis.loc[positive].copy()

        analysis["log_ratio_pct"] = (
            (
                np.log(analysis["value"])
                - np.log(analysis["baseline_value"])
            )
            * 100
        )

        # Keep baseline observations. Their log ratio is exactly 0 and
        # allows "baseline" to be used as the model reference level.
        formula = (
            "log_ratio_pct ~ "
            'C(period, Treatment(reference="baseline"))'
        )
    else:
        formula = (
            'value ~ C(period, Treatment(reference="baseline"))'
        )

    model = smf.mixedlm(
        formula,
        data=analysis,
        groups=analysis["participant_id"],
    )

    return model.fit()

def prepare_cox_model_data(
    data: dict[str, pd.DataFrame],
    data_type: str,
    cycle_data: pd.DataFrame,
    per_value: int = 1,
    ratios: bool = False,
    event_column: str = "hospitalization",
    value_column: str | None = None,
    aggregation: str | Callable | None = None,
    max_interpolation_gap_days: int | None = 7,
    recent_value_shift: int = 1,
    recent_value_window: int = 3,
    recent_value_min_periods: int = 1,
    recent_value_aggregation: str = "median",
) -> pd.DataFrame:
    #pylint: disable=too-many-locals,too-many-statements
    """Prepare data for a Cox proportional hazards model.

    Args:
        data: Mapping of data type names to data frames.
        data_type: Key in ``data`` identifying the metric to prepare.
        cycle_data: Data frame containing cycle information.
        per_value: Devide the value by this number to get 1000 steps/day for example. Defaults to 1 (no change).
        ratios: If True, calculate ratios instead of absolute values. Defaults to False.
        event_column: Column name for the event of interest. Defaults to
            ``"hospitalization"``.
        value_column: Measurement column. Defaults to ``value`` for steps and
            resting heart rate, ``rmssd`` for HRV, and ``bpm`` for heart rate.
        aggregation: Aggregation used when multiple records occur on one
            participant-day. Defaults to ``"sum"`` for steps and ``"mean"``
            for other metrics.
        max_interpolation_gap_days: Maximum number of days between the closest
            measurements surrounding a missing event date. Use ``None`` for no
            maximum.
        recent_value_shift: Number of observations to shift before calculating
            the recent-value rolling statistic.
        recent_value_window: Number of observations in the rolling window.
        recent_value_min_periods: Minimum number of observations required in a
            rolling window.
        recent_value_aggregation: Statistic used for recent values. Must be ``"median"``
            or ``"mean"``.

    Returns:
        A daily data frame containing ``participant_id``, ``date``, ``value``,
        ``days_after_chemo``, and ``period``.
    """
    default_value_columns = {
            "steps": "value",
            "hrv": "rmssd",
            "hr": "bpm",
            "rhr": "value",
            "smoothed_cadence_by_zone1": "cadence",
            "smoothed_cadence_by_zone2": "cadence",
            "smoothed_cadence_by_zone3": "cadence",
            "smoothed_cadence_by_zone4": "cadence",
            "smoothed_cadence_by_zone5": "cadence",
        }
    default_aggregations = {"steps": "sum", "hrv": "median", "hr": "median", "rhr": "median"}

    if data_type not in data:
        raise ValueError(f"Data does not contain the metric {data_type!r}.")

    if max_interpolation_gap_days is not None and max_interpolation_gap_days < 0:
        raise ValueError("max_interpolation_gap_days must be non-negative or None.")
    if recent_value_shift < 0:
        raise ValueError("recent_value_shift must be non-negative.")
    if recent_value_window <= 0:
        raise ValueError("recent_value_window must be positive.")
    if not 0 <= recent_value_min_periods <= recent_value_window:
        raise ValueError(
            "recent_value_min_periods must be between 0 and recent_value_window."
        )
    if recent_value_aggregation not in {"median", "mean"}:
        raise ValueError("recent_value_aggregation must be 'median' or 'mean'.")

    metric = data[data_type].copy()
    value_column = value_column or default_value_columns.get(data_type, "value")
    aggregation = aggregation or default_aggregations.get(data_type, "median")

    required_columns = {"participant_id",
                        "hospitalization_dates", 
                        "ed_visits_dates", 
                        "cycle_dates", 
                        value_column}
    missing_columns = required_columns - set(metric.columns)
    if missing_columns:
        raise ValueError(
            f"{data_type!r} data is missing required columns: {sorted(missing_columns)}."
        )

    # Get baseline statistics for the specified data type and aggregation.
    baseline_stats = get_baseline_stats(data, cycle_data, DATA_TYPE, aggregation, value_column = value_column)

    metric_timestamp = get_timestamp_column(metric)
    date_source = metric[metric_timestamp]

    metric["date"] = pd.to_datetime(date_source, errors="coerce").dt.normalize()
    metric["value"] = pd.to_numeric(metric[value_column], errors="coerce")

    daily = (
        metric.groupby(
            ["participant_id", "date"], as_index=False
        )["value"]
        .agg(aggregation)
    )

    event_columns = {
        "hospitalization": "hospitalization_dates",
        "ed_visit": "ed_visits_dates",
        "cycle_dose_changed": "cycle_dates",
    }

    # Initialize event flags on the aggregated daily rows.
    for flag in event_columns:
        daily[flag] = False

    daily["chemo_cycle"] = False

    for pat_id, patient_data in metric.groupby("participant_id"):

        #PT data for this participant
        patient_mask = daily["participant_id"].eq(pat_id)
        patient_dates = daily.loc[patient_mask, "date"]
        patient_values = (
            daily.loc[patient_mask, ["date", "value"]]
            .dropna(subset=["date"])
            .sort_values("date")
        )

        #Cylce data for this participant
        cycle_data_patient = cycle_data[cycle_data["participant_id"] == pat_id].copy()
        cycle_dates = cycle_data_patient["Date"].dropna().unique()
        cycle_dates = pd.to_datetime(cycle_dates, errors="coerce").normalize()

        event_dates_by_flag = {}
        missing_event_dates = set()

        # Process events for this participant
        for flag, date_column in event_columns.items():
            # Extract and normalize the event dates for this participant
            raw_dates = (
                patient_data[date_column]
                .explode()
                .dropna()
            )

            event_dates = pd.to_datetime(
                raw_dates,
                errors="coerce",
                format="mixed",
            )

            invalid = event_dates.isna()

            if invalid.any():
                fallback_date = patient_dates.max()

                logging.warning(
                    "Unparseable dates for participant %s in column %s: %s. "
                    "Replacing with last measurement date: %s",
                    pat_id,
                    date_column,
                    raw_dates.loc[invalid].unique().tolist(),
                    fallback_date,
                )

                event_dates = event_dates.fillna(fallback_date)

            event_dates = event_dates.dt.normalize().dropna().unique()
            event_dates_by_flag[flag] = event_dates
            missing_event_dates.update(
                set(event_dates) - set(patient_dates.dropna())
            )

        # Add event dates absent from the measurements. Their values are
        # linearly interpolated from the closest measurements on either side.
        if missing_event_dates:
            interpolated_rows = []

            for event_date in sorted(missing_event_dates):
                before = patient_values[patient_values["date"] < event_date].iloc[-1:]
                after = patient_values[patient_values["date"] > event_date].iloc[:1]
                value = np.nan

                if not before.empty and not after.empty:
                    before_date = before.iloc[0]["date"]
                    after_date = after.iloc[0]["date"]
                    gap_days = (after_date - before_date).days

                    if (
                        max_interpolation_gap_days is None
                        or gap_days <= max_interpolation_gap_days
                    ):
                        before_value = before.iloc[0]["value"]
                        after_value = after.iloc[0]["value"]
                        if pd.notna(before_value) and pd.notna(after_value):
                            fraction = (
                                (event_date - before_date).total_seconds()
                                / (after_date - before_date).total_seconds()
                            )
                            value = before_value + fraction * (after_value - before_value)

                interpolated_rows.append(
                    {"participant_id": pat_id, "date": event_date, "value": value}
                )

            daily = pd.concat([daily, pd.DataFrame(interpolated_rows)], ignore_index=True)
            patient_mask = daily["participant_id"].eq(pat_id)
            patient_dates = daily.loc[patient_mask, "date"]

        daily.loc[patient_mask, "chemo_cycle"] = patient_dates.isin(cycle_dates)

        for flag, event_dates in event_dates_by_flag.items():
            daily.loc[patient_mask, flag] = patient_dates.isin(event_dates)

    #make start and stop column for cox model
    daily = daily.sort_values(["participant_id", "date"]).copy()

    daily["interval_start_date"] = daily["date"]
    daily["interval_stop_date"] = (
        daily["date"] + pd.Timedelta(days=1)
    )
    daily["chemo_start"] = (
        daily["date"]
        .where(daily["chemo_cycle"].eq(True))
        .groupby(daily["participant_id"])
        .transform("min")
    )

    daily["start"] = (
        daily["date"] - daily["chemo_start"]
    ).dt.days

    daily["stop"] = daily["start"] + 1
    daily = daily[daily["start"] >= 0].copy()

    if ratios:
        has_baseline = baseline_stats["participant_id"].unique()
        daily = daily[daily["participant_id"].isin(has_baseline)].copy()

        daily = daily.merge(
            baseline_stats[["participant_id", "median"]],
            on="participant_id",
            how="left",
            validate="many_to_one",
        )

        positive = (
            daily["value"].gt(0)
            & daily["median"].gt(0)
        )

        dropped = int((~positive).sum())
        dropped_ids = (
            daily.loc[~positive, "participant_id"].unique().tolist()
        )

        if dropped:
            logging.warning(
                "Dropping %d observations for participants %s because log "
                "ratios require positive values and positive baselines.",
                dropped,
                dropped_ids,
            )

        daily = daily.loc[positive].copy()

        daily["value"] = (
            (
                np.log(daily["value"])
                - np.log(daily["median"])
            )
            * 100
        )

    daily["value_recent"] = (
        daily.groupby("participant_id")["value"]
        .transform(
            lambda x: x.shift(recent_value_shift)
            .rolling(recent_value_window, min_periods=recent_value_min_periods)
            .agg(recent_value_aggregation)
        )
    )

    cox_input = (
        daily[
            [
                "participant_id",
                "start",
                "stop",
                "value_recent",
                event_column,
            ]
        ]
        .rename(columns={event_column: "event"})
        .sort_values(["participant_id", "start", "stop"])
        .copy()
    )

    cox_input["event"] = cox_input["event"].astype(int)
    #adjust for log ratio's
    if ratios:
        per_value = 100 * np.log(1 + per_value)

    cox_input["value_recent"] = cox_input["value_recent"] / per_value

    # Count events strictly BEFORE each interval.
    previous_events = (
        cox_input.groupby("participant_id")["event"].cumsum()
        - cox_input["event"]
    )

    # Keep all pre-event intervals AND the first event interval.
    cox_input = cox_input.loc[previous_events.eq(0)].copy()

    return cox_input

def cox_model(
        data: dict[str, pd.DataFrame],
        data_type: str,
        cycle_data: pd.DataFrame,
        per_value: int = 1,
        ratios: bool = False,
        event_column: str = "hospitalization",
        value_column: str | None = None,
        aggregation: str | Callable | None = None,
        max_interpolation_gap_days: int = 7,
        recent_value_window: int = 3,
        recent_value_min_periods: int = 1,
        recent_value_aggregation: str = "mean"):
        #pylint: disable=too-many-locals,too-many-statements
    """Placeholder for future implementation of a Cox proportional hazards model."""

    df = prepare_cox_model_data(
        data,
        data_type,
        cycle_data,
        per_value,
        ratios = ratios,
        event_column = event_column,
        value_column = value_column,
        aggregation = aggregation,
        max_interpolation_gap_days = max_interpolation_gap_days,
        recent_value_window = recent_value_window,
        recent_value_min_periods = recent_value_min_periods,
        recent_value_aggregation = recent_value_aggregation
        )

    model = CoxTimeVaryingFitter()
    df = df.dropna(subset=["value_recent"]).reset_index(drop=True)
    model.fit(
        df,
        id_col="participant_id",
        start_col="start",
        stop_col="stop",
        event_col="event",
        formula="value_recent",
    )

    return model

if __name__ == "__main__":

    data = load_from_pickle(data_keys=["rhr",
                                        "steps",
                                        "hrv",
                                        "smoothed_cadence_by_zone1",
                                        "smoothed_cadence_by_zone2",
                                        "smoothed_cadence_by_zone3",
                                        "smoothed_cadence_by_zone4",
                                        "smoothed_cadence_by_zone5",
                                        "cycles"])

    DATA_TYPE = "smoothed_cadence_by_zone2"
    cycle_data = data["cycles"]

    datatypes = ["smoothed_cadence_by_zone1",
                 "smoothed_cadence_by_zone2",
                 "smoothed_cadence_by_zone3",
                 "smoothed_cadence_by_zone4",
                 "smoothed_cadence_by_zone5",
                 "cycles"]

    data_filt = {dt: data[dt] for dt in datatypes if dt in data}
    pat_ids =  data_filt["cycles"]["participant_id"].unique()

    data_filt = add_side_effect_column(pat_ids, data_filt)

    for datatype in datatypes:
        if datatypes != "cycles":
            data[datatype] = data_filt[datatype]

    RESULTS_DIR = Path("figures/cox_results")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    settings = {
        "rhr": {
            "ratios": True,
            "per_value": 0.05,
            "value_column": "value"
        },
        "steps": {
            "ratios": False,
            "per_value": 1000,
            "value_column": "value"
        },
        "hrv": {
            "ratios": True,
            "per_value": 0.05,
            "value_column": "rmssd"
        },
        "smoothed_cadence_by_zone1": {
            "ratios": False,
            "per_value": 5,
            "value_column": "cadence"
        },
        "smoothed_cadence_by_zone2": {
            "ratios": False,
            "per_value": 5,
            "value_column": "cadence"
        },
        "smoothed_cadence_by_zone3": {
            "ratios": False,
            "per_value": 5,
            "value_column": "cadence"
        },
        "smoothed_cadence_by_zone4": {
            "ratios": False,
            "per_value": 5,
            "value_column": "cadence"
        },
        "smoothed_cadence_by_zone5": {
            "ratios": False,
            "per_value": 5,
            "value_column": "cadence"
        }
    }

    for DATA_TYPE, params in settings.items():
        for event in ["hospitalization", "ed_visit", "cycle_dose_changed"]:
            try:
                if DATA_TYPE == "hrv":
                    if event == "ed_visit":
                        print("help!")
                model = cox_model(
                    data,
                    ratios=params["ratios"],
                    per_value=params["per_value"],
                    data_type=DATA_TYPE,
                    value_column=params["value_column"],
                    event_column=event,
                    cycle_data=data["cycles"],
                    max_interpolation_gap_days=3
                )

                results.append({
                    "data_type": DATA_TYPE,
                    "event": event,
                    "status": "fitted",
                    "hr": float(model.summary.loc["value_recent", "exp(coef)"]),
                    "hr_lower_95": float(model.summary.loc["value_recent", "exp(coef) lower 95%"]),
                    "hr_upper_95": float(model.summary.loc["value_recent", "exp(coef) upper 95%"]),
                    "p_value": float(model.summary.loc["value_recent", "p"]),
                    "n_events": int(model.event_observed.sum()),
                    "n_periods": int(model._n_examples),
                    "per_value": params["per_value"],
                    "ratios": params["ratios"],
                })

            except (LinAlgError, ConvergenceError, ValueError) as exc:
                logging.warning(
                    "Could not fit Cox model for %s / %s: %s",
                    DATA_TYPE,
                    event,
                    exc,
                )

                results.append({
                    "data_type": DATA_TYPE,
                    "event": event,
                    "status": "not_fitted",
                    "error": str(exc),
                    "per_value": params["per_value"],
                    "ratios": params["ratios"],
                })
    results_df = pd.DataFrame(results)
    results_path = RESULTS_DIR / "cox_model_results.csv"
    results_df.to_csv(results_path, index=False)

    logging.info("Saved %d Cox model results to %s", len(results_df), results_path)
