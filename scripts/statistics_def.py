import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging

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
                           merge_cadence_with_hr,
                           get_smoothed_cadence_by_zone)
import statsmodels.formula.api as smf

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
    value_column: str | None = None,
    aggregation: str | Callable | None = None,
) -> pd.DataFrame:
    """Prepare data for a Cox proportional hazards model.

    Args:
        data: Mapping of data type names to data frames.
        data_type: Key in ``data`` identifying the metric to prepare.
        value_column: Measurement column. Defaults to ``value`` for steps and
            resting heart rate, ``rmssd`` for HRV, and ``bpm`` for heart rate.
        aggregation: Aggregation used when multiple records occur on one
            participant-day. Defaults to ``"sum"`` for steps and ``"mean"``
            for other metrics.

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
        patient_mask = daily["participant_id"].eq(pat_id)
        patient_dates = daily.loc[patient_mask, "date"]
        cycle_data_patient = cycle_data[cycle_data["participant_id"] == pat_id].copy()
        cycle_dates = cycle_data_patient["Date"].dropna().unique()
        cycle_dates = pd.to_datetime(cycle_dates, errors="coerce").normalize()
        daily.loc[patient_mask, "chemo_cycle"] = patient_dates.isin(cycle_dates)
        for flag, date_column in event_columns.items():
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


    event_column = "hospitalization"

    daily["value_recent"] = (
        daily.groupby("participant_id")["value"]
        .transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).median()
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
        value_column: str | None = None,
        aggregation: str | Callable | None = None,):
    """Placeholder for future implementation of a Cox proportional hazards model."""

    df = prepare_cox_model_data(
        data, data_type, cycle_data,value_column, aggregation
    )
    return df

if __name__ == "__main__":
    # pat_ids = list(range(1, 31))
    # pat_ids, data = load_plot_data(pat_ids, data_types=("hr", "steps", "cycles"))
    # data = add_side_effect_column(pat_ids, data)
    # steps_data = data["steps"]
    # hr_data = data["hr"]
    # cycles = data["cycles"]

    # hr_data_zoned, zone_thresholds = get_hr_zones(hr_data, cycles)
    # smoothed_cadence_by_zone = []
    # for pat_id in pat_ids:
    #     patient_steps = steps_data.loc[
    #         steps_data["participant_id"].eq(pat_id)
    #     ]
    #     patient_hr = hr_data_zoned.loc[
    #         hr_data_zoned["participant_id"].eq(pat_id)
    #     ]

    #     data_merge = merge_cadence_with_hr(
    #         patient_steps[["dateTime", "cadence"]], patient_hr
    #     )
    #     patient_smoothed = get_smoothed_cadence_by_zone(
    #         data_merge, window_size=3
    #     )
    #     patient_smoothed["participant_id"] = pat_id
    #     smoothed_cadence_by_zone.append(patient_smoothed)

    # smoothed_cadence_by_zone = pd.concat(
    #     smoothed_cadence_by_zone, ignore_index=True
    # )
    # smoothed_cadence_by_zone = smoothed_cadence_by_zone[
    #     ["participant_id"]
    #     + [
    #         column
    #         for column in smoothed_cadence_by_zone.columns
    #         if column != "participant_id"
    #     ]
    # ]

    # for zone in range(1, 6):
    #     zone_name = f"zone{zone}"
    #     column_name = f"smoothed_cadence_by_{zone_name}"
    #     cadence_data = smoothed_cadence_by_zone[
    #         smoothed_cadence_by_zone["zone"].eq(zone_name)
    #     ]
    #     cadence_data = cadence_data[["participant_id", "date", "cadence"]].copy()
    #     data[column_name] = cadence_data

    data = load_from_pickle(data_keys=["rhr",
                                        "steps",
                                        "hrv",
                                        "smoothed_cadence_by_zone1",
                                        "smoothed_cadence_by_zone2",
                                        "smoothed_cadence_by_zone3",
                                        "smoothed_cadence_by_zone4",
                                        "smoothed_cadence_by_zone5",
                                        "cycles"])


    before_after_chemo_data = get_before_and_after_chemo(data, no_ae_only=False)
    cox_input = cox_model(data, data_type="rhr", cycle_data=data["cycles"])