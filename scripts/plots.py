import logging

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

from preprocessing import Preprocessor
from loaders import ParticipantDataLoader
from parsers import normalize_part_ids, align_metric_against_ref

def _plot_event_periods(
    ax,
    periods,
    line_color="red",
    span_color="red",
    line_style="--",
    alpha=0.2,
    label=None,
):
    first = True

    def _plot_marker(date):
        nonlocal first

        ax.axvline(
            date,
            color=line_color,
            linestyle=line_style,
            label=label if first else None,
        )
        label_height = {
            "Hospitalization": 0.7,
            "ED Visit": 0.8,
            "Contact HCP": 0.05
        }
        ax.text(
            date,
            label_height.get(label, 0),
            label,
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="bottom",
            ha="left" if label == "Hospitalization" else "right",
            color=line_color,
        )

        first = False

    def _plot_span(start, end):
        ax.axvspan(
            start,
            end,
            color=span_color,
            alpha=alpha,
        )
        _plot_marker(start)

    for period in periods:

        if period is None:
            continue

        # Single date
        if isinstance(period, pd.Timestamp):
            _plot_marker(period)

        # [date1, date2] -> two separate markers
        elif (
            isinstance(period, list)
            and len(period) == 2
            and all(isinstance(x, pd.Timestamp) for x in period)
        ):
            for date in period:
                _plot_marker(date)

        # Handle nested structures
        elif isinstance(period, list):

            for item in period:

                # [date, [start, end]]
                if isinstance(item, pd.Timestamp):
                    _plot_marker(item)

                # [[start, end]] -> span
                elif (
                    isinstance(item, list)
                    and len(item) == 2
                    and all(isinstance(x, pd.Timestamp) for x in item)
                ):
                    _plot_span(*item)

def plot_weekends_spans(ax, steps_data):
    """Plot shaded spans for weekends in the steps data."""
    data = steps_data.copy()
    data["dateTime"] = pd.to_datetime(data["dateTime"], errors="coerce")

    # Remove invalid timestamps first to avoid NaT comparison errors
    unique_dates = (
        data["dateTime"]
        .dropna()
        .dt.normalize()      # keep as Timestamp at midnight
        .drop_duplicates()
        .sort_values()
    )

    weekend_labeled = False
    for date in unique_dates:
        if date.weekday() == 5:  # Saturday
            ax.axvspan(
                date,
                date + pd.Timedelta(days=2),
                color="gray",
                alpha=0.1,
                label="Weekend" if not weekend_labeled else None,
            )
            weekend_labeled = True

def plot_hr_zones(ax, part_heart_data, zone_thresholds, cmap=None):
    """Plot horizontal bands for heart rate zones on the given axis."""
    
    if cmap is None:
        cmap = mpl.colormaps.get_cmap("RdYlGn_r")
    finite_bpm = part_heart_data["bpm"].dropna()
    max_bpm = finite_bpm.max() if not finite_bpm.empty else 200.0

    lower = 0
    num_zones = len(zone_thresholds)
    zone_tick_positions = []
    zone_tick_labels = []

    ax_zone = ax.twinx()

    for idx, (key, upper) in enumerate(zone_thresholds.items()):
        upper_val = max_bpm if not np.isfinite(upper) else float(upper)
        frac = idx / max(1, num_zones - 1)

        ax.axhspan(lower, upper_val, xmin=0, xmax=1, alpha=0.5, color=cmap(frac))

        if key != "zero":
            ax.axhline(y=upper_val, color="black", linestyle="--", linewidth=0.5)
            zone_tick_positions.append(upper_val)
            zone_tick_labels.append(f"{key} ({upper_val:.0f})")

        lower = upper_val

    ax_zone.set_ylim(ax.get_ylim())
    ax_zone.set_yticks(zone_tick_positions)
    ax_zone.set_yticklabels(zone_tick_labels, fontsize=8)

    return ax_zone

def plot_hr_cadence(ax, data_merge, window_size, cmap=None):
    """Plot cadence against heart rate zones on the given axis."""
    if cmap is None:
        cmap = mpl.colormaps.get_cmap("RdYlGn_r")

    # Group by HR zone and plot median cadence
    data_merge["date"] = data_merge["dateTime"].dt.date
    median_cadence_by_zone = (
        data_merge.groupby(["date", "zone"])["cadence"]
        .median()
        .reset_index()
    )
    median_cadence_by_zone = median_cadence_by_zone.sort_values(["zone", "date"])
    median_cadence_by_zone["cadence"] = (
        median_cadence_by_zone.groupby("zone")["cadence"]
        .transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    )

    for i, (zone, group) in enumerate(median_cadence_by_zone.groupby("zone")):

        frac = i / max(1, len(median_cadence_by_zone["zone"].unique()) - 1)
        ax.plot(group["date"], group["cadence"], label=zone, color=cmap(frac))

def plot_cycle_markers(ax, cycles):
    """Plot vertical lines and labels for cycle change points."""
    dates = pd.to_datetime(cycles.get("Date", []), errors="coerce")
    periods = cycles.get("Period", [])
    labeled = False
    for date, period in zip(dates, periods):
        if pd.isna(date) or period in ("Baseline", "Dagelijks"):
            continue
        ax.axvline(x=date, color="red", linestyle="--", alpha=0.5, label="Chemo cycle" if not labeled else None)
        ax.text(date, 0.8, f"Cycle: {period}", transform=ax.get_xaxis_transform(),
                rotation=90, va="bottom", ha="right", color="red")
        labeled = True

def plot_all_event_periods(ax, cycles):
    """Plot hospitalization, ED visits and contact HCP periods if present."""
    events = [
        ("hospitalizations_date", "darkred", "red", 0.25, "Hospitalization"),
        ("ed_visits_date", "navy", "lightblue", 0.15, "ED Visit"),
        ("contact_hcp_date", "darkorange", "orange", 0.15, "Contact HCP"),
    ]
    for col, line_color, span_color, alpha, label in events:
        _plot_event_periods(ax, cycles.get(col, None),
                           line_color=line_color, span_color=span_color,
                           alpha=alpha, label=label)

def plot_info(ax, cycles, data):
    """Plot cycle markers and event periods on the given axis."""
    plot_cycle_markers(ax, cycles)
    plot_all_event_periods(ax, cycles)
    plot_weekends_spans(ax, data)

def plot_rhr(ax, rhr_data, label, style=None):
    if style is None:
        style = {
            "color": "orange",
            "marker": "o",
            "markersize": 4,
            "alpha": 1,
        }

    ax.plot(
        rhr_data["dateTime"],
        rhr_data["value"],
        label=label,
        **style
    )
    
def plot_hrv_over_time(ax, hrv_data, label, style=None):
    if style is None:
        style = {
                "color": "green",
                "marker": "o",
                "markersize": 4,
                "alpha": 1,
            } 

    mask = hrv_data["rmssd"].notna()
    ax.plot(
        hrv_data["dateTime"][mask],
        hrv_data["rmssd"][mask],
        label=label,
        **style
    )

def plot_steps_over_time(ax, steps_data, label, style=None):
    """Plot cumulative steps over time on the given axis."""
    # plot per-day cumulative step traces, label only on first traced day
    if style is None:
        style = {
            "color": "blue",
            "alpha": 0.4,
        }
    labeled = False
    for _, day_data in steps_data.sort_values("dateTime").groupby("date"):
        ax.plot(
            day_data["dateTime"],
            day_data["cumsum"],
            label=label if not labeled else None,
            **style
        )
        labeled = True

def plot_steps_moving_average(ax, steps_data, windows: list[int]=[5], style=None):
    """Plot moving averages of steps data on the given axis."""

    if style is None:
        style = {
            "color": mpl.colormaps.get_cmap("Oranges"), 
            "alpha": 0.7,
        }
    n_windows = len(windows)
    for idx, window in enumerate(windows):
        frac = idx / max(1, n_windows - 1)
        col = f"daily_steps_ma{window}"
        if col in steps_data.columns:
            daily_plot = (
                steps_data[["date", col]]
                .drop_duplicates()
                .sort_values("date")
            )
            ax.plot(
                daily_plot["date"],
                daily_plot[col],
                label=f"{window}-day MA",
                color=style.get("color")(frac) if callable(style.get("color")) else style.get("color"),
                alpha=style.get("alpha"),
            )
        else:
            logging.warning(f"Column '{col}' not found in steps_data. Skipping moving average plot for window {window}."
                            f"available mov_averages: {steps_data.columns.str.contains('daily_steps_ma')}")

def load_plot_data(part_ids, data_types=None):
    participant_loader = ParticipantDataLoader()
    preprocessor = Preprocessor()

    part_ids = normalize_part_ids(part_ids)

    valid_types = {"rhr", "hrv", "hr", "steps", "cycles"}
    if data_types is None:
        logging.info("No data types specified. Defaulting to all available types.", UserWarning)
        data_types = ("rhr", "hrv", "hr", "steps", "cycles")
    elif isinstance(data_types, str):
        data_types = (data_types,)

    invalid = set(data_types) - valid_types
    if invalid:
        raise ValueError(f"Unsupported data types: {sorted(invalid)}\n Supported types: {sorted(valid_types)}")

    payload = {}

    if "rhr" in data_types:
        payload["rhr"] = preprocessor.process_resting_heart_rate(part_id=part_ids)
    if "hrv" in data_types:
        payload["hrv"] = preprocessor.process_heart_rate_variability(part_id=part_ids)
    if "hr" in data_types:
        payload["hr"] = preprocessor.process_heart_rate(part_id=part_ids)
    if "steps" in data_types:
        payload["steps"] = preprocessor.process_step_count(part_id=part_ids)
    if "cycles" in data_types:
        payload["cycles"] = participant_loader.cycle_data(part_id=part_ids)

    # Backward-compatible default return shape
    if data_types == ("rhr", "hrv", "steps", "cycles"):
        return (
            part_ids,
            payload.get("rhr"),
            payload.get("hrv"),
            payload.get("hr"),
            payload.get("steps"),
            payload.get("cycles"),
        )

    return part_ids, payload

def create_figure_field(part_ids):
    n_parts = len(part_ids)
    nrows = 1 if n_parts == 1 else 2
    ncols = (n_parts + nrows - 1) // nrows

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(10 * ncols, 5 * nrows),
        squeeze=False,
    )
    if n_parts >= 1:
        axes = axes.ravel()

    for ax in axes[n_parts:]:
        ax.set_visible(False)

    return fig, axes

def create_figure_panel(part_ids, nrows=2):
    n_parts = len(part_ids)
    ncols = n_parts      # one column per participant

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(10 * ncols, 5 * nrows),
        squeeze=False, 
        sharex="col"
    )

    return fig, axes

def set_legend(ax, ax2=None, loc_primary="upper left", loc_secondary="upper right"):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper left")
    
    if ax2 is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        if handles2:
            ax2.legend(handles2, labels2, loc="upper right")

def _plot_participant_panel(ax, part_id, hrv_data, steps_data, cycles):
    part_hrv_data = hrv_data[hrv_data["participant_id"] == part_id].copy()
    part_hrv_data.loc[part_hrv_data["rmssd"] == 0, "rmssd"] = pd.NA
    part_cycles = cycles[cycles["participant_id"] == part_id]

    plot_hrv_over_time(ax, part_hrv_data, part_cycles, part_id)
    ax.set_zorder(2)

    ax2 = ax.twinx()
    part_steps_data = steps_data[steps_data["participant_id"] == part_id]
    ax2.plot(
        part_steps_data["date"],
        part_steps_data["daily_steps_ma5"],
        alpha=0.7,
        color="blue",
    )
    ax2.set_ylabel("Steps")



    plt.grid()
    ax.set_title(f"Participant {part_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("HRV (RMSSD)")

    return ax2


if __name__ == "__main__":
    part_ids = [11, 12, 13, 14, 16, 17, 18, 20]  # List of participant IDs to process
    part_ids, payload = load_plot_data(part_ids, data_types=("hrv", "steps", "cycles"))

    hrv_data = payload.get("hrv")
    steps_data = payload.get("steps")
    cycles = payload.get("cycles")

    fig, axes = create_figure_field(part_ids)

    for index, part_id in enumerate(part_ids):
        _plot_participant_panel(axes[index], part_id, hrv_data, steps_data, cycles)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
