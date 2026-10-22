import logging
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from loaders import ParticipantDataLoader
from parsers import align_metric_against_ref, get_moving_mean_valid, normalize_part_ids
from preprocessing import Preprocessor, load_plot_data, merge_cadence_with_hr


def _plot_event_periods(
    ax: plt.Axes,
    periods: Any,
    line_color: str = "red",
    span_color: str = "red",
    line_style: str = "--",
    alpha: float = 0.2,
    label: str | None = None,
) -> None:
    """Draw event markers and optional shaded spans on a Matplotlib axis.

    Args:
        ax (plt.Axes): Axis to draw the event annotations on.
        periods (Any): Event date(s) or event intervals to plot.
        line_color (str): Color used for the event line and label. Defaults to "red".
        span_color (str): Fill color for time spans. Defaults to "red".
        line_style (str): Style for the vertical event marker lines. Defaults to "--".
        alpha (float): Transparency for filled spans. Defaults to 0.2.
        label (str | None): Label to display for the event. Defaults to None.

    Returns:
        None: Plots directly onto the provided axis.
    """
    first = True

    def _plot_marker(date: pd.Timestamp) -> None:
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
            "Contact HCP": 0.05,
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

    def _plot_span(start: pd.Timestamp, end: pd.Timestamp) -> None:
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

        if isinstance(period, pd.Timestamp):
            _plot_marker(period)
        elif (
            isinstance(period, list)
            and len(period) == 2
            and all(isinstance(x, pd.Timestamp) for x in period)
        ):
            for date in period:
                _plot_marker(date)
        elif isinstance(period, list):
            for item in period:
                if isinstance(item, pd.Timestamp):
                    _plot_marker(item)
                elif (
                    isinstance(item, list)
                    and len(item) == 2
                    and all(isinstance(x, pd.Timestamp) for x in item)
                ):
                    _plot_span(*item)

def plot_weekends_spans(ax: plt.Axes, steps_data: pd.DataFrame) -> None:
    """Shade weekend periods on a plot based on the available timestamps.

    Args:
        ax (plt.Axes): Axis to add the weekend shading to.
        steps_data (pd.DataFrame): DataFrame containing a dateTime column.

    Returns:
        None: Draws weekend spans directly on the supplied axis.
    """
    data = steps_data.copy()
    data["dateTime"] = pd.to_datetime(data["dateTime"], errors="coerce")

    unique_dates = (
        data["dateTime"]
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
    )

    weekend_labeled = False
    for date in unique_dates:
        if date.weekday() == 5:
            ax.axvspan(
                date,
                date + pd.Timedelta(days=2),
                color="gray",
                alpha=0.1,
                label="Weekend" if not weekend_labeled else None,
            )
            weekend_labeled = True

def plot_cycle_markers(ax: plt.Axes, cycles: dict[str, Any]) -> None:
    """Plot vertical cycle-change markers and label them on the axis.

    Args:
        ax (plt.Axes): Axis to annotate with cycle markers.
        cycles (dict[str, Any]): Cycle metadata including Date and Period entries.

    Returns:
        None: Draws cycle lines and labels directly on the axis.
    """
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

def plot_all_event_periods(ax: plt.Axes, cycles: dict[str, Any]) -> None:
    """Plot hospitalization, ED visit, and HCP contact periods from cycle metadata.

    Args:
        ax (plt.Axes): Axis to plot event annotations on.
        cycles (dict[str, Any]): Dictionary containing event date columns.

    Returns:
        None: Draws all event markers and spans to the supplied axis.
    """
    events = [
        ("hospitalizations_date", "darkred", "red", 0.25, "Hospitalization"),
        ("ed_visits_date", "navy", "lightblue", 0.15, "ED Visit"),
        ("contact_hcp_date", "darkorange", "orange", 0.15, "Contact HCP"),
    ]
    for col, line_color, span_color, alpha, label in events:
        _plot_event_periods(ax, cycles.get(col, None),
                           line_color=line_color, span_color=span_color,
                           alpha=alpha, label=label)

def plot_info(ax: plt.Axes, cycles: dict[str, Any], data: pd.DataFrame) -> None:
    """Draw cycle markers, clinical events, and weekend spans in one plot layer.

    Args:
        ax (plt.Axes): Axis to annotate.
        cycles (dict[str, Any]): Cycle and event metadata.
        data (pd.DataFrame): Source data used for weekend shading.

    Returns:
        None: Adds all plot annotations to the axis.
    """
    plot_cycle_markers(ax, cycles)
    plot_all_event_periods(ax, cycles)
    plot_weekends_spans(ax, data)

def plot_rhr(ax: plt.Axes, rhr_data: pd.DataFrame, label: str, style: dict[str, Any] | None = None, window: int = 3) -> None:
    """Plot resting heart rate while keeping missing days from creating false gaps.

    Args:
        ax (plt.Axes): Axis to plot on.
        rhr_data (pd.DataFrame): DataFrame with dateTime and value columns.
        label (str): Legend label for the series.
        style (dict[str, Any] | None): Matplotlib plotting style dictionary. Defaults to None.
        window (int): Moving-average window size. Defaults to 3.

    Returns:
        None: Plots the smoothed RHR series into the provided axis.
    """
    if style is None:
        style = {
            "color": "orange",
            "marker": "o",
            "markersize": 4,
            "alpha": 1,
        }

    valid_rhr = rhr_data.loc[rhr_data["value"].notna(), ["dateTime", "value"]].copy()
    valid_rhr["dateTime"] = pd.to_datetime(valid_rhr["dateTime"], errors="coerce")
    valid_rhr = valid_rhr.dropna(subset=["dateTime", "value"]).sort_values("dateTime").reset_index(drop=True)

    if not valid_rhr.empty:
        full_index = pd.date_range(
            valid_rhr["dateTime"].min(),
            valid_rhr["dateTime"].max(),
            freq="D",
        )
        valid_rhr = (
            valid_rhr.set_index("dateTime")
            .reindex(full_index)
            .rename_axis("dateTime")
            .reset_index()
        )

    smoothed = get_moving_mean_valid(valid_rhr, window)
    smoothed_series = smoothed[window].reset_index(drop=True)

    ax.plot(
        valid_rhr["dateTime"],
        smoothed_series,
        label=label,
        **style
    )

def plot_hrv_over_time(ax: plt.Axes, hrv_data: pd.DataFrame, label: str, window: int, style: dict[str, Any] | None = None) -> None:
    """Plot RMSSD over time while skipping long periods of missing data.

    Args:
        ax (plt.Axes): Axis to plot on.
        hrv_data (pd.DataFrame): DataFrame with dateTime and rmssd columns.
        label (str): Legend label for the HRV series.
        window (int): Moving-average window size.
        style (dict[str, Any] | None): Matplotlib plotting style dictionary. Defaults to None.

    Returns:
        None: Plots the smoothed HRV series to the supplied axis.
    """
    if style is None:
        style = {
            "color": "green",
            "marker": "o",
            "markersize": 4,
            "alpha": 1,
        }

    mask = hrv_data["rmssd"].notna()
    valid_hrv = hrv_data.loc[mask, ["dateTime", "rmssd"]].copy()
    valid_hrv["dateTime"] = pd.to_datetime(valid_hrv["dateTime"], errors="coerce")
    valid_hrv = valid_hrv.dropna(subset=["dateTime", "rmssd"]).sort_values("dateTime").reset_index(drop=True)

    if not valid_hrv.empty:
        full_index = pd.date_range(
            valid_hrv["dateTime"].min(),
            valid_hrv["dateTime"].max(),
            freq="D",
        )
        valid_hrv = (
            valid_hrv.set_index("dateTime")
            .reindex(full_index)
            .rename_axis("dateTime")
            .reset_index()
        )

    smoothed = get_moving_mean_valid(valid_hrv, window)
    smoothed_series = smoothed[window].reset_index(drop=True)

    ax.plot(
        valid_hrv["dateTime"],
        smoothed_series,
        label=label,
        **style
    )

def plot_steps_over_time(ax: plt.Axes, steps_data: pd.DataFrame, label: str, style: dict[str, Any] | None = None) -> None:
    """Plot cumulative daily step traces across the time span of the data.

    Args:
        ax (plt.Axes): Axis to plot on.
        steps_data (pd.DataFrame): DataFrame containing dateTime and cumsum columns.
        label (str): Legend label for the participant or trace.
        style (dict[str, Any] | None): Matplotlib plotting style options. Defaults to None.

    Returns:
        None: Adds the step traces to the supplied axis.
    """
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

def plot_steps_moving_average(ax: plt.Axes, steps_data: pd.DataFrame, windows: list[int] = [5], style: dict[str, Any] | None = None) -> None:
    """Plot one or more moving-average step trends for the daily step series.

    Args:
        ax (plt.Axes): Axis to plot on.
        steps_data (pd.DataFrame): DataFrame containing daily step columns such as daily_steps_ma5.
        windows (list[int]): Moving-average windows to plot. Defaults to [5].
        style (dict[str, Any] | None): Extra matplotlib styling to apply to each line. Defaults to None.

    Returns:
        None: Draws step moving-average lines into the supplied axis.
    """
    if style is None:
        style = {
            "linewidth": 2,
            "alpha": 0.7,
        }

    style_for_plot = style.copy()
    cmap_or_color = style_for_plot.pop("color", mpl.colormaps.get_cmap("Blues"))

    n_windows = len(windows)
    min_frac, max_frac = 0.35, 0.9
    for idx, window in enumerate(windows):
        frac = idx / max(1, n_windows - 1) if n_windows > 1 else 0.5
        mapped_frac = min_frac + (max_frac - min_frac) * frac
        col = f"daily_steps_ma{window}"
        if col in steps_data.columns:
            daily_plot = (
                steps_data[["date", col]]
                .drop_duplicates()
                .sort_values("date")
                .drop_duplicates(subset="date", keep="last")
                .set_index("date")
            )

            if daily_plot.empty:
                continue

            full_index = pd.date_range(daily_plot.index.min(), daily_plot.index.max(), freq="D")
            daily_plot = daily_plot.reindex(full_index)

            ax.plot(
                daily_plot.index,
                daily_plot[col],
                label=f"{window}-day steps moving mean",
                color=cmap_or_color(mapped_frac) if callable(cmap_or_color) else cmap_or_color,
                **style_for_plot
            )
        else:
            logging.warning(f"Column '{col}' not found in steps_data. Skipping moving average plot for window {window}."
                            f"available mov_averages: {steps_data.columns.str.contains('daily_steps_ma')}")

    ax.set_ylabel("Steps")

def plot_hr_zones(ax: plt.Axes, part_heart_data: pd.DataFrame, zone_thresholds: dict[str, float], cmap: Any = None) -> plt.Axes:
    """Draw horizontal heart-rate zones as colored bands on the provided axis.

    Args:
        ax (plt.Axes): Primary axis to draw the zone bands on.
        part_heart_data (pd.DataFrame): Heart-rate data containing a bpm column.
        zone_thresholds (dict[str, float]): Mapping of heart-rate zone names to upper bounds.
        cmap (Any): Colormap used for zone coloring. Defaults to None.

    Returns:
        plt.Axes: Secondary axis created for zone labels.
    """
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

def plot_hr_cadence(ax: plt.Axes, data_merge: pd.DataFrame, window_size: int, zones_to_omit: tuple[int, ...] = (1, 4, 5), cmap: Any = None, style: dict[str, Any] | None = None) -> None:
    """Plot median cadence by heart-rate zone, smoothed over time.

    Args:
        ax (plt.Axes): Axis to draw the cadence traces on.
        data_merge (pd.DataFrame): Merged heart-rate and cadence data.
        window_size (int): Moving-average window used for the cadence smoothing.
        zones_to_omit (tuple[int, ...]): Zone numbers to skip from plotting. Defaults to (1, 4, 5).
        cmap (Any): Colormap for per-zone line colors. Defaults to None.
        style (dict[str, Any] | None): Matplotlib style arguments for each line. Defaults to None.

    Returns:
        None: Draws all cadence lines on the axis.
    """
    if style is None:
        style = {
            "linewidth": 2,
            "linestyle": "-",
            "alpha": 1,
            "marker": "x",
            "markersize": 6,
        }

    if cmap is None:
        cmap = LinearSegmentedColormap.from_list(
            "custom",
            [
                (0.0, "#004D29"),
                (0.1, "#33ff00"),
                (0.3, "#C96100"),
                (1.0, "#570014"),
            ]
        )

    data_merge["date"] = pd.to_datetime(data_merge["dateTime"], errors="coerce").dt.date
    median_cadence_by_zone = (
        data_merge.groupby(["date", "zone"])["cadence"]
        .median()
        .reset_index()
    )
    median_cadence_by_zone = median_cadence_by_zone.sort_values(["zone", "date"])

    smoothed_frames = []
    for _, group in median_cadence_by_zone.groupby("zone", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        group = group.drop_duplicates(subset="date", keep="last")
        smoothed = get_moving_mean_valid(group[["date", "cadence"]], [window_size])[window_size]
        if hasattr(smoothed, "to_numpy"):
            group["cadence"] = smoothed.to_numpy()
        else:
            group["cadence"] = smoothed
        smoothed_frames.append(group)

    median_cadence_by_zone = pd.concat(smoothed_frames, ignore_index=True)

    for i, (zone, group) in enumerate(median_cadence_by_zone.groupby("zone")):
        zone_number = pd.to_numeric(str(zone).replace("zone", ""), errors="coerce")
        if pd.notna(zone_number) and zone_number in zones_to_omit:
            continue

        zone_series = group.sort_values("date").drop_duplicates(subset="date", keep="last").set_index("date")["cadence"]
        if zone_series.empty:
            continue

        full_index = pd.date_range(zone_series.index.min(), zone_series.index.max(), freq="D")
        zone_series = zone_series.reindex(full_index)

        frac = i / max(1, len(median_cadence_by_zone["zone"].unique()) - 1)
        ax.plot(zone_series.index, zone_series.values, label=f"{window_size}-day mov avg: {zone}", color=cmap(frac), **style)

def plot_hr_over_time(ax: plt.Axes, hr_data: pd.DataFrame, label: str, style: dict[str, Any] | None = None) -> None:
    """Plot heart-rate values over time for a participant.

    Args:
        ax (plt.Axes): Axis to draw the heart-rate series on.
        hr_data (pd.DataFrame): DataFrame with dateTime and bpm columns.
        label (str): Legend label for the line.
        style (dict[str, Any] | None): Matplotlib style options. Defaults to None.

    Returns:
        None: Adds the heart-rate trace to the axis.
    """
    if style is None:
        style = {
            "color": "red",
            "alpha": 1,
        }

    mask = hr_data["bpm"].notna()
    ax.plot(
        hr_data["dateTime"][mask],
        hr_data["bpm"][mask],
        label=label,
        **style
    )

def plot_zone_ratios(ax: plt.Axes, data_merge: pd.DataFrame, cmap: Any = None) -> None:
    """Plot the ratio of time spent in each heart-rate zone by day.

    Args:
        ax (plt.Axes): Axis to plot the stacked zone ratios on.
        data_merge (pd.DataFrame): DataFrame containing dateTime and zone columns.
        cmap (Any): Colormap for the stacked areas. Defaults to None.

    Returns:
        None: Draws the zone-ratio stack plot on the axis.
    """
    if cmap is None:
        cmap = mpl.colormaps.get_cmap("RdYlGn_r")

    data_merge["date"] = data_merge["dateTime"].dt.date
    zone_counts = (
        data_merge.groupby(["date", "zone"])
        .size()
        .unstack(fill_value=0)
    )

    zone_ratios = zone_counts.div(zone_counts.sum(axis=1), axis=0)
    zones = list(zone_ratios.columns)
    colors = [cmap(i / max(1, len(zones) - 1)) for i in range(len(zones))]

    ax.stackplot(
        zone_ratios.index,
        [zone_ratios[z].values for z in zones],
        labels=zones,
        colors=colors,
        alpha=0.6,
    )
    ax.set_ylim(zone_ratios.iloc[:, 0].min() - 0.05, 1)

def plot_step_zone_ratios(ax: plt.Axes, data_merge: pd.DataFrame, cmap: Any = None) -> None:
    """Plot total daily steps stacked by heart-rate zone.

    Args:
        ax (plt.Axes): Axis to plot the stacked step totals on.
        data_merge (pd.DataFrame): Heart-rate/step data with dateTime, value, and zone columns.
        cmap (Any): Colormap to use for stacked areas. Defaults to None.

    Returns:
        None: Draws the daily step budget by zone.
    """
    if cmap is None:
        cmap = LinearSegmentedColormap.from_list(
            "custom",
            [
                (0.0, "#004D29"),
                (0.3, "#21a700"),
                (0.7, "#BB5A00"),
                (1.0, "#570014"),
            ]
        )

    data = data_merge.copy()
    if data.empty:
        return

    data["date"] = pd.to_datetime(data["dateTime"], errors="coerce").dt.normalize()
    data["value"] = pd.to_numeric(data.get("value", 0), errors="coerce").fillna(0)

    zone_totals = (
        data.groupby(["date", "zone"], dropna=False)["value"]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )

    if zone_totals.empty:
        return

    zones = list(zone_totals.columns)
    colors = [cmap(i / max(1, len(zones) - 1)) for i in range(len(zones))]

    ax.stackplot(
        zone_totals.index,
        [zone_totals[z].to_numpy() for z in zones],
        labels=zones,
        colors=colors,
        alpha=0.6,
    )

    ax.set_ylabel("Daily steps")
    ax.set_ylim(bottom=zone_totals.sum(axis=1).min() - 100)
    ax.margins(x=0.01)

def create_figure_field(part_ids: list[int]) -> tuple[plt.Figure, np.ndarray]:
    """Create a grid of axes for a small set of participant panels.

    Args:
        part_ids (list[int]): Participant identifiers to include in the figure.

    Returns:
        tuple[plt.Figure, np.ndarray]: Figure and axes array for the generated panel layout.
    """
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

def create_figure_panel(part_ids: list[int], nrows: int = 2) -> tuple[plt.Figure, np.ndarray]:
    """Create a panel layout with one column per participant.

    Args:
        part_ids (list[int]): Participant identifiers to include in the panel.
        nrows (int): Number of rows in the figure layout. Defaults to 2.

    Returns:
        tuple[plt.Figure, np.ndarray]: Figure and axes grid for the participant panel.
    """
    n_parts = len(part_ids)
    ncols = n_parts

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(10 * ncols, 5 * nrows),
        squeeze=False,
        sharex="col"
    )

    return fig, axes

def set_legend(ax: plt.Axes, ax2: plt.Axes | None = None, loc_primary: str = "upper left", loc_secondary: str = "upper right") -> None:
    """Attach legends to a primary axis and an optional secondary axis.

    Args:
        ax (plt.Axes): Primary axis with legend handles.
        ax2 (plt.Axes | None): Secondary axis to add a legend for. Defaults to None.
        loc_primary (str): Position of the primary legend. Defaults to "upper left".
        loc_secondary (str): Position of the secondary legend. Defaults to "upper right".

    Returns:
        None: Sets the legends on the provided axes.
    """
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc=loc_primary)

    if ax2 is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        if handles2:
            ax2.legend(handles2, labels2, loc=loc_secondary)

def _plot_participant_panel(ax: plt.Axes, part_id: int, hrv_data: pd.DataFrame, steps_data: pd.DataFrame, cycles: pd.DataFrame) -> plt.Axes:
    """Create a compact participant panel with HRV and step trends.

    Args:
        ax (plt.Axes): Axis to plot the panel on.
        part_id (int): Participant identifier.
        hrv_data (pd.DataFrame): HRV measurements.
        steps_data (pd.DataFrame): Step counts for the same participant.
        cycles (pd.DataFrame): Cycle metadata used to annotate periods.

    Returns:
        plt.Axes: Secondary axis created for the step trace.
    """
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
    part_ids = [11, 12, 13, 14, 16, 17, 18, 20]
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
