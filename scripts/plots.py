import matplotlib.pyplot as plt
import pandas as pd


def plot_event_periods(
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

def _plot_cycle_markers(ax, cycles):
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

def _plot_all_event_periods(ax, cycles):
    """Plot hospitalization, ED visits and contact HCP periods if present."""
    events = [
        ("hospitalizations_date", "darkred", "red", 0.25, "Hospitalization"),
        ("ed_visits_date", "navy", "lightblue", 0.15, "ED Visit"),
        ("contact_hcp_date", "darkorange", "orange", 0.15, "Contact HCP"),
    ]
    for col, line_color, span_color, alpha, label in events:
        plot_event_periods(ax, cycles.get(col, None),
                           line_color=line_color, span_color=span_color,
                           alpha=alpha, label=label)

def plot_rhr_over_time(fig, ax, rhr_data, cycles, label):
    ax.plot(
        rhr_data["dateTime"],
        rhr_data["value"],
        label=label,
        alpha=1,
        color="orange",
        marker="o",
        markersize=4,
    )

    
    
    _plot_cycle_markers(ax, cycles)
    _plot_all_event_periods(ax, cycles)
    plot_weekends_spans(ax, rhr_data)

def plot_hrv_over_time(fig, ax, hrv_data, cycles, part_ids):
    mask = hrv_data["rmssd"].notna()
    ax.plot(
        hrv_data["dateTime"][mask],
        hrv_data["rmssd"][mask],
        label=f"Participant {part_ids}",
        alpha=1,
        color="green",
        marker="o",
        markersize=4,
    )

    _plot_cycle_markers(ax, cycles)
    _plot_all_event_periods(ax, cycles)
    plot_weekends_spans(ax, hrv_data)

def plot_steps_over_time(fig, ax, steps_data, cycles, part_ids):
    # plot per-day cumulative step traces, label only on first traced day
    labeled = False
    for _, day_data in steps_data.sort_values("dateTime").groupby("date"):
        ax.plot(
            day_data["dateTime"],
            day_data["cumsum"],
            label=f"Participant {part_ids}" if not labeled else None,
            alpha=0.4,
            color="blue",
        )
        labeled = True

    # plot moving averages (2..5) if present; label only once
    # for window, alpha in [(2, 0.25), (3, 0.4), (4, 0.55), (5, 0.7)]:
    #     col = f"daily_steps_ma{window}"
    #     if col in steps_data.columns:
    #         daily_plot = (
    #             steps_data[["date", col]]
    #             .drop_duplicates()
    #             .sort_values("date")
    #         )
    #         ax.plot(
    #             daily_plot["date"],
    #             daily_plot[col],
    #             label=f"Participant {part_ids} ({window}-day MA)" if not labeled else None,
    #             color="orange",
    #             alpha=alpha,
    #         )
    #         labeled = True

    _plot_cycle_markers(ax, cycles)
    _plot_all_event_periods(ax, cycles)
    plot_weekends_spans(ax, steps_data)
