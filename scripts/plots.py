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

        ax.text(
            date,
            0.6 if label == "Hospitalization" else 0.3,
            label,
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="bottom",
            ha="right",
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
    # Ensure 'dateTime' is a datetime object
    steps_data['dateTime'] = pd.to_datetime(steps_data['dateTime'])

    # Get unique dates from the steps data
    unique_dates = steps_data['dateTime'].dt.date.unique()

    for date in unique_dates:
        # Check if the date is a Saturday or Sunday
        if pd.Timestamp(date).weekday() == 5:  # 5: Saturday, 6: Sunday
            ax.axvspan(
                pd.Timestamp(date),
                pd.Timestamp(date) + pd.Timedelta(days=2),
                color='gray',
                alpha=0.1,
                label='Weekend' if date == unique_dates[0] else None
            )

def plot_hrv_over_time(fig, ax, hrv_data, cycles, part_ids):
    ax.plot(
        hrv_data['dateTime'],
        hrv_data['rmssd'],
        label=f'Participant {part_ids}',
        alpha = 0.4,
        color='green'
    )
    for date, period in zip(cycles['Date'], cycles['Period']):
        if period == "Baseline":
            continue
        if date == "Dagelijks":
            continue

        ax.axvline(x=date, color='red', linestyle='--', alpha=0.5,
                label=str(period))
        ax.text(date, 0.8, f"Cycle: {period}",
                transform=ax.get_xaxis_transform(),
                rotation=90, va='bottom', ha='right', color='red')

    plot_event_periods(
        ax,
        cycles["hospitalizations_date"],
        line_color="darkred",
        span_color="red",
        alpha=0.25,
        label="Hospitalization"
    )

    plot_event_periods(
        ax,
        cycles["ed_visits_date"],
        line_color="navy",
        span_color="lightblue",
        alpha=0.15,
        label="ED Visit"
    )

    plot_event_periods(
        ax,
        cycles["contact_hcp_date"],
        line_color="darkorange",
        span_color="orange",
        alpha=0.15,
        label="Contact HCP"
    )

    plot_weekends_spans(ax, hrv_data)


def plot_steps_over_time(fig, ax, steps_data, cycles, part_ids):
    for i, (_, day_data) in enumerate(steps_data.sort_values('dateTime').groupby('date')):
        ax.plot(
            day_data['dateTime'],
            day_data['cumsum'],
            label=f'Participant {part_ids}' if i == 0 else None,
            alpha = 0.4,
            color='blue'
        )
    daily_plot = (
        steps_data[["date", "daily_steps_ma3"]]
        .drop_duplicates()
        .sort_values("date")
    )

    for window, alpha in [(2, 0.25), (3, 0.4), (4, 0.55), (5, 0.7)]:
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
                label=f"Participant {part_ids} ({window}-day MA)" if i == 0 else None,
                color="orange",
                alpha=alpha,
            )
    for date, period in zip(cycles['Date'], cycles['Period']):
        if period == "Baseline":
            continue
        if date == "Dagelijks":
            continue
        ax.axvline(x=date, color='red', linestyle='--', alpha=0.5,
                label=str(period))
        ax.text(date, 0.8, f"Cycle: {period}",
                transform=ax.get_xaxis_transform(),
                rotation=90, va='bottom', ha='right', color='red')
    plot_event_periods(
        ax,
        cycles["hospitalizations_date"],
        line_color="darkred",
        span_color="red",
        alpha=0.25,
        label="Hospitalization"
    )

    plot_event_periods(
        ax,
        cycles["ed_visits_date"],
        line_color="navy",
        span_color="lightblue",
        alpha=0.15,
        label="ED Visit"
    )

    plot_event_periods(
        ax,
        cycles["contact_hcp_date"],
        line_color="darkorange",
        span_color="orange",
        alpha=0.15,
        label="Contact HCP"
    )

    plot_weekends_spans(ax, steps_data)


