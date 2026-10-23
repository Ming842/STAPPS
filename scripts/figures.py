import os
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from plot_defs import (
    plot_hr_cadence,
    plot_hr_over_time,
    plot_hr_zones,
    plot_info,
    plot_rhr,
    plot_step_zone_ratios,
    plot_steps_moving_average,
    plot_steps_over_time,
    plot_zone_ratios,
    set_legend,
    plot_hrv_over_time,
)
from preprocessing import load_plot_data, merge_cadence_with_hr
from scripts.config import FIGURE_FOLDER
from statistics_def import get_hr_zones


def hrv_and_steps(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    """Plot HRV and step traces together with cycle annotations.

    Args:
        ax (plt.Axes): Axis to draw the plot on.
        data (dict[str, pd.DataFrame]): Data dictionary containing steps, HRV, and cycles.

    Returns:
        None: Adds the HRV/steps figure to the provided axis.
    """
    steps_data = data["steps"]
    hrv_data = data["hrv"]
    cycles = data["cycles"]

    plot_hrv_over_time(ax, hrv_data, "HRV", 3, style={"color": "orange",
                                                       "linewidth": 2,
                                                       "linestyle": "-",
                                                       "alpha": 1.0,
                                                       "marker": "o",
                                                       "markersize": 4})

    ax2 = ax.twinx()
    plot_steps_over_time(ax2, steps_data, label="steps", style={"color": "blue", "linewidth": 1, "linestyle": "-", "alpha": 0.3})
    plot_steps_moving_average(ax2, steps_data,
                              style={"linewidth": 2,
                                     "linestyle": "--",
                                     })
    plot_info(ax, cycles, steps_data)

    set_legend(ax, ax2)


def rhr_and_steps(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    """Plot resting heart rate and step traces together.

    Args:
        ax (plt.Axes): Axis to draw the plot on.
        data (dict[str, pd.DataFrame]): Data dictionary containing steps, RHR, and cycles.

    Returns:
        None: Adds the RHR/steps figure to the provided axis.
    """
    steps_data = data["steps"]
    rhr_data = data["rhr"]
    cycles = data["cycles"]

    plot_rhr(ax, rhr_data, label="Resting Heart Rate")
    plot_info(ax, cycles, steps_data)

    ax2 = ax.twinx()
    plot_steps_over_time(ax2, steps_data, label="steps", style={"color": "blue", "linewidth": 1, "linestyle": "-", "alpha": 0.3})
    plot_steps_moving_average(ax2, steps_data,
                              style={"linewidth": 2,
                                     "linestyle": "--",
                                     })
    set_legend(ax, ax2)


def hr_and_zones(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    """Plot heart rate and heart-rate zones together.

    Args:
        ax (plt.Axes): Axis to draw the plot on.
        data (dict[str, pd.DataFrame]): Data dictionary containing steps, HR, and cycles.

    Returns:
        None: Adds the HR/zones figure to the provided axis.
    """
    steps_data = data["steps"]
    hr_data = data["hr"]
    cycles = data["cycles"]

    hr_data_zoned, zone_thresholds = get_hr_zones(hr_data, cycles)
    plot_hr_zones(ax, hr_data_zoned, zone_thresholds)
    plot_hr_over_time(ax, hr_data, label="Heart Rate", style={"linewidth": 1, "linestyle": "-", "alpha": 1})
    plot_info(ax, cycles, steps_data)

    set_legend(ax)


def cadence_and_steps(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    """Plot cadence and step traces together.

    Args:
        ax (plt.Axes): Axis to draw the plot on.
        data (dict[str, pd.DataFrame]): Data dictionary containing steps, HR, and cycles.

    Returns:
        None: Adds the cadence/steps figure to the provided axis.
    """
    steps_data = data["steps"]
    hr_data = data["hr"]
    cycles = data["cycles"]

    hr_data_zoned, zone_thresholds = get_hr_zones(hr_data, cycles)
    data_merge = merge_cadence_with_hr(steps_data[["dateTime", "cadence"]], hr_data_zoned)
    plot_hr_cadence(ax, data_merge, 3)

    ax2 = ax.twinx()
    plot_steps_over_time(ax2, steps_data, label="steps", style={"color": "blue", "linewidth": 1, "linestyle": "-", "alpha": 0.3})
    plot_steps_moving_average(ax2, steps_data, style={"linewidth": 2, "linestyle": "--"})
    plot_info(ax, cycles, steps_data)
    set_legend(ax, ax2)


def zone_ratios_hr(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    """Plot heart-rate zone ratios across the study period.

    Args:
        ax (plt.Axes): Axis to draw the plot on.
        data (dict[str, pd.DataFrame]): Data dictionary containing steps, HR, and cycles.

    Returns:
        None: Adds the HR zone-ratio figure to the provided axis.
    """
    steps_data = data["steps"]
    hr_data = data["hr"]
    cycles = data["cycles"]

    hr_data_zoned, _ = get_hr_zones(hr_data, cycles)
    plot_zone_ratios(ax, hr_data_zoned)
    plot_info(ax, cycles, steps_data)

    ax2 = ax.twinx()
    plot_steps_moving_average(ax2, steps_data, style={"linewidth": 2, "linestyle": "--"})
    set_legend(ax, ax2)


def steps_zone_ratios(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    """Plot total daily steps split by heart-rate zone.

    Args:
        ax (plt.Axes): Axis to draw the plot on.
        data (dict[str, pd.DataFrame]): Data dictionary containing steps, HR, and cycles.

    Returns:
        None: Adds the steps-by-zone figure to the provided axis.
    """
    steps_data = data["steps"]
    hr_data = data["hr"]
    cycles = data["cycles"]

    hr_data_zoned, _ = get_hr_zones(hr_data, cycles)
    data_merge = merge_cadence_with_hr(steps_data[["dateTime", "value"]], hr_data_zoned)
    plot_step_zone_ratios(ax, data_merge)

    ax2 = ax.twinx()
    plot_steps_over_time(ax2, steps_data, label="steps", style={"color": "blue", "linewidth": 1, "linestyle": "-", "alpha": 0.3})
    plot_steps_moving_average(ax2, steps_data, style={"linewidth": 2, "linestyle": "--"})
    plot_info(ax, cycles, steps_data)
    set_legend(ax, ax2)


FIGURE_CONFIG = [
    {
        "key": "hrv_and_steps",
        "plotter": hrv_and_steps,
        "title": "HRV and Steps Over Time",
        "ylabel": "HRV (ms)",
        "filename": "hrv_and_steps",
    },
    {
        "key": "rhr_and_steps",
        "plotter": rhr_and_steps,
        "title": "Resting Heart Rate and Steps Over Time",
        "ylabel": "Resting Heart Rate (bpm)",
        "filename": "rhr_and_steps",
    },
    {
        "key": "hr_and_zones",
        "plotter": hr_and_zones,
        "title": "Heart Rate and Zones Over Time",
        "ylabel": "Heart Rate (bpm)",
        "filename": "hr_and_zones",
    },
    {
        "key": "cadence_and_steps",
        "plotter": cadence_and_steps,
        "title": "Cadence and Steps Over Time",
        "ylabel": "Cadence (steps/min)",
        "filename": "cadence_and_steps",
    },
    {
        "key": "steps_zone_ratios",
        "plotter": steps_zone_ratios,
        "title": "Steps and Zone Ratios Over Time",
        "ylabel": "Steps",
        "filename": "steps_zone_ratios",
    },
    {
        "key": "hr_zone_ratios",
        "plotter": zone_ratios_hr,
        "title": "Heart Rate Zone Ratios Over Time",
        "ylabel": "Heart Rate Zone Ratios",
        "filename": "hr_zone_ratios",
    },
]


def create_figure(data: dict[str, pd.DataFrame], participant_id: int, config: dict[str, Any]) -> plt.Figure:
    """Create a single participant figure for one configured plot type.

    Args:
        data (dict[str, pd.DataFrame]): Participant data payload.
        participant_id (int): Participant identifier used in the title.
        config (dict[str, Any]): Plot configuration for the requested figure.

    Returns:
        plt.Figure: Matplotlib figure with the requested panel.
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    config["plotter"](ax, data)
    ax.set_title(f"Participant {participant_id}: {config['title']}")
    ax.set_ylabel(config["ylabel"])
    ax.set_xlabel("Time (YYYY-MM-DD)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def main(pat_id: int | list[int] | tuple[int, ...]) -> tuple[int, dict[str, plt.Figure]]:
    """Generate all configured figures for one participant.

    Args:
        pat_id (int | list[int] | tuple[int, ...]): Participant ID or IDs to load.

    Returns:
        tuple[int, dict[str, plt.Figure]]: Participant identifier and generated figures.
    """
    requested_id = pat_id[0] if isinstance(pat_id, (list, tuple)) else pat_id
    loaded_ids, data = load_plot_data([requested_id], data_types=["steps", "cycles", "hr", "hrv", "rhr"])
    participant_id = loaded_ids[0] if isinstance(loaded_ids, (list, tuple)) else loaded_ids

    figures = {}
    for cfg in FIGURE_CONFIG:
        figures[cfg["filename"]] = create_figure(data, participant_id, cfg)

    return participant_id, figures


def save_figures(participant_id: int, figures: dict[str, plt.Figure], base_path: str) -> None:
    """Save each figure in a per-participant folder.

    Args:
        participant_id (int): Participant identifier used in the output folder and filename.
        figures (dict[str, plt.Figure]): Generated figure collection.
        base_path (str): Base output directory.

    Returns:
        None: Saves the figures to disk.
    """
    participant_path = os.path.join(base_path, f"PT{participant_id}")
    os.makedirs(participant_path, exist_ok=True)

    for filename, fig in figures.items():
        fig.savefig(os.path.join(participant_path, f"PT{participant_id}_{filename}.png"))


def close_figures(figures: dict[str, plt.Figure]) -> None:
    """Close all open matplotlib figures.

    Args:
        figures (dict[str, plt.Figure]): Mapping of names to figure objects.

    Returns:
        None: Closes each figure handle.
    """
    for fig in figures.values():
        plt.close(fig)


if __name__ == "__main__":
    part_ids = range(1, 31)
    base_path = FIGURE_FOLDER 
    for pt_id in tqdm(part_ids, desc="Plotting participants", unit="participant"):
        try:
            participant_id, figs = main(pt_id)
            save_figures(participant_id, figs, base_path)
            close_figures(figs)
        except ValueError as e:
            print(f"Error plotting participant {pt_id}: {e}")
