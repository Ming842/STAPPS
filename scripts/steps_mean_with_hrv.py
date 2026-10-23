import matplotlib.pyplot as plt
import pandas as pd

from loaders import ParticipantDataLoader
from plot_defs import (
    create_figure_panel,
    plot_hr_cadence,
    plot_hrv_over_time,
    plot_info,
    plot_rhr,
    plot_step_zone_ratios,
    plot_steps_moving_average,
    plot_steps_over_time,
    plot_zone_ratios,
    set_legend,
)
from preprocessing import Preprocessor, load_plot_data, merge_cadence_with_hr, part_isin_data
from statistics_def import get_hr_zones

def plot_pat_field(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    """Plot HRV and step trends for one participant.

    Args:
        ax (plt.Axes): Axis to draw the plot on.
        data (dict[str, pd.DataFrame]): Dictionary containing participant step, HRV, and cycle data.

    Returns:
        None: Adds HRV and step traces to the axis.
    """
    steps_data = data["steps"]
    hrv_data = data["hrv"]
    cycles = data["cycles"]

    plot_hrv_over_time(ax, hrv_data, label="HRV")
    
    ax2 = ax.twinx()
    plot_steps_over_time(ax2, steps_data, label="steps")
    plot_steps_moving_average(ax2, steps_data, windows=[3, 4, 5, 6])
    plot_info(ax, cycles, steps_data)

    set_legend(ax, ax2)

def plot_pat_field_2(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    """Plot resting heart rate and step trends for one participant.

    Args:
        ax (plt.Axes): Axis to draw the plot on.
        data (dict[str, pd.DataFrame]): Dictionary containing participant step, RHR, and cycle data.

    Returns:
        None: Adds RHR and step traces to the axis.
    """
    steps_data = data["steps"]
    rhr_data = data["rhr"]
    cycles = data["cycles"]

    plot_rhr(ax, rhr_data, label="Resting Heart Rate")
    plot_info(ax, cycles, steps_data)

    ax2 = ax.twinx()
    plot_steps_moving_average(ax2, steps_data,
                              style={"color": "tab:blue",
                                     "linewidth": 2,
                                     "linestyle": "--",
                                     })
    set_legend(ax, ax2)

def plot_pat_field_3(ax: plt.Axes, data: dict[str, pd.DataFrame]) -> None:
    """Plot cadence-by-zone trends and step traces for one participant.

    Args:
        ax (plt.Axes): Axis to draw the plot on.
        data (dict[str, pd.DataFrame]): Dictionary containing participant step, heart-rate, and cycle data.

    Returns:
        None: Adds the cadence and step traces to the axis.
    """
    steps_data = data["steps"]
    hr_data = data["hr"]
    cycles = data["cycles"]

    hr_data_zoned, zone_thresholds = get_hr_zones(hr_data, cycles)
    data_merge = merge_cadence_with_hr(steps_data[["dateTime", "cadence"]], hr_data_zoned)
    plot_hr_cadence(ax, data_merge, 3)

    ax2 = ax.twinx()
    plot_steps_moving_average(ax2, steps_data, style={"color": "tab:blue", "linewidth": 2, "linestyle": "--"})
    plot_info(ax, cycles, steps_data)
    set_legend(ax, ax2)

def main(pat_ids: list[int]) -> None:
    """Create a three-panel summary figure for a list of participant IDs.

    Args:
        pat_ids (list[int]): Participant IDs to plot.

    Returns:
        None: Displays the generated figure and saves it in the default output path.
    """
    pat_ids, data = load_plot_data(pat_ids, data_types=["steps", "hrv", "cycles", "hr", "rhr"])

    _, axes = create_figure_panel(pat_ids, nrows=3)

    for i, pat_id in enumerate(pat_ids):
        data = part_isin_data([pat_id], data)
        ax = axes[0, i]
        plot_pat_field(ax, data)
        ax.set_title(f"Participant {pat_id}")
        ax.set_xlabel("Date")
        ax.set_ylabel("HRV (RMSSD)")

        ax2 = axes[1, i]
        plot_pat_field_2(ax2, data)
        ax2.set_ylabel("Resting Heart Rate (bpm)")

        ax3 = axes[2, i]
        plot_pat_field_3(ax3, data)
        ax3.set_ylabel("Cadence (steps/min)")

if __name__ == "__main__":
    i = 1
    try:
        main([i])
        plt.savefig(rf"\\storage.erasmusmc.nl\v\vcl02\ONCO\Data\01 Onderzoeksbestanden\STAPPS\STAPPS_Main\STAPPS\plots\figures\participant_{i}_plots2.png")
    except Exception as e:
        print(f"Error plotting participant {i}: {e}")


