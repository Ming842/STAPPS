import matplotlib.pyplot as plt
import pandas as pd

from preprocessing import Preprocessor
from loaders import ParticipantDataLoader

from statistics_def import get_hr_zones
from plots import (load_plot_data, 
                   create_figure_panel, 
                   plot_steps_moving_average, 
                   set_legend, 
                   plot_hrv_over_time, 
                   plot_info, 
                   plot_steps_over_time, 
                   plot_hr_cadence)

def plot_pat_field(ax, pat_ids, steps, hrv, cycles):
    steps_data = steps[steps['participant_id'].isin(pat_ids)]
    hrv_data = hrv[hrv['participant_id'].isin(pat_ids)]
    cycles = cycles[cycles['participant_id'].isin(pat_ids)]

    plot_hrv_over_time(ax, hrv_data, label="HRV")

    ax2 = ax.twinx()
    plot_steps_over_time(ax2, steps_data, label="steps")
    plot_steps_moving_average(ax2, steps_data, windows=[3,4,5,6])
    plot_info(ax, cycles, steps_data)

    set_legend(ax, ax2)

def plot_pat_field_2(ax, pat_id, data):
    preprocessor =  Preprocessor()
    part_steps_data = data["steps"][data["steps"]['participant_id'].isin(pat_id)]
    part_hr_data = data["hr"][data["hr"]['participant_id'].isin(pat_id)]
    part_cycles = data["cycles"][data["cycles"]['participant_id'].isin(pat_id)]
    hr_data, _ = get_hr_zones(part_hr_data, part_cycles)
    data_merge = preprocessor.merge_cadence_with_hr(part_steps_data[["dateTime", "cadence"]], hr_data)
    plot_hr_cadence(ax, data_merge, 3)
    ax2 = ax.twinx()
    plot_steps_moving_average(ax2, part_steps_data)
    plot_info(ax, part_cycles, part_steps_data)

    set_legend(ax, ax2)

def main():
    pat_ids = [14]  # List of participant IDs to process

    pat_ids, data = load_plot_data(pat_ids, data_types=["steps", "hrv", "cycles", "hr"])

    _, axes = create_figure_panel(pat_ids)

    for i, pat_id in enumerate(pat_ids):
        ax = axes[0,i]
        plot_pat_field(ax, [pat_id], data["steps"], data["hrv"], data["cycles"])
        ax.set_title(f"Participant {pat_id}")
        ax.set_xlabel("Date")
        ax.set_ylabel("HRV (RMSSD)")

        ax2 = axes[1,i]
        plot_pat_field_2(ax2, [pat_id], data)
        ax2.set_ylabel("Cadence (steps/min)")

        
if __name__ == "__main__":
    main()
    plt.show()
