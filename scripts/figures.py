import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from preprocessing import Preprocessor, merge_cadence_with_hr, part_isin_data, load_plot_data
from loaders import ParticipantDataLoader

from statistics_def import get_hr_zones
import os
from tqdm import tqdm
from plot_defs import ( 
                   create_figure_panel,
                   plot_hr_over_time,
                   plot_hr_zones,
                   plot_rhr,
                   plot_step_zone_ratios, 
                   plot_steps_moving_average, 
                   plot_zone_ratios, 
                   set_legend, 
                   plot_hrv_over_time, 
                   plot_info, 
                   plot_steps_over_time, 
                   plot_hr_cadence)

def hrv_and_steps(ax, data):
    steps_data = data["steps"]
    hrv_data = data["hrv"]
    cycles = data["cycles"]

    plot_hrv_over_time(ax, hrv_data, "HRV",3, style={"color": "orange", 
                                                           "linewidth": 2, 
                                                           "linestyle": "-", 
                                                           "alpha": 1.0, 
                                                           "marker": "o",
                                                           "markersize": 4})
    
    ax2 = ax.twinx()
    plot_steps_over_time(ax2, steps_data, label="steps", style={"color": "blue", "linewidth": 1, "linestyle": "-", "alpha": 0.3})
    plot_steps_moving_average(ax2, steps_data, 
                              style = {"linewidth": 2, 
                                       "linestyle": "--",
                                       })
    plot_info(ax, cycles, steps_data)

    set_legend(ax, ax2)

def rhr_and_steps(ax, data):
    steps_data = data["steps"]
    rhr_data = data["rhr"]
    cycles = data["cycles"]

    plot_rhr(ax, rhr_data, label="Resting Heart Rate")
    plot_info(ax, cycles, steps_data)

    ax2 = ax.twinx()
    plot_steps_over_time(ax2, steps_data, label="steps",style={"color": "blue", "linewidth": 1, "linestyle": "-", "alpha": 0.3})
    plot_steps_moving_average(ax2, steps_data, 
                              style = {"linewidth": 2, 
                                       "linestyle": "--",
                                       })
    set_legend(ax, ax2)

def hr_and_zones(ax, data):
    steps_data = data["steps"]
    hr_data = data["hr"]
    cycles = data["cycles"]

    hr_data_zoned, zone_thresholds = get_hr_zones(hr_data, cycles)
    plot_hr_zones(ax, hr_data_zoned, zone_thresholds)
    plot_hr_over_time(ax, hr_data, label="Heart Rate", style = {"linewidth": 1, "linestyle": "-", "alpha": 1}) 
    plot_info(ax, cycles, steps_data)

    set_legend(ax)

def cadence_and_steps(ax, data):
    steps_data = data["steps"]
    hr_data = data["hr"]
    cycles = data["cycles"]

    hr_data_zoned, zone_thresholds = get_hr_zones(hr_data, cycles)
    data_merge = merge_cadence_with_hr(steps_data[["dateTime", "cadence"]], hr_data_zoned)
    plot_hr_cadence(ax, data_merge, 3)

    ax2 = ax.twinx()
    plot_steps_over_time(ax2, steps_data, label="steps",style={"color": "blue", "linewidth": 1, "linestyle": "-", "alpha": 0.3})
    plot_steps_moving_average(ax2, steps_data, style = {"linewidth": 2, "linestyle": "--"})
    plot_info(ax, cycles, steps_data)
    set_legend(ax, ax2)

def zone_ratios_hr(ax, data):
    steps_data = data["steps"]
    hr_data = data["hr"]
    cycles = data["cycles"]

    hr_data_zoned, _ = get_hr_zones(hr_data, cycles)
    # data_merge = merge_cadence_with_hr(steps_data[["dateTime", "cadence"]], hr_data_zoned)
    plot_zone_ratios(ax, hr_data_zoned)
    plot_info(ax, cycles, steps_data)

    ax2 = ax.twinx()
    plot_steps_moving_average(ax2, steps_data, style = {"linewidth": 2, "linestyle": "--"})
    set_legend(ax, ax2)

def steps_zone_ratios(ax, data):
    steps_data = data["steps"]
    hr_data = data["hr"]
    cycles = data["cycles"]

    hr_data_zoned, _ = get_hr_zones(hr_data, cycles)
    data_merge = merge_cadence_with_hr(steps_data[["dateTime", "value"]], hr_data_zoned)
    plot_step_zone_ratios(ax, data_merge)
   
    ax2 = ax.twinx()
    plot_steps_over_time(ax2, steps_data, label="steps",style={"color": "blue", "linewidth": 1, "linestyle": "-", "alpha": 0.3})
    plot_steps_moving_average(ax2, steps_data, style = {"linewidth": 2, "linestyle": "--"})
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


def create_figure(data, participant_id, config):
    fig, ax = plt.subplots(figsize=(15, 8))
    config["plotter"](ax, data)
    ax.set_title(f"Participant {participant_id}: {config['title']}")
    ax.set_ylabel(config["ylabel"])
    ax.set_xlabel("Time (YYYY-MM-DD)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def main(pat_id):
    requested_id = pat_id[0] if isinstance(pat_id, (list, tuple)) else pat_id
    loaded_ids, data = load_plot_data([requested_id], data_types=["steps", "cycles", "hr", "hrv", "rhr"])
    participant_id = loaded_ids[0] if isinstance(loaded_ids, (list, tuple)) else loaded_ids

    figures = {}
    for cfg in FIGURE_CONFIG:
        figures[cfg["filename"]] = create_figure(data, participant_id, cfg)

    return participant_id, figures


def save_figures(participant_id, figures, base_path):
    participant_path = os.path.join(base_path, f"PT{participant_id}")
    os.makedirs(participant_path, exist_ok=True)

    for filename, fig in figures.items():
        fig.savefig(os.path.join(participant_path, f"PT{participant_id}_{filename}.png"))


def close_figures(figures):
    for fig in figures.values():
        plt.close(fig)


if __name__ == "__main__":
    part_ids = range(1, 31)
    base_path = r"\\storage.erasmusmc.nl\v\vcl02\ONCO\Data\01 Onderzoeksbestanden\STAPPS\STAPPS_Main\STAPPS\plots\figures"

    for pt_id in tqdm(part_ids, desc="Plotting participants", unit="participant"):
        try:
            participant_id, figs = main(pt_id)
            save_figures(participant_id, figs, base_path)
            close_figures(figs)
        except ValueError as e:
            print(f"Error plotting participant {pt_id}: {e}")
