import matplotlib.pyplot as plt
import pandas as pd

from preprocessing import Preprocessor
from loaders import ParticipantDataLoader
from plots import plot_rhr_over_time, plot_hrv_over_time, plot_steps_over_time

part_ids = [13]  # List of participant IDs to process
participant_loader = ParticipantDataLoader()
preprocessor = Preprocessor()

rhr_data = preprocessor.process_resting_heart_rate(part_id= part_ids)
hrv_data = preprocessor.process_heart_rate_variability(part_id= part_ids)
step_data = preprocessor.process_step_count(part_id= part_ids)
cycles = participant_loader.cycle_data(part_id = part_ids)

n_parts = len(part_ids)
nrows = 2            # two rows: top = HRV plot, bottom = user-defined plot
ncols = n_parts      # one column per participant

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(10 * ncols, 5 * nrows),
    squeeze=False, 
    sharex="col"
)

for i, part_id in enumerate(part_ids):
    ax_top = axes[0, i]
    ax_bottom = axes[1, i]

    part_rhr_data = rhr_data[rhr_data['participant_id'] == part_id]
    part_step_data = step_data[step_data['participant_id'] == part_id]
    part_cycles = cycles[cycles['participant_id'] == part_id]
    part_hrv_data = hrv_data[hrv_data['participant_id'] == part_id]
    
    # Top subplot (HRV + steps)
    plot_rhr_over_time(fig, ax_top, part_rhr_data, part_cycles, "Daily Resting Heart Rate")

    ax_top.set_zorder(2)
    ax2 = ax_top.twinx()
    labeled = False
    for _, day_data in part_step_data.sort_values("dateTime").groupby("date"):
        ax2.plot(
            day_data["dateTime"],
            day_data["cumsum"],
            label=f"Participant {part_ids}" if not labeled else None,
            alpha=0.4,
            color="blue",
        )
        labeled = True
    ax2.plot(part_step_data['date'], part_step_data['daily_steps_ma5'], alpha=0.7, color='blue', label='Steps 5d MA')
    ax2.set_ylabel('Steps')
    handles, labels = ax_top.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    if handles2:
        ax2.legend(handles2, labels2, loc='upper right')
    if handles:
        ax_top.legend(handles, labels, loc='upper left')

    ax_top.grid(True)
    ax_top.set_title(f'Participant {part_id}')
    ax_top.set_xlabel('Date')
    ax_top.set_ylabel('HRV (RMSSD)')

    # Bottom subplot (placeholder for user)
    # fill ax_bottom with your custom plots
    ax_bottom.set_zorder(2)
    ax2_bottom = ax_bottom.twinx()

    plot_hrv_over_time(fig, ax_bottom, part_hrv_data, part_cycles, "Daily HRV (RMSSD)")
    ax2_bottom.plot(part_step_data['date'], part_step_data['daily_steps_ma5'], alpha=0.7, color='blue', label='Steps 5d MA')
    ax2_bottom.plot(part_step_data['date'], part_step_data['daily_steps_ma3'], alpha=0.4, color='tab:cyan', label='Steps 3d MA')
    ax2_bottom.set_ylabel('Steps')

    handles, labels = ax_bottom.get_legend_handles_labels()
    handles2, labels2 = ax2_bottom.get_legend_handles_labels()
    if handles2:
        ax2_bottom.legend(handles2, labels2, loc='upper right')
    if handles:
        ax_bottom.legend(handles, labels, loc='upper left')

    ax_bottom.grid(True)
    ax_bottom.set_title(f'Participant {part_id}')
    ax_bottom.set_xlabel('Date')
    ax_bottom.set_ylabel('HRV (RMSSD)')


plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
