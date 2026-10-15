import matplotlib.pyplot as plt
import pandas as pd

from preprocessing import Preprocessor
from loaders import ParticipantDataLoader
from plots import plot_rhr_over_time

part_ids = [1]  # List of participant IDs to process
participant_loader = ParticipantDataLoader()
preprocessor = Preprocessor()

rhr_data = preprocessor.process_resting_heart_rate(part_id= part_ids)
step_data = preprocessor.process_step_count(part_id= part_ids)
cycles = participant_loader.cycle_data(part_id = part_ids)

n_parts = len(part_ids)
nrows = 1 if n_parts == 1 else 2
ncols = (n_parts + nrows - 1) // nrows  # fill columns with up to 2 rows

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(10 * ncols, 5 * nrows),
    squeeze=False
)
axes = axes.ravel()

# hide any unused subplot axes (when n_parts is odd)
for ax in axes[n_parts:]:
    ax.set_visible(False)

for i, part_id in enumerate(part_ids):
    ax = axes[i]
    part_rhr_data = rhr_data[rhr_data['participant_id'] == part_id]
    part_step_data = step_data[step_data['participant_id'] == part_id]

    part_cycles = cycles[cycles['participant_id'] == part_id]
    
    plot_rhr_over_time(fig, ax, part_rhr_data, part_cycles, "Daily Resting Heart Rate")
    
    ax.set_zorder(2)
    ax2 = ax.twinx()
    ax2.plot(part_step_data['date'], part_step_data['daily_steps_ma5'], alpha=0.7, color='blue', label='Steps 5d MA')
    ax2.plot(part_step_data['date'], part_step_data['daily_steps_ma3'], alpha=0.4, color='tab:cyan', label='Steps 3d MA')
    
    ax2.set_ylabel('Steps')

    # show legend for primary and secondary axes
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    if handles2:
        ax2.legend(handles2, labels2, loc='upper right')
    if handles:
        ax.legend(handles, labels, loc='upper left')
    plt.grid()
    ax.set_title(f'Participant {part_id}')
    ax.set_xlabel('Date')
    ax.set_ylabel('HRV (RMSSD)')




plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
