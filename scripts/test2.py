import matplotlib.pyplot as plt
import pandas as pd

from preprocessing import Preprocessor
from loaders import ParticipantDataLoader
from plots import plot_hrv_over_time


part_ids = [11,12,13,14,16,17,18,20]  # List of participant IDs to process
participant_loader = ParticipantDataLoader()
preprocessor = Preprocessor()
hrv_data = preprocessor.process_heart_rate_variability(part_id = part_ids)
steps_data = preprocessor.process_step_count(part_id = part_ids)

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
    part_hrv_data = hrv_data[hrv_data['participant_id'] == part_id]
    part_hrv_data[part_hrv_data["rmssd"] == 0] = pd.NA
    part_cycles = cycles[cycles['participant_id'] == part_id]

    plot_hrv_over_time(fig, ax, part_hrv_data, part_cycles, part_id)
    ax.set_zorder(2)
    ax2 = ax.twinx()
    part_steps_data = steps_data[steps_data['participant_id'] == part_id]
    ax2.plot(part_steps_data['date'], part_steps_data['daily_steps_ma5'], alpha=0.7, color='blue')
    # ax2.plot(part_steps_data['date'], part_steps_data['daily_steps_ma3'], alpha=0.4, color='blue')
    
    ax2.set_ylabel('Steps')
    # show legend for primary and secondary axes
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    if handles:
        ax.legend(handles, labels, loc='upper left')
    if handles2:
        ax2.legend(handles2, labels2, loc='upper right')

    plt.legend(loc='upper left')
    plt.grid()
    ax.set_title(f'Participant {part_id}')
    ax.set_xlabel('Date')
    ax.set_ylabel('HRV (RMSSD)')




plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
