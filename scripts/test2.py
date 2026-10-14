import matplotlib.pyplot as plt
import pandas as pd

from preprocessing import Preprocessor
from loaders import ParticipantDataLoader
from plots import plot_hrv_over_time


part_ids = [14,15]
participant_loader = ParticipantDataLoader()
preprocessor = Preprocessor()
hrv_data = preprocessor.process_heart_rate_variability(part_id = part_ids)

cycles = participant_loader.cycle_data(part_id = part_ids)

fig, axes = plt.subplots(2, 1, figsize=(20, 10))
axes = axes.ravel()

for i, part_id in enumerate(part_ids):
    ax = axes[i]
    part_hrv_data = hrv_data[hrv_data['participant_id'] == part_id]
    part_cycles = cycles[cycles['participant_id'] == part_id]

    plot_hrv_over_time(fig, ax, part_hrv_data, part_cycles, part_id)
    ax.set_title(f'Participant {part_id}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Steps')




plt.xticks(rotation=45)
plt.tight_layout()
plt.show()