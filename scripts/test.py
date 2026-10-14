"""Test data and configuration for the STAPPS."""

import matplotlib.pyplot as plt
import pandas as pd

from preprocessing import Preprocessor
from loaders import ParticipantDataLoader

from plots import plot_steps_over_time


part_ids = [14,15]
participant_loader = ParticipantDataLoader()
preprocessor = Preprocessor()
steps_data = preprocessor.process_step_count(part_id = part_ids)
cycles = participant_loader.cycle_data(part_id = part_ids)

if part_ids is not None:
    cycles = cycles[cycles['participant_id'].isin(part_ids)]
# cumsum per day of steps per participant


fig, axes = plt.subplots(1, 2, figsize=(20, 10))
axes = axes.ravel()

for i, part_id in enumerate(part_ids):
    ax = axes[i]
    part_steps_data = steps_data[steps_data['participant_id'] == part_id]
    part_cycles = cycles[cycles['participant_id'] == part_id]

    plot_steps_over_time(fig, ax, part_steps_data, part_cycles, part_id)
    ax.set_title(f'Participant {part_id}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Steps')



plt.xticks(rotation=45)
plt.tight_layout()
plt.show()