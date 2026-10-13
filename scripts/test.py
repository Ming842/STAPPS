"""Test data and configuration for the STAPPS."""
import matplotlib.pyplot as plt
import pandas as pd
from preprocessing import Preprocessor
from loaders import ParticipantDataLoader

part_ids = None

participant_loader = ParticipantDataLoader()
preprocessor = Preprocessor()
steps_data = preprocessor.process_step_count(part_id = part_ids)
cycles = participant_loader.cycle_data()
# cycles = cycles[cycles['participant_id'] == part_ids]
# cumsum per day of steps per participant
steps_data['date'] = pd.to_datetime(steps_data['dateTime']).dt.date
steps_data['cumsum'] = steps_data.groupby(['participant_id', 'date'])['value'].cumsum()

plt.figure(figsize=(12, 6))
for i, (_, day_data) in enumerate(steps_data.sort_values('dateTime').groupby('date')):
    plt.plot(
        day_data['dateTime'],
        day_data['cumsum'],
        label=f'Participant {part_ids}' if i == 0 else None,
        color='blue'
    )
for date, period in zip(cycles['Date'], cycles['Period']):
    if period == "Baseline":
        continue
    if date == "Dagelijks":
        continue
    plt.axvline(x=date, color='red', linestyle='--', alpha=0.5,
                label=str(period))
    plt.text(date, 0.8, f"Cycle: {period}",
             transform=plt.gca().get_xaxis_transform(),
             rotation=90, va='bottom', ha='right', color='red')
plt.legend()
plt.title('Step Count Over Time')
plt.xlabel('DateTime')
plt.ylabel('Step Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()