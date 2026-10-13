"""Test data and configuration for the STAPPS."""
import matplotlib.pyplot as plt
import pandas as pd
from preprocessing import Preprocessor
from loaders import ParticipantDataLoader

participant_loader = ParticipantDataLoader()
preprocessor = Preprocessor()
# sleep_data = preprocessor.process_sleep_score()
steps_data = preprocessor.process_step_count()
heart_rate_data = preprocessor.process_heart_rate()
cycles = participant_loader.cycle_data()

# cumsum per day of steps per participant
steps_data['date'] = pd.to_datetime(steps_data['dateTime']).dt.date
steps_data['cumsum'] = steps_data.groupby(['participant_id', 'date'])['value'].cumsum()
part_id = 1
steps_data_pt = steps_data[steps_data['participant_id'] == part_id]
heart_rate_data_pt = heart_rate_data[heart_rate_data['participant_id'] == part_id]
cycles_pt = cycles[cycles['participant_id'] == part_id]

print(cycles_pt['Date'])
plt.figure(figsize=(12, 6))
plt.plot(steps_data_pt['dateTime'], steps_data_pt['cumsum'], label=f'Participant {part_id}', color='blue')
for date, period in zip(cycles_pt['Date'], cycles_pt['Period']):
    if period == "Baseline":
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