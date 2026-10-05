"""Test data and configuration for the STAPPS."""
import matplotlib.pyplot as plt
import pandas as pd
from preprocessing import Preprocessor

preprocessor = Preprocessor()
# sleep_data = preprocessor.process_sleep_score()
steps_data = preprocessor.process_step_count()

# cumsum per day of steps per participant
steps_data['date'] = pd.to_datetime(steps_data['dateTime']).dt.date
steps_data['cumsum'] = steps_data.groupby(['participant_id', 'date'])['value'].cumsum()

plt.figure(figsize=(12, 6))
for j,part_id in enumerate(steps_data['participant_id'].unique()):
    if part_id == 1:
        participant_data = steps_data[steps_data['participant_id'] == part_id]
        for i, (date, day_data) in enumerate(participant_data.groupby('date')):
            time_series = day_data['dateTime'] - day_data['dateTime'].min() + pd.Timedelta(days=i)
            plt.plot(
                time_series,
                pd.to_numeric(day_data['cumsum'], errors='coerce'),
                linestyle='-',
                # alpha=1/45 * (j + 1),
                label=f'Participant {part_id}, {date}',
            )
plt.title('Step Count Over Time')
plt.xlabel('DateTime')
plt.ylabel('Step Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()